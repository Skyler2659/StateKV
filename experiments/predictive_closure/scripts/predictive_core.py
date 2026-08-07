"""Core numerical objects for the preregistered predictive-closure experiment."""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.integrate import quad
from transformers import AutoModelForCausalLM, AutoTokenizer


TensorTuple = Tuple[torch.Tensor, ...]


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential)


def exact_kl(full_logits: np.ndarray, changed_logits: np.ndarray) -> float:
    z0 = np.asarray(full_logits, dtype=np.float64)
    z1 = np.asarray(changed_logits, dtype=np.float64)
    p = stable_softmax(z0)
    log_p = z0 - (np.max(z0) + np.log(np.exp(z0 - np.max(z0)).sum()))
    log_q = z1 - (np.max(z1) + np.log(np.exp(z1 - np.max(z1)).sum()))
    return float(np.sum(p * (log_p - log_q)))


def fisher_variance(probability: np.ndarray, direction: np.ndarray) -> float:
    p = np.asarray(probability, dtype=np.float64)
    value = np.asarray(direction, dtype=np.float64)
    mean = float(np.dot(p, value))
    return float(np.dot(p, value * value) - mean * mean)


def fisher_score(logits: np.ndarray, direction: np.ndarray, midpoint: bool) -> float:
    z = np.asarray(logits, dtype=np.float64)
    w = np.asarray(direction, dtype=np.float64)
    probability = stable_softmax(z + (0.5 * w if midpoint else 0.0))
    return max(0.0, 0.5 * fisher_variance(probability, w))


def adaptive_path_fisher(
    full_logits: np.ndarray,
    delta_logits: np.ndarray,
    rtol: float = 1.0e-8,
    atol: float = 1.0e-10,
    limit: int = 50,
) -> Dict[str, float]:
    z = np.asarray(full_logits, dtype=np.float64)
    delta = np.asarray(delta_logits, dtype=np.float64)

    def integrand(point: float) -> float:
        probability = stable_softmax(z + float(point) * delta)
        return (1.0 - float(point)) * fisher_variance(probability, delta)

    value, integration_error = quad(
        integrand, 0.0, 1.0, epsrel=rtol, epsabs=atol, limit=limit
    )
    endpoint = z + delta
    truth = exact_kl(z, endpoint)
    return {
        "path_fisher": float(value),
        "exact_kl": truth,
        "absolute_error": abs(float(value) - truth),
        "relative_error": abs(float(value) - truth) / max(abs(truth), 1.0e-30),
        "integration_error_estimate": float(integration_error),
    }


def cosine(left: torch.Tensor, right: torch.Tensor, eps: float = 1.0e-30) -> float:
    a = left.detach().double().reshape(-1).cpu()
    b = right.detach().double().reshape(-1).cpu()
    denominator = max(float(a.norm() * b.norm()), eps)
    return float(torch.dot(a, b) / denominator)


def relative_l2(
    predicted: torch.Tensor, truth: torch.Tensor, eps: float = 1.0e-30
) -> float:
    difference = (predicted.detach().double() - truth.detach().double()).norm()
    return float(difference / max(float(truth.detach().double().norm()), eps))


def normalized_regret(score: np.ndarray, truth: np.ndarray) -> float:
    score = np.asarray(score, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    selected = int(np.nanargmin(score))
    return float(
        (truth[selected] - np.nanmin(truth))
        / max(float(np.nanmax(truth) - np.nanmin(truth)), 1.0e-30)
    )


@dataclass(frozen=True)
class CandidateMask:
    candidate_id: str
    source: str
    keep_prefix_rows: Tuple[int, ...]
    seed: int
    metadata: Mapping[str, Any]

    @property
    def mask_hash(self) -> str:
        text = json.dumps(list(self.keep_prefix_rows), separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class ReferenceQuery:
    prefix_length: int
    query_token_id: int
    prefix_cache: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
    full_cache_with_query: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
    logits: torch.Tensor
    attentions: Tuple[torch.Tensor, ...]
    projected_attention: Dict[int, torch.Tensor]


class InterventionController:
    """Inject a tuple of layer blocks after o_proj and before residual addition."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.current: Optional[TensorTuple] = None
        self.capture: bool = False
        self.projected_attention: Dict[int, torch.Tensor] = {}
        self.handles = []
        for layer_index, layer in enumerate(model.model.layers):
            self.handles.append(
                layer.self_attn.register_forward_hook(
                    self._hook(layer_index)
                )
            )

    def _hook(self, layer_index: int):
        def hook(_module: Any, _inputs: Tuple[Any, ...], output: Tuple[Any, ...]):
            projected = output[0]
            if self.capture:
                self.projected_attention[layer_index] = (
                    projected.detach().float().cpu().clone()
                )
            if self.current is None:
                return output
            injection = self.current[layer_index]
            if injection.ndim == 1:
                injection = injection.view(1, 1, -1)
            injection = injection.to(device=projected.device, dtype=projected.dtype)
            if projected.shape[-2] != 1:
                raise RuntimeError(
                    "interventions are defined only for one-token replay"
                )
            changed = projected + injection
            return (changed,) + tuple(output[1:])

        return hook

    @contextlib.contextmanager
    def use(
        self, interventions: Optional[TensorTuple], capture: bool = False
    ):
        if self.current is not None:
            raise RuntimeError("nested intervention contexts are forbidden")
        self.current = interventions
        self.capture = bool(capture)
        if capture:
            self.projected_attention = {}
        try:
            yield self
        finally:
            self.current = None
            self.capture = False

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class DifferentiableQwen:
    """One-token full, physical-mask, and additive-intervention replay."""

    def __init__(self, checkpoint: Path, device: str, seed: int):
        set_determinism(seed)
        if device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.float16,
            local_files_only=True,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        ).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.config.use_cache = True
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, local_files_only=True, trust_remote_code=False
        )
        self.controller = InterventionController(self.model)
        self.layers = int(self.model.config.num_hidden_layers)
        self.hidden_size = int(self.model.config.hidden_size)
        self.query_heads = int(self.model.config.num_attention_heads)
        self.kv_heads = int(self.model.config.num_key_value_heads)
        self.head_dim = self.hidden_size // self.query_heads
        self.group_size = self.query_heads // self.kv_heads

    def close(self) -> None:
        self.controller.close()
        del self.model
        if self.device.type == "mps":
            torch.mps.empty_cache()

    def encode_prompt(self, prompt: str) -> List[int]:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return [
            int(value)
            for value in self.tokenizer.encode(text, add_special_tokens=True)
        ]

    @torch.no_grad()
    def prefill(self, prefix_ids: Sequence[int]) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        inputs = torch.tensor([list(prefix_ids)], dtype=torch.long, device=self.device)
        output = self.model(
            input_ids=inputs,
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
        return tuple(
            (key.detach(), value.detach())
            for key, value in output.past_key_values
        )

    def _forward_query(
        self,
        prefix_cache: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        query_token_id: int,
        logical_position: int,
        interventions: Optional[TensorTuple] = None,
        output_attentions: bool = False,
        capture_projected: bool = False,
    ) -> Any:
        token = torch.tensor(
            [[int(query_token_id)]], dtype=torch.long, device=self.device
        )
        position = torch.tensor(
            [[int(logical_position)]], dtype=torch.long, device=self.device
        )
        cache_position = torch.tensor(
            [int(logical_position)], dtype=torch.long, device=self.device
        )
        with self.controller.use(interventions, capture=capture_projected):
            return self.model(
                input_ids=token,
                past_key_values=prefix_cache,
                position_ids=position,
                cache_position=cache_position,
                use_cache=True,
                output_attentions=output_attentions,
                return_dict=True,
            )

    @torch.no_grad()
    def reference_query(self, token_ids: Sequence[int]) -> ReferenceQuery:
        if len(token_ids) < 2:
            raise ValueError("reference query needs a prefix and query token")
        prefix = list(token_ids[:-1])
        query = int(token_ids[-1])
        cache = self.prefill(prefix)
        output = self._forward_query(
            cache,
            query,
            logical_position=len(prefix),
            output_attentions=True,
            capture_projected=True,
        )
        return ReferenceQuery(
            prefix_length=len(prefix),
            query_token_id=query,
            prefix_cache=cache,
            full_cache_with_query=tuple(
                (key.detach(), value.detach())
                for key, value in output.past_key_values
            ),
            logits=output.logits[0, -1].detach(),
            attentions=tuple(value.detach() for value in output.attentions),
            projected_attention=dict(self.controller.projected_attention),
        )

    def zero_interventions(self) -> TensorTuple:
        return tuple(
            torch.zeros(
                (1, 1, self.hidden_size),
                dtype=torch.float16,
                device=self.device,
            )
            for _ in range(self.layers)
        )

    def intervention_logits(
        self,
        reference: ReferenceQuery,
        interventions: TensorTuple,
    ) -> torch.Tensor:
        output = self._forward_query(
            reference.prefix_cache,
            reference.query_token_id,
            reference.prefix_length,
            interventions=interventions,
            output_attentions=False,
        )
        return output.logits[0, -1]

    @torch.no_grad()
    def physical_logits(
        self,
        reference: ReferenceQuery,
        keep_prefix_rows: Sequence[int],
        masked_layers: Optional[Iterable[int]] = None,
    ) -> torch.Tensor:
        chosen_layers = (
            set(range(self.layers))
            if masked_layers is None
            else set(int(value) for value in masked_layers)
        )
        rows_cpu = torch.tensor(list(keep_prefix_rows), dtype=torch.long)
        compacted = []
        for layer, (key, value) in enumerate(reference.prefix_cache):
            if layer in chosen_layers:
                rows = rows_cpu.to(key.device)
                compacted.append(
                    (
                        key.index_select(2, rows),
                        value.index_select(2, rows.to(value.device)),
                    )
                )
            else:
                compacted.append((key, value))
        output = self._forward_query(
            tuple(compacted),
            reference.query_token_id,
            reference.prefix_length,
            output_attentions=False,
        )
        return output.logits[0, -1].detach()

    def jvp(
        self,
        reference: ReferenceQuery,
        direction: TensorTuple,
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        zeros = self.zero_interventions()

        def mapping(*blocks: torch.Tensor) -> torch.Tensor:
            return self.intervention_logits(reference, tuple(blocks))

        try:
            primal, tangent = torch.func.jvp(mapping, zeros, direction)
            return primal, tangent, "torch.func.jvp"
        except (NotImplementedError, RuntimeError) as first_error:
            # This is still a true autograd JVP, but uses the double-backward
            # implementation.  The fallback is recorded row-wise.
            try:
                primal, tangent = torch.autograd.functional.jvp(
                    mapping,
                    zeros,
                    direction,
                    create_graph=False,
                    strict=True,
                )
                return primal, tangent, "torch.autograd.functional.jvp"
            except Exception as second_error:
                raise RuntimeError(
                    "both JVP implementations failed; "
                    f"forward-mode={first_error}; double-backward={second_error}"
                ) from second_error

    def vjp(
        self, reference: ReferenceQuery, cotangent: torch.Tensor
    ) -> Tuple[torch.Tensor, TensorTuple]:
        zeros = self.zero_interventions()

        def mapping(*blocks: torch.Tensor) -> torch.Tensor:
            return self.intervention_logits(reference, tuple(blocks))

        output, pullback = torch.func.vjp(mapping, *zeros)
        return output, tuple(pullback(cotangent))


def direct_injections(
    model: DifferentiableQwen,
    reference: ReferenceQuery,
    keep_prefix_rows: Sequence[int],
    arithmetic_dtype: torch.dtype = torch.float32,
) -> Tuple[TensorTuple, List[Dict[str, Any]]]:
    """Compute direct U(C) and closed-form identity diagnostics."""
    keep_prefix = list(int(value) for value in keep_prefix_rows)
    keep_with_current = keep_prefix + [reference.prefix_length]
    injections: List[torch.Tensor] = []
    identity_rows: List[Dict[str, Any]] = []
    for layer in range(model.layers):
        attention = (
            reference.attentions[layer][0, :, -1, :]
            .to(dtype=arithmetic_dtype)
        )
        values = (
            reference.full_cache_with_query[layer][1][0]
            .to(dtype=arithmetic_dtype)
        )
        repeated_values = values.repeat_interleave(model.group_size, dim=0)
        rows = torch.tensor(
            keep_with_current, dtype=torch.long, device=attention.device
        )
        kept_attention = attention.index_select(1, rows)
        denominator = kept_attention.sum(dim=1)
        if bool((denominator <= 0).any()):
            raise FloatingPointError("candidate retained zero attention mass")
        full_head = torch.einsum("hn,hnd->hd", attention, repeated_values)
        kept_head = torch.einsum(
            "hk,hkd->hd",
            kept_attention,
            repeated_values.index_select(1, rows),
        ) / denominator[:, None]
        direct_head = kept_head - full_head

        deleted_mask = torch.ones(
            attention.shape[1], dtype=torch.bool, device=attention.device
        )
        deleted_mask[rows] = False
        deleted_attention = attention[:, deleted_mask]
        deleted_values = repeated_values[:, deleted_mask, :]
        deleted_mass = deleted_attention.sum(dim=1)
        closed = torch.einsum(
            "hk,hkd->hd",
            deleted_attention,
            full_head[:, None, :] - deleted_values,
        ) / (1.0 - deleted_mass)[:, None]

        weight = model.model.model.layers[layer].self_attn.o_proj.weight
        projected = F.linear(
            direct_head.reshape(1, -1).to(weight.dtype),
            weight,
            bias=None,
        ).reshape(1, 1, -1)
        injections.append(projected)

        for head in range(model.query_heads):
            difference = direct_head[head].float() - closed[head].float()
            truth_norm = float(direct_head[head].float().norm())
            absolute = float(difference.abs().max())
            relative = float(difference.norm()) / max(truth_norm, 1.0e-30)
            identity_rows.append(
                {
                    "layer": layer,
                    "query_head": head,
                    "kv_head": head // model.group_size,
                    "dtype": str(arithmetic_dtype).replace("torch.", ""),
                    "maximum_absolute_error": absolute,
                    "relative_error": relative,
                    "denominator": float(1.0 - deleted_mass[head]),
                    "deleted_attention_mass": float(deleted_mass[head]),
                    "finite": bool(
                        torch.isfinite(direct_head[head]).all()
                        and torch.isfinite(closed[head]).all()
                    ),
                }
            )
    return tuple(injections), identity_rows


def euclidean_direct_energy(interventions: TensorTuple) -> float:
    return float(
        sum(
            block.detach().double().square().sum().cpu().item()
            for block in interventions
        )
    )


def make_smoke_candidates(
    model: DifferentiableQwen,
    reference: ReferenceQuery,
    total_budget: int,
    sink_size: int,
    recent_size: int,
    core_size: int,
    seed: int,
) -> List[CandidateMask]:
    prefix_length = reference.prefix_length
    recent_prefix = max(0, int(recent_size) - 1)
    sink = list(range(min(sink_size, prefix_length)))
    recent = list(range(max(0, prefix_length - recent_prefix), prefix_length))
    mandatory = set(sink + recent)
    eligible = [
        value for value in range(prefix_length) if value not in mandatory
    ]
    if len(eligible) < core_size:
        raise ValueError("smoke prompt does not have enough eligible core rows")
    score = torch.zeros(prefix_length, dtype=torch.float64)
    for attention in reference.attentions:
        score += attention[0, :, -1, :prefix_length].double().mean(dim=0).cpu()
    ranked = sorted(eligible, key=lambda row: (-float(score[row]), row))
    attention_core = ranked[:core_size]
    old_core = eligible[:core_size]
    candidates = [
        CandidateMask(
            "smoke_attention",
            "attention_only",
            tuple(sorted(mandatory | set(attention_core))),
            seed,
            {"core_size": core_size},
        ),
        CandidateMask(
            "smoke_old",
            "old_stale_core",
            tuple(sorted(mandatory | set(old_core))),
            seed + 1,
            {"core_size": core_size},
        ),
    ]
    expected_prefix_budget = total_budget - 1
    for candidate in candidates:
        if len(candidate.keep_prefix_rows) != expected_prefix_budget:
            raise RuntimeError(
                f"candidate prefix budget {len(candidate.keep_prefix_rows)} "
                f"!= {expected_prefix_budget}"
            )
    if len({value.mask_hash for value in candidates}) != len(candidates):
        raise RuntimeError("smoke candidates are not physically distinct")
    return candidates

