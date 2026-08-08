"""Training-free state sketches and direct retained-set optimization.

This module is intentionally provisional research instrumentation.  It turns
the exact retained-set attention response into a fixed random sketch, carries
that sketch across cache interventions, and searches retained sets without
running a portfolio of external cache selectors.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import torch

from statekv.core.actions import set_level_attention_delta


@dataclass(frozen=True)
class DirectActionDecision:
    """Result of switching-regularized direct retained-set search."""

    retained_positions: Tuple[int, ...]
    shortlist_positions: Tuple[int, ...]
    action_risk: float
    switch_cost: float
    objective: float
    swaps: int
    refreshed: bool
    action_signature: torch.Tensor
    single_deletion_risks: Tuple[Tuple[int, float], ...]


def fixed_rademacher_projection(
    input_width: int,
    sketch_width: int,
    *,
    seed: int,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Create a deterministic Johnson--Lindenstrauss sign projection."""

    if int(input_width) < 1 or int(sketch_width) < 1:
        raise ValueError("projection widths must be positive")
    target = torch.device("cpu") if device is None else torch.device(device)
    # Use the CPU generator for deterministic parity on CPU, CUDA, and MPS,
    # then move the fixed matrix once to the requested execution device.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    signs = torch.randint(
        0,
        2,
        (int(input_width), int(sketch_width)),
        generator=generator,
        dtype=torch.int64,
        device="cpu",
    )
    return (2 * signs - 1).to(dtype=dtype, device=target) / math.sqrt(
        float(sketch_width)
    )


def project_action_response(
    response: torch.Tensor,
    *,
    output_projection: Optional[torch.Tensor] = None,
    sketch_projection: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Flatten and project a local attention response into sketch space.

    Projection matrices use ``[input_width, output_width]`` orientation.  A
    multi-head response is flattened before applying the model output
    projection, matching concatenation before ``W_O``.
    """

    if not torch.isfinite(response).all():
        raise ValueError("action response must be finite")
    vector = response.reshape(-1)
    for name, projection in (
        ("output", output_projection),
        ("sketch", sketch_projection),
    ):
        if projection is None:
            continue
        if projection.ndim != 2 or int(projection.shape[0]) != int(vector.numel()):
            raise ValueError(
                "%s projection must have shape [current_width, projected_width]"
                % name
            )
        vector = vector.to(dtype=projection.dtype, device=projection.device)
        vector = vector @ projection
    return vector


def retained_action_signature(
    attention: torch.Tensor,
    values: torch.Tensor,
    retained_positions: Iterable[int],
    *,
    output_projection: Optional[torch.Tensor] = None,
    sketch_projection: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return the projected exact response of a retained-set action."""

    response = set_level_attention_delta(
        attention, values, retained_positions
    )
    return project_action_response(
        response,
        output_projection=output_projection,
        sketch_projection=sketch_projection,
    )


def query_continuity_decay(
    query: torch.Tensor,
    previous_query: torch.Tensor,
) -> float:
    """Use clipped query cosine as a parameter-free state-retention factor."""

    if query.shape != previous_query.shape:
        raise ValueError("queries must have equal shape")
    left = query.reshape(-1).double()
    right = previous_query.reshape(-1).double()
    denominator = float(left.norm().item() * right.norm().item())
    if denominator <= 0.0:
        return 0.0
    cosine = float(torch.dot(left, right).item() / denominator)
    return float(min(1.0, max(0.0, cosine)))


def update_state_sketch(
    previous_state: torch.Tensor,
    action_signature: torch.Tensor,
    decay: float,
) -> torch.Tensor:
    """Apply ``m_next = decay * m + phi(action)`` without learned weights."""

    if previous_state.shape != action_signature.shape:
        raise ValueError("state and action signature must have equal shape")
    if not 0.0 <= float(decay) <= 1.0:
        raise ValueError("state decay must lie in [0, 1]")
    if not torch.isfinite(previous_state).all() or not torch.isfinite(
        action_signature
    ).all():
        raise ValueError("state and action signature must be finite")
    return float(decay) * previous_state + action_signature


def sketch_action_risk(
    state_sketch: torch.Tensor,
    action_signature: torch.Tensor,
) -> torch.Tensor:
    """Evaluate state--action interaction plus current action energy."""

    if state_sketch.shape != action_signature.shape:
        raise ValueError("state and action signature must have equal shape")
    interaction = torch.dot(state_sketch.reshape(-1), action_signature.reshape(-1))
    energy = 0.5 * torch.dot(
        action_signature.reshape(-1), action_signature.reshape(-1)
    )
    return interaction + energy


def _switch_exchanges(
    retained: Sequence[int], previous_retained: Optional[Sequence[int]]
) -> float:
    if previous_retained is None:
        return 0.0
    return 0.5 * float(
        len(set(retained).symmetric_difference(previous_retained))
    )


def _single_deletion_signatures(
    attention: torch.Tensor,
    values: torch.Tensor,
    *,
    output_projection: Optional[torch.Tensor],
    sketch_projection: Optional[torch.Tensor],
    minimum_retained_mass: float,
) -> torch.Tensor:
    """Vectorize exact one-token deletion responses for shortlist scoring."""

    total_mass = attention.sum(dim=-1, keepdim=True)
    retained_mass = total_mass - attention
    if torch.any(retained_mass <= float(minimum_retained_mass)):
        raise ValueError("single deletion leaves numerically zero attention mass")
    full_output = torch.sum(attention.unsqueeze(-1) * values, dim=-2)
    response = attention.unsqueeze(-1) * (
        full_output.unsqueeze(-2) - values
    ) / retained_mass.unsqueeze(-1)
    rows = response.movedim(-2, 0).reshape(int(attention.shape[-1]), -1)
    for name, projection in (
        ("output", output_projection),
        ("sketch", sketch_projection),
    ):
        if projection is None:
            continue
        if projection.ndim != 2 or int(projection.shape[0]) != int(rows.shape[-1]):
            raise ValueError(
                "%s projection must have shape [current_width, projected_width]"
                % name
            )
        rows = rows.to(dtype=projection.dtype, device=projection.device) @ projection
    return rows


def direct_retained_set(
    attention: torch.Tensor,
    values: torch.Tensor,
    *,
    budget: int,
    state_sketch: torch.Tensor,
    previous_retained: Optional[Sequence[int]] = None,
    mandatory_positions: Sequence[int] = (),
    output_projection: Optional[torch.Tensor] = None,
    sketch_projection: Optional[torch.Tensor] = None,
    shortlist_ratio: float = 2.0,
    switch_penalty: float = 0.0,
    max_swaps: int = 32,
    minimum_improvement: float = 1.0e-12,
    minimum_retained_mass: float = 1.0e-12,
) -> DirectActionDecision:
    """Directly optimize a retained set with a two-stage training-free search.

    The shortlist ranks tokens by the risk of deleting each token alone.  The
    local-search stage evaluates the exact renormalized response of every
    proposed retained set, so the final decision is set-level rather than a
    top-k token score.  Previous retained positions are always admitted to the
    search pool so a switching penalty can genuinely preserve the old action.
    """

    if attention.ndim < 1 or values.ndim < 2:
        raise ValueError("attention and values must include a position axis")
    if attention.shape != values.shape[:-1]:
        raise ValueError("attention and values have incompatible shapes")
    if not torch.isfinite(attention).all() or not torch.isfinite(values).all():
        raise ValueError("attention and values must be finite")
    position_count = int(attention.shape[-1])
    keep = int(budget)
    if keep < 1 or keep > position_count:
        raise ValueError("budget must lie between one and the position count")
    if float(shortlist_ratio) < 1.0:
        raise ValueError("shortlist ratio must be at least one")
    if float(switch_penalty) < 0.0:
        raise ValueError("switch penalty must be non-negative")
    if int(max_swaps) < 0:
        raise ValueError("max swaps must be non-negative")

    mandatory = tuple(sorted(set(int(value) for value in mandatory_positions)))
    if mandatory and (mandatory[0] < 0 or mandatory[-1] >= position_count):
        raise IndexError("mandatory position lies outside the attention support")
    if len(mandatory) > keep:
        raise ValueError("mandatory positions exceed the cache budget")

    previous: Optional[Tuple[int, ...]] = None
    if previous_retained is not None:
        previous = tuple(sorted(set(int(value) for value in previous_retained)))
        if len(previous) != keep:
            raise ValueError("previous retained set must match the cache budget")
        if previous[0] < 0 or previous[-1] >= position_count:
            raise IndexError("previous retained position lies outside support")

    single_signatures = _single_deletion_signatures(
        attention,
        values,
        output_projection=output_projection,
        sketch_projection=sketch_projection,
        minimum_retained_mass=minimum_retained_mass,
    )
    state = state_sketch.to(
        dtype=single_signatures.dtype, device=single_signatures.device
    ).reshape(-1)
    if int(state.numel()) != int(single_signatures.shape[-1]):
        raise ValueError("state sketch width does not match action signatures")
    single_risks_tensor = torch.stack(
        [sketch_action_risk(state, row) for row in single_signatures]
    )
    single_risks = {
        position: float(single_risks_tensor[position].item())
        for position in range(position_count)
    }

    mandatory_set = set(mandatory)
    eligible = [
        position for position in range(position_count) if position not in mandatory_set
    ]
    core_budget = keep - len(mandatory)
    ordered = sorted(eligible, key=lambda position: (-single_risks[position], position))
    shortlist_size = min(
        len(eligible),
        max(core_budget, int(math.ceil(float(shortlist_ratio) * core_budget))),
    )
    pool = set(ordered[:shortlist_size])
    if previous is not None:
        pool.update(position for position in previous if position not in mandatory_set)

    selected = set(mandatory)
    if previous is not None:
        previous_core = [
            position for position in previous if position not in mandatory_set
        ]
        previous_core.sort(key=lambda position: (-single_risks[position], position))
        selected.update(previous_core[:core_budget])
    for position in ordered:
        if len(selected) >= keep:
            break
        if position in pool:
            selected.add(position)
    if len(selected) != keep:
        raise RuntimeError("direct search could not construct a budget-feasible set")
    pool.update(selected - mandatory_set)

    def evaluate(retained: Sequence[int]) -> Tuple[float, float, float, torch.Tensor]:
        signature = retained_action_signature(
            attention,
            values,
            retained,
            output_projection=output_projection,
            sketch_projection=sketch_projection,
        ).to(dtype=state.dtype, device=state.device)
        risk = float(sketch_action_risk(state, signature).item())
        cost = float(switch_penalty) * _switch_exchanges(retained, previous)
        return risk + cost, risk, cost, signature

    objective, action_risk, switch_cost, signature = evaluate(sorted(selected))
    swaps = 0
    for _ in range(int(max_swaps)):
        best = None
        for dropped in sorted(selected - mandatory_set):
            for added in sorted(pool - selected):
                proposal = sorted((selected - {dropped}) | {added})
                trial = evaluate(proposal)
                key = (trial[0], added, dropped)
                if best is None or key < best[0]:
                    best = (key, dropped, added, trial)
        if best is None or best[3][0] >= objective - float(minimum_improvement):
            break
        _, dropped, added, trial = best
        selected.remove(dropped)
        selected.add(added)
        objective, action_risk, switch_cost, signature = trial
        swaps += 1

    retained = tuple(sorted(selected))
    return DirectActionDecision(
        retained_positions=retained,
        shortlist_positions=tuple(sorted(pool | mandatory_set)),
        action_risk=float(action_risk),
        switch_cost=float(switch_cost),
        objective=float(objective),
        swaps=int(swaps),
        refreshed=bool(previous is not None and retained != previous),
        action_signature=signature.detach().clone(),
        single_deletion_risks=tuple(
            (position, single_risks[position]) for position in range(position_count)
        ),
    )


__all__ = [
    "DirectActionDecision",
    "direct_retained_set",
    "fixed_rademacher_projection",
    "project_action_response",
    "query_continuity_decay",
    "retained_action_signature",
    "sketch_action_risk",
    "update_state_sketch",
]
