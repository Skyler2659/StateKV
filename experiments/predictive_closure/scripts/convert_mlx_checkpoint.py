#!/usr/bin/env python3
"""Deterministically dequantize the historical MLX checkpoint into PyTorch FP16.

The output contains ordinary floating-point PyTorch modules.  No quantized
operator participates in JVP/VJP.  Conversion is atomic and records content
hashes plus a short cross-backend logit check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import torch
from mlx_lm import load as mlx_load
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


DEFAULT_SOURCE = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
DEFAULT_REVISION = "8b403126fc14f14cfc99bb4cfa72ecbc129ea677"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nested_get(root: Any, dotted: str) -> Any:
    current = root
    for part in dotted.split("."):
        current = current[int(part)] if part.isdigit() else getattr(current, part)
    return current


def mlx_tensor_for_parameter(mlx_model: Any, parameter_name: str) -> np.ndarray:
    module_name, leaf = parameter_name.rsplit(".", 1)
    module = nested_get(mlx_model, module_name)
    if leaf == "weight" and all(
        hasattr(module, name) for name in ("weight", "scales", "biases")
    ):
        value = mx.dequantize(
            module.weight,
            module.scales,
            module.biases,
            group_size=int(getattr(module, "group_size", 64)),
            bits=int(getattr(module, "bits", 4)),
        )
    else:
        value = getattr(module, leaf)
    mx.eval(value)
    return np.asarray(value)


def instantiate_empty_fp16(config: Any) -> torch.nn.Module:
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    model = model.half()
    model.to_empty(device="cpu")
    return model


def copy_parameters(mlx_model: Any, torch_model: torch.nn.Module) -> dict:
    copied = []
    for name, parameter in torch_model.named_parameters():
        # Qwen ties lm_head to embed_tokens, so named_parameters normally emits
        # only the embedding.  Keep the guard for transformers variants that do
        # emit both names.
        source_name = (
            "model.embed_tokens.weight"
            if name == "lm_head.weight" and not hasattr(mlx_model, "lm_head")
            else name
        )
        array = mlx_tensor_for_parameter(mlx_model, source_name)
        if tuple(array.shape) != tuple(parameter.shape):
            raise RuntimeError(
                f"shape mismatch for {name}: MLX {array.shape}, Torch {tuple(parameter.shape)}"
            )
        tensor = torch.from_numpy(np.array(array, copy=True)).to(dtype=torch.float16)
        parameter.data.copy_(tensor)
        parameter.requires_grad_(False)
        copied.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
            }
        )
    torch_model.tie_weights()
    return {"parameter_count": len(copied), "parameters": copied}


@torch.inference_mode()
def validate_logits(
    mlx_model: Any,
    tokenizer: Any,
    torch_model: torch.nn.Module,
) -> dict:
    prompt = "A short deterministic checkpoint parity probe: 2 + 3 ="
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    mlx_logits = mlx_model(mx.array([ids]))[0, -1]
    mx.eval(mlx_logits)
    left = np.asarray(mlx_logits, dtype=np.float64)
    right = (
        torch_model(torch.tensor([ids], dtype=torch.long), use_cache=False)
        .logits[0, -1]
        .float()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    denominator = max(np.linalg.norm(left) * np.linalg.norm(right), 1e-30)
    cosine = float(np.dot(left, right) / denominator)
    relative_l2 = float(np.linalg.norm(left - right) / max(np.linalg.norm(left), 1e-30))
    return {
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "token_count": len(ids),
        "logit_cosine": cosine,
        "logit_relative_l2": relative_l2,
        "logit_max_absolute_error": float(np.max(np.abs(left - right))),
        "passed": bool(cosine >= 0.999 and relative_l2 <= 0.10),
    }


def convert(source: str, revision: str, output: Path) -> Path:
    metadata_path = output / "conversion_metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("status") == "complete":
            print(str(output.resolve()))
            return output

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(
        tempfile.mkdtemp(prefix=output.name + ".partial.", dir=str(output.parent))
    )
    started = time.time()
    try:
        mlx_model, mlx_tokenizer = mlx_load(
            source, tokenizer_config={"trust_remote_code": False}
        )
        source_snapshot = Path(
            os.path.realpath(
                Path.home()
                / ".cache/huggingface/hub"
                / "models--mlx-community--Qwen2.5-1.5B-Instruct-4bit"
                / "snapshots"
                / revision
            )
        )
        config = AutoConfig.from_pretrained(
            source_snapshot, local_files_only=True, trust_remote_code=False
        )
        tokenizer = AutoTokenizer.from_pretrained(
            source_snapshot, local_files_only=True, trust_remote_code=False
        )
        torch_model = instantiate_empty_fp16(config)
        copy_summary = copy_parameters(mlx_model, torch_model)
        parity = validate_logits(mlx_model, tokenizer, torch_model)
        if not parity["passed"]:
            raise RuntimeError(f"cross-backend checkpoint parity failed: {parity}")

        torch_model.save_pretrained(
            partial, safe_serialization=True, max_shard_size="5GB"
        )
        tokenizer.save_pretrained(partial)
        weight_files = sorted(partial.glob("*.safetensors"))
        metadata = {
            "status": "complete",
            "source": source,
            "revision": revision,
            "source_model_sha256": sha256_file(source_snapshot / "model.safetensors"),
            "source_tokenizer_sha256": sha256_file(source_snapshot / "tokenizer.json"),
            "execution_dtype": "float16",
            "quantized_ops_in_execution_graph": False,
            "torch_version": torch.__version__,
            "elapsed_seconds": time.time() - started,
            "weight_files": [
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in weight_files
            ],
            "copy_summary": copy_summary,
            "parity": parity,
        }
        with (partial / "conversion_metadata.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
        if output.exists():
            raise FileExistsError(
                f"incomplete output already exists; inspect before replacing: {output}"
            )
        os.replace(partial, output)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    print(str(output.resolve()))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/predictive_closure/checkpoints/"
            "qwen25_15b_instruct_4bit_dequant_fp16"
        ),
    )
    args = parser.parse_args()
    convert(args.source, args.revision, args.output)


if __name__ == "__main__":
    main()

