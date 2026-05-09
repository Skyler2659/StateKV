import inspect
import types

import torch

import transformers.models.llama.modeling_llama as modeling_llama
from transformers.models.llama.modeling_llama import LlamaAttention, rotate_half

__all__ = ["enable_llama_pos_shift_attention"]

_ORIG_APPLY_ROTARY = modeling_llama.apply_rotary_pos_emb
_ACTIVE_ATTN_STACK = []
_APPLY_PATCHED = False


def _build_key_position_ids(query_position_ids: torch.Tensor, kv_len: int):
    bsz = int(query_position_ids.shape[0])
    return torch.arange(kv_len, device=query_position_ids.device).unsqueeze(0).expand(bsz, -1)


def _dispatch_apply_rotary(q, k, cos, sin, *args, **kwargs):
    """Dispatcher around HF apply_rotary_pos_emb without replacing attention forward."""
    if not _ACTIVE_ATTN_STACK:
        return _ORIG_APPLY_ROTARY(q, k, cos, sin, *args, **kwargs)

    attn = _ACTIVE_ATTN_STACK[-1]
    if not getattr(attn, "_pos_shift_enabled", False):
        return _ORIG_APPLY_ROTARY(q, k, cos, sin, *args, **kwargs)

    query_pos = kwargs.get("position_ids")
    if query_pos is None:
        query_pos = getattr(attn, "_pos_shift_query_position_ids", None)
    if query_pos is None or not torch.is_tensor(query_pos):
        return _ORIG_APPLY_ROTARY(q, k, cos, sin, *args, **kwargs)

    key_pos = _build_key_position_ids(query_pos, kv_len=int(k.shape[-2]))

    sig = inspect.signature(_ORIG_APPLY_ROTARY)
    if "position_ids" in sig.parameters:
        q_kwargs = dict(kwargs)
        q_kwargs["position_ids"] = query_pos
        k_kwargs = dict(kwargs)
        k_kwargs["position_ids"] = key_pos
        q_rot, _ = _ORIG_APPLY_ROTARY(q, q, cos, sin, *args, **q_kwargs)
        k_rot, _ = _ORIG_APPLY_ROTARY(k, k, cos, sin, *args, **k_kwargs)
    else:
        cos_q = cos.squeeze(1).squeeze(0)[query_pos].unsqueeze(1)
        sin_q = sin.squeeze(1).squeeze(0)[query_pos].unsqueeze(1)
        cos_k = cos.squeeze(1).squeeze(0)[key_pos].unsqueeze(1)
        sin_k = sin.squeeze(1).squeeze(0)[key_pos].unsqueeze(1)
        q_rot = (q * cos_q) + (rotate_half(q) * sin_q)
        k_rot = (k * cos_k) + (rotate_half(k) * sin_k)
    return q_rot, k_rot


def _wrap_llama_attention_forward(attn_module):
    if getattr(attn_module, "_pos_shift_forward_wrapped", False):
        return

    original_forward = attn_module.forward

    def wrapped_forward(self, *args, **kwargs):
        # Capture position_ids from forward call (catches old rotary_emb API too)
        if "position_ids" in kwargs:
            self._pos_shift_query_position_ids = kwargs["position_ids"]
        elif len(args) >= 3 and torch.is_tensor(args[2]):
            self._pos_shift_query_position_ids = args[2]
        _ACTIVE_ATTN_STACK.append(self)
        try:
            return original_forward(*args, **kwargs)
        finally:
            _ACTIVE_ATTN_STACK.pop()

    attn_module.forward = types.MethodType(wrapped_forward, attn_module)
    attn_module._pos_shift_forward_wrapped = True
    attn_module._pos_shift_enabled = True


def _patch_apply_rotary_once():
    global _APPLY_PATCHED
    if _APPLY_PATCHED:
        return
    modeling_llama.apply_rotary_pos_emb = _dispatch_apply_rotary
    _APPLY_PATCHED = True


def enable_llama_pos_shift_attention(model):
    """Enable position-shifted RoPE for llama-family models.

    Patches apply_rotary_pos_emb so that query tokens receive shifted position ids
    while key tokens receive physical cache position ids.  The native attention
    forward is NOT replaced — only the rotary helper and a thin stack tracker are
    wrapped.
    """
    _patch_apply_rotary_once()
    for module in model.modules():
        if isinstance(module, LlamaAttention):
            _wrap_llama_attention_forward(module)
