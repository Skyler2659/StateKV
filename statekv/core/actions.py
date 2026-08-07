"""Functional-state and retained-set action primitives."""
from __future__ import annotations

from typing import Iterable, Tuple

import torch


def functional_history_state(
    history_boundary: torch.Tensor,
    reference_boundary: torch.Tensor,
) -> torch.Tensor:
    """Return the functional displacement induced by compression history."""

    if history_boundary.shape != reference_boundary.shape:
        raise ValueError("history and reference boundaries must have equal shape")
    return history_boundary - reference_boundary


def _retained_index(
    retained_positions: Iterable[int],
    position_count: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    retained = tuple(sorted(set(int(position) for position in retained_positions)))
    if not retained:
        raise ValueError("a retention action must keep at least one position")
    if retained[0] < 0 or retained[-1] >= int(position_count):
        raise IndexError("retained position lies outside the attention support")
    retained_index = torch.tensor(retained, dtype=torch.long, device=device)
    retained_mask = torch.zeros(position_count, dtype=torch.bool, device=device)
    retained_mask[retained_index] = True
    deleted_index = torch.nonzero(~retained_mask, as_tuple=False).flatten()
    return retained_index, deleted_index


def set_level_attention_delta(
    attention: torch.Tensor,
    values: torch.Tensor,
    retained_positions: Iterable[int],
    *,
    minimum_retained_mass: float = 1.0e-12,
) -> torch.Tensor:
    """Compute the exact fixed-operating-point output change after deletion.

    ``attention`` has shape ``(..., positions)`` and ``values`` has shape
    ``(..., positions, value_width)``.  The returned tensor is the retained
    attention output minus the full-cache output.  All heads/batches share the
    same retained position set, matching the paper's set-level action.
    """

    if attention.ndim < 1 or values.ndim < 2:
        raise ValueError("attention and values must include a position axis")
    if attention.shape != values.shape[:-1]:
        raise ValueError(
            "attention shape must equal values shape without the value axis"
        )
    if not torch.isfinite(attention).all() or not torch.isfinite(values).all():
        raise ValueError("attention and values must be finite")

    retained_index, deleted_index = _retained_index(
        retained_positions, attention.shape[-1], attention.device
    )
    full_output = torch.sum(attention.unsqueeze(-1) * values, dim=-2)
    retained_mass = attention.index_select(-1, retained_index).sum(dim=-1)
    if torch.any(retained_mass <= float(minimum_retained_mass)):
        raise ValueError("retained attention mass is numerically zero")

    if deleted_index.numel() == 0:
        return torch.zeros_like(full_output)
    deleted_attention = attention.index_select(-1, deleted_index)
    deleted_values = values.index_select(-2, deleted_index)
    numerator = torch.sum(
        deleted_attention.unsqueeze(-1)
        * (full_output.unsqueeze(-2) - deleted_values),
        dim=-2,
    )
    return numerator / retained_mass.unsqueeze(-1)
