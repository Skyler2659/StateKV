"""Immutable mapping between cache rows and original token positions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PositionMap:
    """Map current cache rows to original token positions.

    ``None`` represents a padded, physically allocated row that is not a valid
    token. Mutating operations return a new map so a saved snapshot cannot be
    changed by later cache evictions.
    """

    positions: Tuple[Optional[int], ...]

    def __post_init__(self) -> None:
        normalized = tuple(None if value is None else int(value) for value in self.positions)
        object.__setattr__(self, "positions", normalized)
        valid = [value for value in normalized if value is not None]
        if any(value < 0 for value in valid):
            raise ValueError("original token positions must be non-negative")
        if len(valid) != len(set(valid)):
            raise ValueError("valid original token positions must be unique")

    @classmethod
    def identity(cls, length: int, start: int = 0) -> "PositionMap":
        if length < 0 or start < 0:
            raise ValueError("length and start must be non-negative")
        return cls(tuple(range(start, start + length)))

    @classmethod
    def from_iterable(cls, positions: Iterable[Optional[int]]) -> "PositionMap":
        return cls(tuple(positions))

    def __len__(self) -> int:
        return len(self.positions)

    @property
    def valid_mask(self) -> Tuple[bool, ...]:
        return tuple(value is not None for value in self.positions)

    @property
    def valid_positions(self) -> Tuple[int, ...]:
        return tuple(value for value in self.positions if value is not None)

    def append(self, original_positions: Iterable[Optional[int]]) -> "PositionMap":
        return PositionMap(self.positions + tuple(original_positions))

    def prune(self, keep_rows: Sequence[int]) -> "PositionMap":
        rows = tuple(int(row) for row in keep_rows)
        if len(rows) != len(set(rows)):
            raise ValueError("keep rows must be unique")
        if any(row < 0 or row >= len(self.positions) for row in rows):
            raise IndexError("keep row is outside the current cache")
        return PositionMap(tuple(self.positions[row] for row in rows))

    def with_padding(self, total_rows: int) -> "PositionMap":
        if total_rows < len(self.positions):
            raise ValueError("total_rows cannot truncate a position map")
        return PositionMap(self.positions + (None,) * (total_rows - len(self.positions)))

    def current_to_original(self, row: int) -> Optional[int]:
        if row < 0 or row >= len(self.positions):
            raise IndexError("cache row is outside the current cache")
        return self.positions[row]

    def original_to_current(self) -> Dict[int, int]:
        return {
            original: row
            for row, original in enumerate(self.positions)
            if original is not None
        }

    def rows_for_original(self, original_positions: Iterable[int]) -> Tuple[int, ...]:
        reverse = self.original_to_current()
        requested = tuple(int(position) for position in original_positions)
        missing = [position for position in requested if position not in reverse]
        if missing:
            raise KeyError(f"original positions are not in the cache: {missing}")
        return tuple(reverse[position] for position in requested)

    def to_dict(self) -> Dict[str, object]:
        return {
            "positions": list(self.positions),
            "valid_mask": list(self.valid_mask),
        }

