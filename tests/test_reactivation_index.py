import numpy as np

from statekv.reactivation_index import (
    ReactivationParams,
    compute_sequence_reactivation,
)


def _artifact(attention, lengths):
    attention = np.asarray(attention, dtype=np.float32)
    cycles, positions = attention.shape
    return {
        "attention": attention.reshape(cycles, 1, 1, positions),
        "position_lengths": np.asarray(lengths, dtype=np.int32),
        "sample_id": np.asarray("synthetic_test"),
        "task": np.asarray("test_task"),
    }


def test_no_reactivation_when_token_always_important():
    # position 0 is top-1 at every cycle -> no entry after min_cycle, no events
    cycles, positions = 32, 20
    attention = np.full((cycles, positions), 0.01)
    attention[:, 0] = 1.0
    art = _artifact(attention, [positions] * cycles)
    result = compute_sequence_reactivation(art, ReactivationParams(top_k=1, dormant_window=4))
    assert result.n_future_important_events == 0
    assert result.n_reactivation_events == 0
    assert result.ri_fraction == 0.0


def test_dormant_token_reactivation_is_counted():
    cycles, positions = 40, 50
    attention = np.full((cycles, positions), 0.001)
    # a fixed set of "always important" distractors
    attention[:, 1:6] = 0.5
    # position 10 sits BELOW the filler (bottom ranks) until it spikes
    attention[:, 10] = 0.0001
    attention[30:, 10] = 2.0
    art = _artifact(attention, [positions] * cycles)
    params = ReactivationParams(top_k=1, dormant_window=16, dormant_rank_quantile=0.5)
    result = compute_sequence_reactivation(art, params)
    # position 10 enters top-1 at cycle 30 after 30 bottom-rank cycles
    assert result.n_reactivation_events == 1
    event = result.events[0]
    assert event.position == 10
    assert event.cycle == 30
    assert event.dormancy_duration == 30
    assert event.reactivation_distance == 30
    assert event.amplitude > 0
    # only position 10 produces an entry (distractor top-1 is a continuation)
    assert result.n_future_important_events == 1


def test_short_dormancy_does_not_count():
    cycles, positions = 40, 50
    attention = np.full((cycles, positions), 0.001)
    attention[:, 1:6] = 0.5
    attention[:, 10] = 0.0001
    attention[20:, 10] = 2.0  # 20 dormant cycles < window 32
    art = _artifact(attention, [positions] * cycles)
    params = ReactivationParams(top_k=1, dormant_window=32, dormant_rank_quantile=0.5)
    result = compute_sequence_reactivation(art, params)
    assert result.n_reactivation_events == 0


def test_positions_respect_cache_entry_time():
    # position 40 only exists from cycle 20 onward
    cycles, positions = 40, 50
    lengths = [20] * 20 + [positions] * 20
    attention = np.full((cycles, positions), 0.001)
    attention[:, 1:6] = 0.5
    attention[:, 40] = 0.0001
    attention[39, 40] = 5.0  # 19 dormant cycles < window 16? 39-20=19 >= 16
    art = _artifact(attention, lengths)
    params = ReactivationParams(top_k=1, dormant_window=25, dormant_rank_quantile=0.5)
    result = compute_sequence_reactivation(art, params)
    # only 19 cycles of existence -> cannot satisfy a 25-cycle dormancy window
    assert result.n_reactivation_events == 0
    assert result.n_future_important_events >= 1
