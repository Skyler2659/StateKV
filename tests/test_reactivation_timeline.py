import numpy as np

from statekv.reactivation_timeline import (
    TimelineReactivationParams,
    compute_timeline_reactivation,
    timeline_importance,
)


def _artifact(prefill, decode, lengths, prompt_length):
    """Build a minimal timeline artifact from (rows, positions) importances."""
    prefill = np.asarray(prefill, dtype=np.float32)
    decode = np.asarray(decode, dtype=np.float32)
    n_blocks, prompt_width = prefill.shape
    n_cycles, width = decode.shape
    assert prompt_width == int(prompt_length)
    assert width == int(prompt_length) + n_cycles
    return {
        "prefill_block_attention": prefill.reshape(n_blocks, 1, 1, prompt_width),
        "decode_attention": decode.reshape(n_cycles, 1, 1, width),
        "position_lengths": np.asarray(lengths, dtype=np.int32),
        "prompt_length": np.asarray(prompt_length, dtype=np.int32),
        "sample_id": np.asarray("synthetic_test"),
        "task": np.asarray("test_task"),
    }


def _uniform_lengths(n_blocks, n_cycles, prompt_length):
    lengths = [prompt_length] * n_blocks
    lengths += [prompt_length + cycle + 1 for cycle in range(n_cycles)]
    return lengths


def _pad_decode(decode_rows, prompt_length, n_cycles):
    width = prompt_length + n_cycles
    decode = np.zeros((n_cycles, width))
    decode[:, :prompt_length] = decode_rows
    return decode


def test_always_important_token_is_type_iii_without_events():
    # position 0 is top-1 at every timeline row -> Type III, no events
    n_blocks, n_cycles, positions = 8, 8, 20
    prompt_length = positions - n_cycles
    prefill = np.full((n_blocks, prompt_length), 0.01)
    prefill[:, 0] = 1.0
    decode = np.full((n_cycles, prompt_length), 0.01)
    decode[:, 0] = 1.0
    art = _artifact(
        prefill,
        _pad_decode(decode, prompt_length, n_cycles),
        _uniform_lengths(n_blocks, n_cycles, prompt_length),
        prompt_length,
    )
    result = compute_timeline_reactivation(
        art, TimelineReactivationParams(top_k=1, dormant_window_rows=4)
    )
    assert result.n_reactivation_events == 0
    assert result.n_entry_events == 0
    assert result.ri_fraction == 0.0
    assert result.n_type_iii_persistent == 1


def test_dormant_token_spiking_in_prefill_is_type_i():
    n_blocks, n_cycles, positions = 12, 4, 30
    prompt_length = positions - n_cycles
    prefill = np.full((n_blocks, prompt_length), 0.001)
    prefill[:, 1:6] = 0.5  # fixed always-important distractors
    prefill[:, 10] = 0.0001  # bottom ranks ...
    prefill[9:, 10] = 2.0  # ... until it spikes inside prefill block row 9
    decode = np.full((n_cycles, prompt_length), 0.001)
    decode[:, 1:6] = 0.5
    decode[:, 10] = 2.0
    art = _artifact(
        prefill,
        _pad_decode(decode, prompt_length, n_cycles),
        _uniform_lengths(n_blocks, n_cycles, prompt_length),
        prompt_length,
    )
    params = TimelineReactivationParams(
        top_k=1, dormant_window_rows=8, dormant_rank_quantile=0.5
    )
    result = compute_timeline_reactivation(art, params)
    assert result.n_reactivation_events == 1
    assert result.n_type_i_events == 1
    assert result.n_type_ii_events == 0
    event = result.events[0]
    assert event.position == 10
    assert event.row == 9
    assert event.phase == "prefill"
    assert event.event_type == "I"
    assert event.dormancy_duration == 9
    assert event.reactivation_distance == 9
    assert event.amplitude > 0
    # only position 10 produces an entry (distractor top-1 is a continuation)
    assert result.n_entry_events == 1


def test_dormant_token_spiking_in_decode_is_type_ii():
    n_blocks, n_cycles, positions = 6, 10, 40
    prompt_length = positions - n_cycles
    prefill = np.full((n_blocks, prompt_length), 0.001)
    prefill[:, 1:6] = 0.5
    prefill[:, 10] = 0.0001
    decode = np.full((n_cycles, prompt_length), 0.001)
    decode[:, 1:6] = 0.5
    decode[:, 10] = 0.0001
    decode[5:, 10] = 2.0  # spike at decode row 5 -> timeline row 6 + 5 = 11
    art = _artifact(
        prefill,
        _pad_decode(decode, prompt_length, n_cycles),
        _uniform_lengths(n_blocks, n_cycles, prompt_length),
        prompt_length,
    )
    params = TimelineReactivationParams(
        top_k=1, dormant_window_rows=8, dormant_rank_quantile=0.5
    )
    result = compute_timeline_reactivation(art, params)
    assert result.n_reactivation_events == 1
    assert result.n_type_i_events == 0
    assert result.n_type_ii_events == 1
    event = result.events[0]
    assert event.position == 10
    assert event.row == 11
    assert event.phase == "decode"
    assert event.event_type == "II"
    # dormancy spans prefill blocks and decode cycles in the same row units
    assert event.dormancy_duration == 11


def test_short_dormancy_does_not_count():
    n_blocks, n_cycles, positions = 8, 8, 40
    prompt_length = positions - n_cycles
    prefill = np.full((n_blocks, prompt_length), 0.001)
    prefill[:, 1:6] = 0.5
    prefill[:, 10] = 0.0001
    prefill[5:, 10] = 2.0  # only 5 dormant rows < window 8
    decode = np.full((n_cycles, prompt_length), 0.001)
    decode[:, 1:6] = 0.5
    decode[:, 10] = 2.0
    art = _artifact(
        prefill,
        _pad_decode(decode, prompt_length, n_cycles),
        _uniform_lengths(n_blocks, n_cycles, prompt_length),
        prompt_length,
    )
    params = TimelineReactivationParams(
        top_k=1, dormant_window_rows=8, dormant_rank_quantile=0.5
    )
    result = compute_timeline_reactivation(art, params)
    assert result.n_reactivation_events == 0


def test_positions_respect_cache_entry_time():
    # realistic prefill block spans of 4: position 40 first becomes an
    # active key in block row 10 (span end 44) and spikes at decode row 3
    n_blocks, n_cycles = 12, 4
    prompt_length = 48
    positions = prompt_length + n_cycles
    lengths = [4 * (block + 1) for block in range(n_blocks)]
    lengths += [prompt_length + cycle + 1 for cycle in range(n_cycles)]
    prefill = np.full((n_blocks, prompt_length), 0.001)
    prefill[:, 1:6] = 0.5
    prefill[:, 40] = 0.0001
    decode = np.full((n_cycles, prompt_length), 0.001)
    decode[:, 1:6] = 0.5
    decode[:, 40] = 0.0001
    decode[3, 40] = 5.0  # timeline row 15; only 5 dormant rows of existence
    art = _artifact(
        prefill,
        _pad_decode(decode, prompt_length, n_cycles),
        lengths,
        prompt_length,
    )
    params = TimelineReactivationParams(
        top_k=1, dormant_window_rows=8, dormant_rank_quantile=0.5
    )
    result = compute_timeline_reactivation(art, params)
    # rows 10..14 = 5 dormant rows < window 8 -> no reactivation event
    assert result.n_reactivation_events == 0
    # the spike at row 15 is still an entry event (denominator)
    assert result.n_entry_events >= 1


def test_timeline_importance_fails_loudly_on_shape_mismatch():
    n_blocks, n_cycles, prompt_length = 4, 4, 16
    art = _artifact(
        np.full((n_blocks, prompt_length), 0.01),
        np.zeros((n_cycles, prompt_length + n_cycles)),
        _uniform_lengths(n_blocks, n_cycles, prompt_length),
        prompt_length,
    )
    art["position_lengths"] = art["position_lengths"][:-1]
    try:
        timeline_importance(art)
    except ValueError:
        pass
    else:
        raise AssertionError("expected a shape-mismatch failure")
