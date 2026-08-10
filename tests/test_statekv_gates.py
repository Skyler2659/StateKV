from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from statekv.statekv_gate_analysis import _evaluate_gate, _fit_scalar_gate
from statekv.statekv_gate_runner import _find_subsequence, _monitor_spec, _softmax_row


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/stages/statekv_p1_p3_gates_qwen3_8b.yaml"


def test_gate_protocol_has_disjoint_calibration_validation_and_test() -> None:
    config = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    calibration = set(config["calibration"]["sample_ids"])
    validation = set(config["p1"]["sample_ids"])
    test = set(config["p3"]["sample_ids"])
    assert not calibration & validation
    assert not calibration & test
    assert not validation & test
    assert config["claim_contract"]["govreport_81_role"] == "diagnostic_case_only"
    for budget in config["p2"]["budgets"]:
        assert budget["total_budget"] == (
            config["sink_size"] + config["recent_size"] + budget["core_budget"]
        )


def test_distribution_telemetry_tracks_requested_token_probabilities() -> None:
    full = torch.tensor([0.0, 2.0, 1.0, -1.0])
    compressed = torch.tensor([0.0, 1.0, 2.0, -1.0])
    row = _softmax_row(full, compressed, {"token_17": [1], "token_14": [2]})
    assert row["argmax_diverged"] is True
    assert row["full_probability_token_17"] > row["compressed_probability_token_17"]
    assert row["compressed_probability_token_14"] > row["full_probability_token_14"]
    assert _find_subsequence([1, 2, 3, 2, 3], [2, 3]) == [1, 2, 3, 4]

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return {"17": [16, 22], "14": [16, 19], "17 sailors": [16, 22, 7]}[text]

    monitored, evidence, _ = _monitor_spec(
        Tokenizer(),
        [0, 16, 22, 7, 9],
        {"token_17": ["17"], "token_14": ["14"]},
        ["17 sailors"],
    )
    assert monitored == {"token_17": [22], "token_14": [19]}
    assert evidence == [1, 2, 3]


def test_tail_gate_is_fit_on_validation_and_applied_without_rethresholding() -> None:
    validation = pd.DataFrame(
        {
            "target_next_exact_kl": np.linspace(0.0, 1.0, 40),
            "budget_l1_change": np.linspace(0.0, 1.0, 40),
            "a2_mask_mean_jaccard": np.linspace(1.0, 0.0, 40),
            "attention_mask_mean_jaccard": np.linspace(1.0, 0.0, 40),
            "compressed_margin": np.linspace(1.0, 0.0, 40),
            "compressed_entropy": np.linspace(0.0, 1.0, 40),
            "maximum_layer_volatility": np.linspace(0.0, 1.0, 40),
            "mean_layer_effective_support": np.linspace(0.0, 1.0, 40),
        }
    )
    gate = _fit_scalar_gate(validation)
    test = validation.copy()
    result = _evaluate_gate(gate, test)
    assert gate["validation_rows"] == 40
    assert result["test_rows"] == 40
    assert result["test_recall"] > 0.0
