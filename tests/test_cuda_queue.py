import json
from pathlib import Path

import pytest

from statekv.cuda_queue import build_queue, gpu_available


def test_queue_has_complete_unique_resumable_jobs(tmp_path):
    plan = Path(__file__).resolve().parents[1] / "configs/cuda/validation.yaml"
    run = build_queue(plan, tmp_path)
    queue = json.loads((run / "queue.json").read_text())
    assert len(queue["jobs"]) == len({job["id"] for job in queue["jobs"]}) == 295
    arms = sum(len(job["config"]["sample_indices"]) * sum(
        1 if policy == "FULL" else len(job["config"]["budgets"])
        for policy in job["config"]["policies"]) for job in queue["jobs"])
    assert arms == 53500
    assert build_queue(plan, tmp_path) == run


def test_other_users_compute_process_prevents_gpu_use(monkeypatch):
    replies = iter(["GPU-test, 20, 0", "GPU-test, 12345"])
    monkeypatch.setattr("subprocess.check_output", lambda *args, **kwargs: next(replies))
    assert gpu_available(3) == (False, "GPU-test")


def test_disallowed_gpu_is_rejected_before_accessing_server():
    with pytest.raises(ValueError, match="only physical"):
        gpu_available(0)
