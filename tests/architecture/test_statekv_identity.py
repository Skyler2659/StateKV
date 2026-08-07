from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_readme_first_screen_establishes_statekv_identity() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0] == "# StateKV"
    assert "State-conditioned physical risk for KV-cache selection and refresh" in lines[2]
    first_screen = "\n".join(lines[:24]).lower()
    assert "l1/l2 leverage" in first_screen
    assert "not as statekv itself" in first_screen


def test_statekv_machine_sources_of_truth_are_linked_from_readme() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "experiments/frozen_registry.yaml" in text
