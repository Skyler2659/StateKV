from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_readme_first_screen_establishes_statekv_identity() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0] == "# StateKV"
    first_screen = "\n".join(lines[:24]).lower()
    # The repository is positioned as a research codebase documenting both
    # positive and negative results; the original method line is closed.
    assert "negative result" in first_screen
    assert "research" in first_screen
    assert "docs/research_history.md" in text
    assert "docs/FINDINGS.md".lower() in text.lower()


def test_statekv_machine_sources_of_truth_are_linked_from_readme() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "experiments/frozen_registry.yaml" in text
