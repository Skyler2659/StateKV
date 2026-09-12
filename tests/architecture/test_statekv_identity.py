from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_readme_first_screen_establishes_statekv_identity() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0] == "# StateKV"
    first_screen = "\n".join(lines[:24]).lower()
    assert "research" in first_screen
    assert "kv-cache" in first_screen
    assert "cheap-r2" in first_screen


def test_statekv_public_reports_are_linked_from_readme() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for path in (
        "docs/README.md",
        "docs/experiments/README.md",
        "docs/experiments/07_cheap_r2.md",
        "docs/experiments/08_benchmark_results.md",
        "docs/FINDINGS.md",
        "docs/REPRODUCIBILITY.md",
    ):
        assert path in text
        assert (ROOT / path).is_file()
