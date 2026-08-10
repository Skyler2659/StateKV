import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _python_sources(directory: Path):
    return (
        path
        for path in directory.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def _imported_modules(source: Path) -> set[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _imports_prefix(source: Path, forbidden: tuple[str, ...]) -> bool:
    return any(
        module == prefix or module.startswith(f"{prefix}.")
        for module in _imported_modules(source)
        for prefix in forbidden
    )


def test_statekv_has_a_canonical_package() -> None:
    package = ROOT / "statekv"
    assert (package / "__init__.py").is_file()
    assert (package / "core" / "actions.py").is_file()
    assert (package / "core" / "risk.py").is_file()
    assert (package / "core" / "decision.py").is_file()
    assert (package / "storage.py").is_file()
    assert (package / "config.py").is_file()
    assert (package / "backend.py").is_file()
    assert (package / "backend_mlx.py").is_file()
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert '"statekv*"' in pyproject


def test_statekv_core_is_independent_of_experiments_and_backends() -> None:
    core = ROOT / "statekv" / "core"
    for source in _python_sources(core):
        assert not _imports_prefix(
            source, ("experiments", "benchmarks", "statekv.backend")
        ), source


def test_legacy_temporal_namespace_is_only_a_compatibility_layer() -> None:
    compatibility = ROOT / "benchmarks" / "torch" / "kvbench" / "temporal"
    for source in compatibility.glob("*.py"):
        if source.name in {"__init__.py", "backend_mlx.py"}:
            continue
        expected = f'_import_module("statekv.{source.stem}")'
        assert expected in source.read_text(encoding="utf-8"), source

    frozen_mlx = compatibility / "backend_mlx.py"
    assert hashlib.sha256(frozen_mlx.read_bytes()).hexdigest() == (
        "07f961347f99c16d7bdf187f76ecc7be6c1bd8a894f01ad67d55c857f031d0a7"
    )


def test_benchmark_packages_do_not_depend_on_statekv_experiments() -> None:
    forbidden_root = ("statekv", "experiments")
    for source in _python_sources(ROOT / "benchmarks" / "mlx" / "src"):
        assert not _imports_prefix(source, forbidden_root), source

    forbidden_cuda = ("src", "experiments", "statekv")
    kvbench = ROOT / "benchmarks" / "torch" / "kvbench"
    for source in _python_sources(kvbench):
        if "temporal" in source.parts:
            continue
        assert not _imports_prefix(source, forbidden_cuda), source


def test_all_benchmark_assets_are_grouped_by_backend() -> None:
    benchmark_root = ROOT / "benchmarks"
    for backend in ("mlx", "torch"):
        directory = benchmark_root / backend
        assert directory.is_dir()
        assert (directory / "pyproject.toml").is_file()
        assert (directory / "configs").is_dir()
        assert (directory / "scripts").is_dir()
        assert (directory / "tests").is_dir()

    moved_root_assets = {
        "src",
        "benchmark.py",
        "cache_baselines.py",
        "data_sources.py",
        "shared_q.py",
        "h2o_llm",
        "l1_llm",
        "plain_llm",
        "rocketkv",
        "snapkv",
        "streaming_llm",
    }
    assert not any((ROOT / name).exists() for name in moved_root_assets)

    assert not (ROOT / "configs" / "eviction").exists()
    assert not (ROOT / "scripts" / "run_benchmark.py").exists()
    statekv_results = {
        directory.name
        for directory in (ROOT / "results").iterdir()
        if directory.is_dir()
    }
    assert statekv_results <= {"temporal_cache_discovery"}


def test_root_contains_only_canonical_documentation_and_no_yaml() -> None:
    canonical_docs = {
        "README.md",
        "docs/FINDINGS.md",
        "docs/FAILURE_ANALYSIS.md",
        "docs/research_history.md",
        "docs/CODE_AUDIT.md",
        "docs/REPRODUCIBILITY.md",
        "docs/NEXT_RESEARCH_DIRECTIONS.md",
        "docs/experiments/EXPERIMENT_REGISTRY.md",
    }
    missing = sorted(
        name for name in canonical_docs if not (ROOT / name).is_file()
    )
    assert not missing, "missing canonical documentation: %s" % ", ".join(missing)

    assert not list(ROOT.glob("*.yaml"))
    assert not list(ROOT.glob("*.yml"))

    allowed_areas = ("docs", "analysis", "assets", "experiments", "results", "tmp")
    markdown = {
        path.relative_to(ROOT)
        for path in ROOT.rglob("*.md")
        if not {".git", ".venv"}.intersection(path.parts)
    }
    disallowed = sorted(
        path.as_posix()
        for path in markdown
        if path.as_posix() != "README.md" and path.parts[0] not in allowed_areas
    )
    assert not disallowed, (
        "markdown outside documented areas: %s" % ", ".join(disallowed)
    )


def test_root_readme_is_the_only_documentation_entrypoint() -> None:
    assert (ROOT / "README.md").is_file()
    assert not (ROOT / "paper").exists()


def test_obvious_intermediate_root_documents_are_removed() -> None:
    removed = {
        "new.md",
        "REFACTOR_PLAN.md",
        "CODEBASE_AUDIT.md",
        "audit_report.md",
        "SMOKE_TEST_REPORT.md",
        "CLOSED_FORM_THEORY_DERIVATION_ZH.md",
        "THEORY_CLOSING_EXPERIMENT_DESIGN_ZH.md",
        "THEORY_MODEL_UPDATE_AFTER_TRAJECTORY_ZH.md",
        "TRAJECTORY_STOCHASTIC_MODEL_DESIGN_ZH.md",
        "TRAJECTORY_STOCHASTIC_MODEL_RESULTS_ZH.md",
    }
    assert not any((ROOT / name).exists() for name in removed)


def test_repository_governance_entrypoints_exist() -> None:
    assert (ROOT / "experiments" / "frozen_registry.yaml").is_file()


def test_canonical_statekv_does_not_mutate_python_path() -> None:
    for source in _python_sources(ROOT / "statekv"):
        assert "sys.path" not in source.read_text(encoding="utf-8"), source
