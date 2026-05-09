import argparse
import re
import subprocess
import sys
from pathlib import Path


ROW_RE = re.compile(
    r"^\s*sink_recent_l1_last\s+\d+\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)",
    re.MULTILINE,
)


def run_one(cmpr_path: Path, py_exe: str, args: argparse.Namespace, recent_keep: int):
    cmd = [
        py_exe,
        "-u",
        str(cmpr_path),
        "--model",
        args.model,
        "--comparison_mode",
        "three",
        "--text_source",
        args.text_source,
        "--max_steps",
        str(args.max_steps),
        "--cache_size",
        str(args.cache_size),
        "--start_size",
        str(args.start_size),
        "--l1_recent_keep",
        str(args.l1_recent_keep),
        "--mixed_recent_keep",
        str(recent_keep),
        "--sketch_dim",
        str(args.sketch_dim),
        "--seed",
        str(args.seed),
        "--progress_every",
        str(args.progress_every),
    ]
    if args.disable_pos_shift:
        cmd.append("--disable_pos_shift")

    print(f"\n=== mixed_recent_keep={recent_keep} ===")
    print(" ".join(cmd))
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    out = completed.stdout + "\n" + completed.stderr
    print(out)

    if completed.returncode != 0:
        return {
            "mixed_recent_keep": recent_keep,
            "ok": False,
            "returncode": completed.returncode,
            "ppl": None,
            "tok_s": None,
        }

    m = ROW_RE.search(out)
    if not m:
        return {
            "mixed_recent_keep": recent_keep,
            "ok": False,
            "returncode": 0,
            "ppl": None,
            "tok_s": None,
        }
    return {
        "mixed_recent_keep": recent_keep,
        "ok": True,
        "returncode": 0,
        "ppl": float(m.group(1)),
        "tok_s": float(m.group(2)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Grid search mixed_recent_keep without modifying cmpr.py"
    )
    parser.add_argument("--python_exe", type=str, default=sys.executable)
    parser.add_argument("--cmpr_path", type=str, default="cmpr.py")
    parser.add_argument(
        "--values",
        type=int,
        nargs="+",
        default=[32, 48, 64, 80, 96],
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--text_source", type=str, default="wikitext")
    parser.add_argument("--max_steps", type=int, default=1536)
    parser.add_argument("--cache_size", type=int, default=128)
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--l1_recent_keep", type=int, default=0)
    parser.add_argument("--sketch_dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument("--disable_pos_shift", action="store_true")
    args = parser.parse_args()

    cmpr_path = Path(args.cmpr_path).resolve()
    if not cmpr_path.exists():
        raise FileNotFoundError(f"cmpr.py not found: {cmpr_path}")

    results = []
    for v in args.values:
        results.append(run_one(cmpr_path, args.python_exe, args, v))

    print("\n=== Grid Search Summary ===")
    print(f"{'mixed_recent_keep':>18} {'ok':>6} {'ppl':>12} {'tok/s':>10}")
    for r in results:
        ppl = f"{r['ppl']:.4f}" if r["ppl"] is not None else "NA"
        tok = f"{r['tok_s']:.2f}" if r["tok_s"] is not None else "NA"
        print(f"{r['mixed_recent_keep']:>18} {str(r['ok']):>6} {ppl:>12} {tok:>10}")

    valid = [r for r in results if r["ok"] and r["ppl"] is not None]
    if not valid:
        print("\nNo valid run was parsed. Please check logs above.")
        return
    best = min(valid, key=lambda x: x["ppl"])
    print(
        f"\nBest mixed_recent_keep={best['mixed_recent_keep']} "
        f"with ppl={best['ppl']:.4f}, tok/s={best['tok_s']:.2f}"
    )


if __name__ == "__main__":
    main()
