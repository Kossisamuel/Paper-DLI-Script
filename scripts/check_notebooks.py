"""Validate and execute notebooks, saving generated copies under results/."""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ("posterdli.ipynb", "discussiondli.ipynb")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", choices=NOTEBOOKS)
    parser.add_argument("--quick", action="store_true", help="Use reduced discussion grids.")
    parser.add_argument("--check-only", action="store_true", help="Validate schema and syntax only.")
    parser.add_argument("--timeout", type=int, default=1800, help="Seconds allowed per cell.")
    args = parser.parse_args()
    if args.quick:
        os.environ["MI_QUICK_RUN"] = "1"
    else:
        os.environ.pop("MI_QUICK_RUN", None)
    os.environ.setdefault("MI_N_JOBS", "1")
    output_dir = ROOT / "results" / ("quick" if args.quick else "full")
    if not args.check_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        packages = [
            "numpy", "pandas", "scikit-learn", "imbalanced-learn", "seaborn",
            "matplotlib", "ucimlrepo", "nbformat", "nbclient", "ipykernel",
        ]
        manifest = {
            "python": sys.version,
            "quick": args.quick,
            "packages": {name: version(name) for name in packages},
        }
        (output_dir / "environment.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    for name in args.notebooks or NOTEBOOKS:
        notebook = nbformat.read(ROOT / name, as_version=4)
        nbformat.validate(notebook)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                ast.parse(cell.source, filename=f"{name}:cell{index}")
        print(f"Validated {name}", flush=True)
        if args.check_only:
            continue
        client = NotebookClient(
            notebook,
            timeout=args.timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
            allow_errors=False,
        )
        client.execute()
        nbformat.write(notebook, output_dir / name)
        print(f"Executed {name}; saved to {output_dir / name}", flush=True)


if __name__ == "__main__":
    main()
