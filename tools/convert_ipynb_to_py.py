#!/usr/bin/env python3
"""
Convert Jupyter notebooks (*.ipynb) to runnable Python scripts without external deps.

Design goals:
- stdlib-only (no nbformat/nbconvert dependency)
- output is always valid Python (cell/line magics are commented out)
- code-cell outputs are intentionally ignored

Typical use:
  python3 tools/convert_ipynb_to_py.py --root . --delete
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any, Iterable


def _iter_ipynb(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.ipynb"):
        if ".git" in path.parts:
            continue
        yield path


def _load_notebook(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _cell_source(cell: dict[str, Any]) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    if isinstance(src, str):
        return src
    return str(src)


def _sanitize_python(source: str) -> str:
    """
    Comment out IPython magics and shell escapes so output is valid Python.
    """
    lines = source.splitlines(True)
    out: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("%%"):
            out.append("# " + line)
            continue
        if stripped.startswith("%") or stripped.startswith("!"):
            out.append("# " + line)
            continue
        out.append(line)
    if out and not out[-1].endswith("\n"):
        out[-1] = out[-1] + "\n"
    return "".join(out)


def _default_out_path(ipynb_path: Path) -> Path:
    stem = ipynb_path.stem
    return ipynb_path.with_name(f"nb_{stem}.py")


def convert_one(ipynb_path: Path, out_path: Path) -> int:
    nb = _load_notebook(ipynb_path)
    cells = nb.get("cells", [])
    if not isinstance(cells, list):
        raise ValueError(f"Invalid notebook format: cells is not list: {ipynb_path}")

    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = (
        f"# Auto-generated from {ipynb_path.as_posix()}\n"
        f"# Generated at {now}\n"
        "# Note: notebook outputs are omitted; IPython magics are commented out.\n\n"
    )

    parts: list[str] = [header]
    code_cells = 0
    for idx, cell in enumerate(cells):
        if not isinstance(cell, dict):
            continue
        if cell.get("cell_type") != "code":
            continue
        code_cells += 1
        raw = _cell_source(cell)
        sanitized = _sanitize_python(raw)
        parts.append(f"# %% [cell {idx}]\n")
        parts.append(sanitized if sanitized.endswith("\n") else sanitized + "\n")
        parts.append("\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(parts), encoding="utf-8")
    return code_cells


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="Root directory to search for *.ipynb")
    ap.add_argument(
        "--delete",
        action="store_true",
        help="Delete the original *.ipynb after successful conversion",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned conversions without writing files",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    ipynbs = sorted(_iter_ipynb(root))
    if not ipynbs:
        print("No .ipynb files found under", root)
        return 0

    converted = 0
    for ipynb_path in ipynbs:
        out_path = _default_out_path(ipynb_path)
        rel_in = os.path.relpath(ipynb_path, root)
        rel_out = os.path.relpath(out_path, root)
        if args.dry_run:
            print(f"[DRY] {rel_in} -> {rel_out}")
            continue

        cells = convert_one(ipynb_path, out_path)
        print(f"[OK] {rel_in} -> {rel_out} (code_cells={cells})")
        converted += 1

        if args.delete:
            ipynb_path.unlink()
            print(f"[DEL] {rel_in}")

    print(f"Converted {converted} notebook(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

