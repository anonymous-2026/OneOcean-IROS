#!/usr/bin/env python3
"""
Inspect a OneOcean combined NetCDF file (schema/dims/vars) without xarray.

This is designed to be lightweight for downstream consumers (Lane A/C/E):
- prints dims and available variables,
- shows whether optional tide vars exist,
- prints key coordinate ranges and basic attributes.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable


COORDS = ("time", "depth", "latitude", "longitude")
CORE_VARS = ("uo", "vo", "so", "thetao", "zos")
TIDE_VARS = ("utide", "vtide", "utotal", "vtotal")


def _default_variants_root() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root.parent / "OceanEnv" / "Data_pipeline" / "Data" / "Combined" / "variants"


def _resolve_path(args: argparse.Namespace) -> Path:
    if args.path:
        path = Path(args.path).expanduser().resolve()
    else:
        variants_root = Path(
            os.environ.get("ONEOCEAN_VARIANTS_ROOT", str(_default_variants_root()))
        ).expanduser()
        path = (variants_root / args.variant / "combined" / "combined_environment.nc").resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return path


def _format_range(arr: np.ndarray) -> str:
    if arr.size == 0:
        return "empty"
    return f"min={np.nanmin(arr)} max={np.nanmax(arr)}"


def _safe_read_1d(ds: h5py.Dataset, max_elems: int = 10_000) -> np.ndarray:
    n = int(ds.shape[0])
    if n <= max_elems:
        return np.asarray(ds[:])
    idx = np.linspace(0, n - 1, max_elems, dtype=np.int64)
    return np.asarray(ds[idx])


def _list_root_datasets(h5: h5py.File) -> list[str]:
    out: list[str] = []
    for k, v in h5.items():
        if isinstance(v, h5py.Dataset):
            out.append(k)
    return sorted(out)


def _present(names: Iterable[str], available: set[str]) -> list[str]:
    return [n for n in names if n in available]


def main() -> None:
    try:
        import h5py  # type: ignore
        import numpy as np  # type: ignore
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or str(e)
        raise SystemExit(
            f"Missing dependency: {missing}. "
            "Install requirements first (e.g., `python -m pip install -r requirements.txt`) "
            "or run with an existing environment that has h5py/numpy."
        )

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, help="Path to combined_environment.nc")
    parser.add_argument(
        "--variant",
        default="scene",
        choices=("tiny", "scene", "public"),
        help="Variant name if --path is not provided",
    )
    args = parser.parse_args()

    path = _resolve_path(args)
    size_mib = path.stat().st_size / (1024 * 1024)

    with h5py.File(path, "r") as h5:
        available = set(_list_root_datasets(h5))

        print("path:", str(path))
        print("size_mib:", round(size_mib, 2))
        print()

        print("coords:")
        for c in COORDS:
            if c not in available:
                print(f"  - {c}: MISSING")
                continue
            d = h5[c]
            arr = _safe_read_1d(d)
            units = d.attrs.get("units")
            units_s = units.decode() if isinstance(units, (bytes, bytearray)) else str(units) if units is not None else None
            print(f"  - {c}: len={d.shape[0]} {_format_range(arr)}" + (f" units={units_s}" if units_s else ""))
        print()

        dims = {c: int(h5[c].shape[0]) for c in COORDS if c in available}
        print("dims:", dims)
        print()

        print("variables (root datasets, excluding coords):")
        var_names = [v for v in sorted(available) if v not in COORDS]
        print("  " + ", ".join(var_names) if var_names else "  (none)")
        print()

        print("core vars present:", _present(CORE_VARS, available))
        print("tide vars present:", _present(TIDE_VARS, available))
        print()

        if "land_mask" in available:
            lm = h5["land_mask"]
            sample = np.asarray(lm[: min(64, lm.shape[0]), : min(64, lm.shape[1])])
            uniq = np.unique(sample)
            print("land_mask:")
            print("  shape:", lm.shape, "dtype:", lm.dtype)
            print("  unique(sample 64x64):", uniq.tolist()[:20])
        print()

        # Minimal sanity sampling for currents (first time/depth) to catch NaN disasters.
        for v in ("uo", "vo"):
            if v not in available:
                continue
            d = h5[v]
            if d.ndim != 4:
                print(f"{v}: unexpected ndim={d.ndim} shape={d.shape}")
                continue
            block = np.asarray(d[0, 0, : min(64, d.shape[2]), : min(64, d.shape[3])], dtype=np.float64)
            print(f"{v}: shape={d.shape} dtype={d.dtype} sample64 {_format_range(block)} nan_count={int(np.isnan(block).sum())}")


if __name__ == "__main__":
    main()
