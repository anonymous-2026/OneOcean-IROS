from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class StageGenConfig:
    dataset_path: Path
    output_dir: Path
    horizontal_scale: float = 0.01
    vertical_scale: float = 0.01
    floor_offset_m: float = 6.0
    stride: int = 1
    obstacle_count: int = 14
    obstacle_radius_m: float = 3.0
    obstacle_height_m: float = 2.5
    seed: int = 0
    seafloor_diffuse_texture: Path | None = None


def _local_xz_from_latlon(
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
    lat0_deg: float,
    lon0_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Equirectangular local tangent approximation (meters)."""
    lat0_rad = math.radians(float(lat0_deg))
    deg2rad = math.pi / 180.0
    dlat = (latitude_deg.astype(np.float64) - float(lat0_deg)) * deg2rad
    dlon = (longitude_deg.astype(np.float64) - float(lon0_deg)) * deg2rad
    x = EARTH_RADIUS_M * math.cos(lat0_rad) * dlon
    z = EARTH_RADIUS_M * dlat
    return x, z


def _iter_grid_faces(n_rows: int, n_cols: int, vertex_offset: int = 0) -> Iterable[tuple[int, int, int]]:
    """Yield CCW faces for an (n_rows x n_cols) grid of vertices."""
    for i in range(n_rows - 1):
        for j in range(n_cols - 1):
            a = vertex_offset + (i * n_cols + j) + 1
            b = vertex_offset + ((i + 1) * n_cols + j) + 1
            c = vertex_offset + ((i + 1) * n_cols + (j + 1)) + 1
            d = vertex_offset + (i * n_cols + (j + 1)) + 1
            yield (a, b, c)
            yield (a, c, d)


def _write_obj(
    obj_path: Path,
    mtl_name: str,
    vertices: list[tuple[float, float, float]],
    uvs: list[tuple[float, float]],
    normals: list[tuple[float, float, float]],
    seafloor_faces: list[tuple[int, int, int]],
    obstacle_faces: list[tuple[int, int, int]],
) -> None:
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    with obj_path.open("w", encoding="utf-8") as f:
        f.write(f"mtllib {mtl_name}\n")
        f.write("o seafloor\n")
        for x, y, z in vertices:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for u, v in uvs:
            f.write(f"vt {u:.6f} {v:.6f}\n")
        for nx, ny, nz in normals:
            f.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")
        f.write("usemtl seafloor\n")
        for a, b, c in seafloor_faces:
            f.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")
        if obstacle_faces:
            f.write("o obstacles\n")
            f.write("usemtl obstacle\n")
            for a, b, c in obstacle_faces:
                f.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")


def _write_mtl(mtl_path: Path, seafloor_texture_name: str | None) -> None:
    mtl_path.parent.mkdir(parents=True, exist_ok=True)
    with mtl_path.open("w", encoding="utf-8") as f:
        f.write("newmtl seafloor\n")
        f.write("Ka 0.09 0.11 0.12\n")
        f.write("Kd 0.75 0.75 0.75\n")
        f.write("Ks 0.03 0.03 0.03\n")
        f.write("Ns 12.0\n")
        f.write("d 1.0\n")
        f.write("illum 2\n\n")
        if seafloor_texture_name:
            f.write(f"map_Kd {seafloor_texture_name}\n\n")
        f.write("newmtl obstacle\n")
        f.write("Ka 0.08 0.07 0.06\n")
        f.write("Kd 0.18 0.16 0.14\n")
        f.write("Ks 0.02 0.02 0.02\n")
        f.write("Ns 8.0\n")
        f.write("d 1.0\n")
        f.write("illum 2\n")


def build_underwater_stage(config: StageGenConfig) -> dict[str, Any]:
    ds_path = config.dataset_path.expanduser().resolve()
    if not ds_path.exists():
        raise FileNotFoundError(f"Dataset not found: {ds_path}")
    out_dir = config.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import h5py  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency 'h5py'. Run this command using an interpreter that has h5py installed "
            "(e.g., /home/shuaijun/miniconda3/bin/python)."
        ) from exc

    with h5py.File(ds_path, "r") as f:
        latitude = np.asarray(f["latitude"][:], dtype=np.float64)
        longitude = np.asarray(f["longitude"][:], dtype=np.float64)
        elevation = np.asarray(f["elevation"][:], dtype=np.float64)

    stride = max(1, int(config.stride))
    latitude_s = latitude[::stride]
    longitude_s = longitude[::stride]
    elevation_s = elevation[::stride, ::stride]
    n_lat = int(latitude_s.size)
    n_lon = int(longitude_s.size)
    if elevation_s.shape != (n_lat, n_lon):
        raise ValueError(
            f"Unexpected elevation shape after stride={stride}: {elevation_s.shape} vs {(n_lat, n_lon)}"
        )

    lat0 = float(np.mean(latitude_s))
    lon0 = float(np.mean(longitude_s))
    x_lon_m, _ = _local_xz_from_latlon(np.full_like(longitude_s, lat0), longitude_s, lat0, lon0)
    _, z_lat_m = _local_xz_from_latlon(latitude_s, np.full_like(latitude_s, lon0), lat0, lon0)
    x_lon = x_lon_m * float(config.horizontal_scale)
    z_lat = z_lat_m * float(config.horizontal_scale)

    elev_max = float(np.nanmax(elevation_s))
    y = (elevation_s - elev_max) * float(config.vertical_scale) - float(config.floor_offset_m)
    y = np.nan_to_num(y, nan=-float(config.floor_offset_m))

    vertices: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    normals: list[tuple[float, float, float]] = []

    # Approximate vertex normals from local slopes (heightfield-style).
    dy_dx = np.zeros_like(y, dtype=np.float64)
    dy_dz = np.zeros_like(y, dtype=np.float64)
    for j in range(n_lon):
        j0 = max(0, j - 1)
        j1 = min(n_lon - 1, j + 1)
        denom = float(x_lon[j1] - x_lon[j0])
        if abs(denom) < 1e-9:
            continue
        dy_dx[:, j] = (y[:, j1] - y[:, j0]) / denom
    for i in range(n_lat):
        i0 = max(0, i - 1)
        i1 = min(n_lat - 1, i + 1)
        denom = float(z_lat[i1] - z_lat[i0])
        if abs(denom) < 1e-9:
            continue
        dy_dz[i, :] = (y[i1, :] - y[i0, :]) / denom

    nvec = np.stack([-dy_dx, np.ones_like(y, dtype=np.float64), -dy_dz], axis=-1)
    n_norm = np.linalg.norm(nvec, axis=-1, keepdims=True)
    nvec = nvec / np.clip(n_norm, 1e-9, None)
    for i in range(n_lat):
        for j in range(n_lon):
            vertices.append((float(x_lon[j]), float(y[i, j]), float(z_lat[i])))
            u = float(j) / max(1.0, float(n_lon - 1))
            v = float(i) / max(1.0, float(n_lat - 1))
            uvs.append((u, v))
            normals.append((float(nvec[i, j, 0]), float(nvec[i, j, 1]), float(nvec[i, j, 2])))

    seafloor_faces = list(_iter_grid_faces(n_rows=n_lat, n_cols=n_lon, vertex_offset=0))
    x_min, x_max = float(np.min(x_lon)), float(np.max(x_lon))
    z_min, z_max = float(np.min(z_lat)), float(np.max(z_lat))

    rng = np.random.default_rng(int(config.seed))
    obstacle_faces: list[tuple[int, int, int]] = []
    obstacle_specs: list[dict[str, Any]] = []
    if config.obstacle_count > 0 and n_lat >= 8 and n_lon >= 8:
        max_tries = int(config.obstacle_count) * 20
        chosen: list[tuple[int, int]] = []
        for _ in range(max_tries):
            if len(chosen) >= int(config.obstacle_count):
                break
            ii = int(rng.integers(2, n_lat - 2))
            jj = int(rng.integers(2, n_lon - 2))
            if any(abs(ii - pi) + abs(jj - pj) < 6 for pi, pj in chosen):
                continue
            chosen.append((ii, jj))

        for (ii, jj) in chosen:
            cx = float(x_lon[jj])
            cz = float(z_lat[ii])
            cy = float(y[ii, jj])
            r = float(config.obstacle_radius_m)
            h = float(config.obstacle_height_m)

            v0 = len(vertices) + 1
            vertices.extend(
                [
                    (cx - r, cy, cz - r),
                    (cx - r, cy, cz + r),
                    (cx + r, cy, cz + r),
                    (cx + r, cy, cz - r),
                    (cx, cy + h, cz),
                ]
            )
            for vx, vy, vz in vertices[-5:]:
                uu = (float(vx) - x_min) / max(1e-6, x_max - x_min)
                vv = (float(vz) - z_min) / max(1e-6, z_max - z_min)
                uvs.append((float(np.clip(uu, 0.0, 1.0)), float(np.clip(vv, 0.0, 1.0))))
                normals.append((0.0, 1.0, 0.0))
            a, b, c, d, top = v0, v0 + 1, v0 + 2, v0 + 3, v0 + 4
            obstacle_faces.extend(
                [
                    (a, b, top),
                    (b, c, top),
                    (c, d, top),
                    (d, a, top),
                    (a, d, c),
                    (a, c, b),
                ]
            )
            obstacle_specs.append(
                {
                    "grid_lat_idx": int(ii),
                    "grid_lon_idx": int(jj),
                    "center_xyz": [cx, cy, cz],
                    "radius_m": r,
                    "height_m": h,
                }
            )

    obj_path = out_dir / "underwater_stage.obj"
    mtl_path = out_dir / "underwater_stage.mtl"
    seafloor_texture_name = None
    if config.seafloor_diffuse_texture is not None:
        texture_path = Path(config.seafloor_diffuse_texture).expanduser().resolve()
        if not texture_path.exists():
            raise FileNotFoundError(f"Seafloor texture not found: {texture_path}")
        texture_dst = out_dir / texture_path.name
        if texture_dst.resolve() != texture_path:
            shutil.copyfile(texture_path, texture_dst)
        seafloor_texture_name = texture_dst.name

    _write_mtl(mtl_path, seafloor_texture_name=seafloor_texture_name)
    _write_obj(
        obj_path=obj_path,
        mtl_name=mtl_path.name,
        vertices=vertices,
        uvs=uvs,
        normals=normals,
        seafloor_faces=seafloor_faces,
        obstacle_faces=obstacle_faces,
    )

    meta = {
        "dataset_path": str(ds_path),
        "output_dir": str(out_dir),
        "horizontal_scale": float(config.horizontal_scale),
        "vertical_scale": float(config.vertical_scale),
        "floor_offset_m": float(config.floor_offset_m),
        "stride": int(stride),
        "origin_latlon_deg": [lat0, lon0],
        "elevation_max_m": elev_max,
        "seafloor_diffuse_texture": seafloor_texture_name or "",
        "grid": {"lat": int(n_lat), "lon": int(n_lon)},
        "extents_sim_m": {
            "x_min": float(np.min(x_lon)),
            "x_max": float(np.max(x_lon)),
            "z_min": float(np.min(z_lat)),
            "z_max": float(np.max(z_lat)),
            "y_min": float(np.min(y)),
            "y_max": float(np.max(y)),
        },
        "obstacles": obstacle_specs,
    }
    meta_path = out_dir / "underwater_stage_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "stage_obj": str(obj_path),
        "stage_mtl": str(mtl_path),
        "stage_meta": str(meta_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a Habitat-Sim stage OBJ from bathymetry in combined_environment.nc")
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to combined_environment.nc (run with an interpreter that has h5py).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for underwater_stage.obj/.mtl/.json.",
    )
    parser.add_argument("--horizontal-scale", type=float, default=0.01)
    parser.add_argument("--vertical-scale", type=float, default=0.01)
    parser.add_argument("--floor-offset-m", type=float, default=6.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--obstacle-count", type=int, default=14)
    parser.add_argument("--obstacle-radius-m", type=float, default=3.0)
    parser.add_argument("--obstacle-height-m", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seafloor-diffuse-texture",
        default="",
        help="Optional path to a diffuse texture image for the seafloor (copied into output-dir).",
    )
    args = parser.parse_args()

    cfg = StageGenConfig(
        dataset_path=Path(args.dataset_path),
        output_dir=Path(args.output_dir),
        horizontal_scale=float(args.horizontal_scale),
        vertical_scale=float(args.vertical_scale),
        floor_offset_m=float(args.floor_offset_m),
        stride=int(args.stride),
        obstacle_count=int(args.obstacle_count),
        obstacle_radius_m=float(args.obstacle_radius_m),
        obstacle_height_m=float(args.obstacle_height_m),
        seed=int(args.seed),
        seafloor_diffuse_texture=Path(args.seafloor_diffuse_texture).expanduser()
        if args.seafloor_diffuse_texture
        else None,
    )
    outputs = build_underwater_stage(cfg)
    print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
