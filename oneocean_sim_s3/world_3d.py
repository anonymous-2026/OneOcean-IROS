from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin
from pathlib import Path
from typing import Optional

import numpy as np


METERS_PER_DEG_LAT = 111_132.0


def meters_per_deg_lon(latitude_deg: float) -> float:
    return METERS_PER_DEG_LAT * max(0.1, cos(radians(latitude_deg)))


def latlon_to_xy(
    lat: float, lon: float, lat0: float, lon0: float, xy_scale: float = 1.0
) -> tuple[float, float]:
    if not (0.0 < xy_scale <= 1.0):
        raise ValueError("xy_scale must be in (0, 1]")
    x_m = (lon - lon0) * meters_per_deg_lon(lat0) * xy_scale
    y_m = (lat - lat0) * METERS_PER_DEG_LAT * xy_scale
    return x_m, y_m


def xy_to_latlon(
    x_m: float, y_m: float, lat0: float, lon0: float, xy_scale: float = 1.0
) -> tuple[float, float]:
    if not (0.0 < xy_scale <= 1.0):
        raise ValueError("xy_scale must be in (0, 1]")
    x_real = float(x_m) / float(xy_scale)
    y_real = float(y_m) / float(xy_scale)
    lat = lat0 + (y_real / METERS_PER_DEG_LAT)
    lon = lon0 + (x_real / meters_per_deg_lon(lat0))
    return lat, lon


@dataclass(frozen=True)
class TerrainSpec:
    grid_size: int = 33
    z_min_m: float = -30.0
    z_max_m: float = -5.0
    # Shrinks latitude/longitude meters into simulation meters to keep the collision mesh
    # well-conditioned in SAPIEN (avoids kilometer-scale triangles).
    xy_scale: float = 0.02
    xy_margin_m: float = 8.0


@dataclass
class TerrainMesh:
    vertices_m: np.ndarray  # (N, 3)
    faces: np.ndarray  # (M, 3) int
    x_coords_m: np.ndarray  # (W,)
    y_coords_m: np.ndarray  # (H,)
    z_grid_m: np.ndarray  # (H, W)
    xy_scale: float
    origin_lat: float
    origin_lon: float
    lat_slice: tuple[int, int]
    lon_slice: tuple[int, int]
    elevation_min: float
    elevation_max: float
    z_min_m: float
    z_max_m: float

    @property
    def bounds_xy_m(self) -> tuple[float, float, float, float]:
        x_min = float(np.min(self.x_coords_m))
        x_max = float(np.max(self.x_coords_m))
        y_min = float(np.min(self.y_coords_m))
        y_max = float(np.max(self.y_coords_m))
        return x_min, x_max, y_min, y_max

    def height_at_xy(self, x_m: float, y_m: float) -> float:
        x_coords = self.x_coords_m
        y_coords = self.y_coords_m

        if x_m <= float(x_coords[0]):
            xi0 = 0
            xi1 = 1
        elif x_m >= float(x_coords[-1]):
            xi0 = len(x_coords) - 2
            xi1 = len(x_coords) - 1
        else:
            xi1 = int(np.searchsorted(x_coords, x_m, side="right"))
            xi0 = xi1 - 1

        if y_m <= float(y_coords[0]):
            yi0 = 0
            yi1 = 1
        elif y_m >= float(y_coords[-1]):
            yi0 = len(y_coords) - 2
            yi1 = len(y_coords) - 1
        else:
            yi1 = int(np.searchsorted(y_coords, y_m, side="right"))
            yi0 = yi1 - 1

        x0 = float(x_coords[xi0])
        x1 = float(x_coords[xi1])
        y0 = float(y_coords[yi0])
        y1 = float(y_coords[yi1])

        tx = 0.0 if x1 == x0 else float((x_m - x0) / (x1 - x0))
        ty = 0.0 if y1 == y0 else float((y_m - y0) / (y1 - y0))

        z00 = float(self.z_grid_m[yi0, xi0])
        z10 = float(self.z_grid_m[yi0, xi1])
        z01 = float(self.z_grid_m[yi1, xi0])
        z11 = float(self.z_grid_m[yi1, xi1])

        z0 = (1.0 - tx) * z00 + tx * z10
        z1 = (1.0 - tx) * z01 + tx * z11
        return (1.0 - ty) * z0 + ty * z1


@dataclass(frozen=True)
class ObstacleSpec:
    shape: str  # sphere or box
    position_m: tuple[float, float, float]
    radius_m: float
    half_size_m: Optional[tuple[float, float, float]] = None
    color_rgb: tuple[int, int, int] = (120, 120, 120)


def build_terrain_mesh(
    *,
    latitude_values: np.ndarray,
    longitude_values: np.ndarray,
    elevation_grid: np.ndarray,
    land_mask: Optional[np.ndarray],
    center_lat: float,
    center_lon: float,
    spec: TerrainSpec,
    output_obj: Optional[Path] = None,
) -> TerrainMesh:
    if elevation_grid.ndim != 2:
        raise ValueError(f"Expected elevation_grid (H,W), got shape {elevation_grid.shape}")
    grid_size = int(spec.grid_size)
    if grid_size < 5 or grid_size % 2 == 0:
        raise ValueError("TerrainSpec.grid_size must be an odd integer >= 5")

    lat_idx = int(np.argmin(np.abs(latitude_values - float(center_lat))))
    lon_idx = int(np.argmin(np.abs(longitude_values - float(center_lon))))

    half = grid_size // 2
    lat_start = max(0, lat_idx - half)
    lon_start = max(0, lon_idx - half)
    lat_end = min(len(latitude_values), lat_start + grid_size)
    lon_end = min(len(longitude_values), lon_start + grid_size)
    lat_start = max(0, lat_end - grid_size)
    lon_start = max(0, lon_end - grid_size)

    lat_subset = np.asarray(latitude_values[lat_start:lat_end], dtype=np.float64)
    lon_subset = np.asarray(longitude_values[lon_start:lon_end], dtype=np.float64)

    elev_subset = np.asarray(elevation_grid[lat_start:lat_end, lon_start:lon_end], dtype=np.float64)
    elev_subset = np.nan_to_num(elev_subset, nan=float(np.nanmin(elev_subset)))
    if land_mask is not None:
        mask_subset = np.asarray(land_mask[lat_start:lat_end, lon_start:lon_end], dtype=np.float64)
        invalid = np.isnan(mask_subset) | (np.round(mask_subset).astype(int) != 0)
        if np.any(invalid):
            elev_min = float(np.nanmin(elev_subset))
            elev_subset = np.where(invalid, elev_min, elev_subset)

    elev_min = float(np.min(elev_subset))
    elev_max = float(np.max(elev_subset))
    z_min = float(spec.z_min_m)
    z_max = float(spec.z_max_m)
    if elev_max - elev_min < 1e-6:
        z_grid = np.full_like(elev_subset, z_min, dtype=np.float64)
    else:
        t = (elev_subset - elev_min) / (elev_max - elev_min)
        z_grid = z_min + t * (z_max - z_min)

    origin_lat = float(lat_subset[len(lat_subset) // 2])
    origin_lon = float(lon_subset[len(lon_subset) // 2])
    xy_scale = float(spec.xy_scale)
    if not (0.0 < xy_scale <= 1.0):
        raise ValueError("TerrainSpec.xy_scale must be in (0, 1]")

    x_coords = np.asarray(
        [(lon - origin_lon) * meters_per_deg_lon(origin_lat) * xy_scale for lon in lon_subset],
        dtype=np.float64,
    )
    y_coords = np.asarray([(lat - origin_lat) * METERS_PER_DEG_LAT * xy_scale for lat in lat_subset], dtype=np.float64)

    vertices = []
    for yi, y in enumerate(y_coords):
        for xi, x in enumerate(x_coords):
            vertices.append((float(x), float(y), float(z_grid[yi, xi])))
    vertices_arr = np.asarray(vertices, dtype=np.float64)

    faces = []
    w = len(x_coords)
    h = len(y_coords)
    for yi in range(h - 1):
        for xi in range(w - 1):
            v0 = yi * w + xi
            v1 = yi * w + (xi + 1)
            v2 = (yi + 1) * w + xi
            v3 = (yi + 1) * w + (xi + 1)
            faces.append((v0, v2, v1))
            faces.append((v1, v2, v3))
    faces_arr = np.asarray(faces, dtype=np.int32)

    if output_obj is not None:
        output_obj.parent.mkdir(parents=True, exist_ok=True)
        with output_obj.open("w", encoding="utf-8") as file:
            for x, y, z in vertices_arr:
                file.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            for a, b, c in faces_arr:
                file.write(f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}\n")

    return TerrainMesh(
        vertices_m=vertices_arr,
        faces=faces_arr,
        x_coords_m=x_coords,
        y_coords_m=y_coords,
        z_grid_m=z_grid.astype(np.float64),
        xy_scale=xy_scale,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        lat_slice=(lat_start, lat_end),
        lon_slice=(lon_start, lon_end),
        elevation_min=elev_min,
        elevation_max=elev_max,
        z_min_m=z_min,
        z_max_m=z_max,
    )


def sample_goal_xy(
    *,
    rng: np.random.Generator,
    terrain: TerrainMesh,
    min_radius_m: float,
    max_radius_m: float,
) -> tuple[float, float]:
    x_min, x_max, y_min, y_max = terrain.bounds_xy_m
    margin = 0.5 * float(terrain.x_coords_m[1] - terrain.x_coords_m[0]) if len(terrain.x_coords_m) > 1 else 5.0

    for _ in range(200):
        r = float(rng.uniform(min_radius_m, max_radius_m))
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        x = r * cos(angle)
        y = r * sin(angle)
        if (x_min + margin) <= x <= (x_max - margin) and (y_min + margin) <= y <= (y_max - margin):
            return float(x), float(y)
    return float(np.clip(0.8 * max_radius_m, x_min + margin, x_max - margin)), 0.0


def build_obstacles(
    *,
    rng: np.random.Generator,
    terrain: TerrainMesh,
    count: int,
    radius_range_m: tuple[float, float] = (1.2, 2.8),
    start_xy: tuple[float, float] = (0.0, 0.0),
    goal_xy: tuple[float, float] = (30.0, 0.0),
    min_clearance_m: float = 6.0,
) -> list[ObstacleSpec]:
    x_min, x_max, y_min, y_max = terrain.bounds_xy_m
    margin = float(terrain.x_coords_m[1] - terrain.x_coords_m[0]) if len(terrain.x_coords_m) > 1 else 5.0
    x_min += margin
    x_max -= margin
    y_min += margin
    y_max -= margin

    obstacles: list[ObstacleSpec] = []
    for _ in range(int(count)):
        for _attempt in range(300):
            x = float(rng.uniform(x_min, x_max))
            y = float(rng.uniform(y_min, y_max))
            radius = float(rng.uniform(radius_range_m[0], radius_range_m[1]))
            z = float(terrain.height_at_xy(x, y) + radius)

            if np.hypot(x - start_xy[0], y - start_xy[1]) < min_clearance_m:
                continue
            if np.hypot(x - goal_xy[0], y - goal_xy[1]) < min_clearance_m:
                continue
            if any(np.hypot(x - o.position_m[0], y - o.position_m[1]) < (o.radius_m + radius + 1.0) for o in obstacles):
                continue

            obstacles.append(
                ObstacleSpec(
                    shape="sphere",
                    position_m=(x, y, z),
                    radius_m=radius,
                    color_rgb=(90, 90, 90),
                )
            )
            break
    return obstacles
