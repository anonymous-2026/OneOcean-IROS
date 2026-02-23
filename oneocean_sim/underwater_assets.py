from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UnderwaterAssetPack:
    sand_diffuse: Path
    sand_normal: Path | None
    rock_diffuse: Path
    rock_normal: Path | None

    def exists(self) -> bool:
        return self.sand_diffuse.exists() and self.rock_diffuse.exists()


def default_underwater_assets_dir() -> Path:
    return (Path("runs") / "_cache" / "polyhaven").resolve()


def resolve_underwater_asset_pack(assets_dir: Path | None = None) -> UnderwaterAssetPack:
    root = assets_dir.resolve() if assets_dir is not None else default_underwater_assets_dir()
    # Default Poly Haven IDs (see fetch CLI).
    sand_id = "aerial_sand"
    rock_id = "mossy_rock"
    sand_root = root / sand_id
    rock_root = root / rock_id

    # PolyHaven stores maps by name; we glob to avoid hardcoding exact filenames.
    sand_diffuse = next(
        iter(sorted(list(sand_root.glob("*_diff_*.png")) + list(sand_root.glob("*diff*.png")))),
        sand_root / "diffuse.png",
    )
    sand_normal = next(iter(sorted(sand_root.glob("*nor_gl*.png"))), None)
    rock_diffuse = next(
        iter(sorted(list(rock_root.glob("*_diff_*.png")) + list(rock_root.glob("*diff*.png")))),
        sand_diffuse,
    )
    rock_normal = next(iter(sorted(rock_root.glob("*nor_gl*.png"))), None)
    return UnderwaterAssetPack(
        sand_diffuse=sand_diffuse,
        sand_normal=sand_normal if sand_normal and sand_normal.exists() else None,
        rock_diffuse=rock_diffuse,
        rock_normal=rock_normal if rock_normal and rock_normal.exists() else None,
    )
