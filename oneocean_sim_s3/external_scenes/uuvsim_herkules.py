from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import urllib.error
import urllib.request

from .collada_loader import load_collada_triangles, mesh_to_obj


@dataclass(frozen=True)
class UUVSimHerkulesAssets:
    scene_id: str
    cache_dir: Path
    shipwreck_obj: Path
    shipwreck_dae: Path
    sources_md: Path


_SCENE_ID = "uuvsim_herkules_shipwreck"
_REPO = "uuvsimulator/uuv_simulator"
_PATH_SHIPWRECK_DAE = "uuv_gazebo_worlds/models/herkules_ship_wreck/meshes/herkules.dae"


def _download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "oneocean_sim_s3/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    output.write_bytes(data)


def ensure_uuvsim_herkules_shipwreck(
    *,
    cache_root: Path,
    max_faces: int = 4000,
) -> UUVSimHerkulesAssets:
    cache_dir = (cache_root / "external_scenes" / _SCENE_ID).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    sources_md = cache_dir / "SOURCES.md"
    if not sources_md.exists():
        sources_md.write_text(
            "\n".join(
                [
                    "# External Scene Sources — UUV Simulator (Herkules Shipwreck)",
                    "",
                    f"- Scene id: `{_SCENE_ID}`",
                    f"- Upstream repo: `{_REPO}` (Apache-2.0)",
                    f"- Asset path: `{_PATH_SHIPWRECK_DAE}`",
                    "- Policy: assets cached locally; not committed to Git.",
                    "",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    shipwreck_dae = cache_dir / "herkules.dae"
    shipwreck_obj = cache_dir / f"herkules_simplified_{int(max_faces)}f.obj"

    if not shipwreck_dae.exists():
        # Try both branches for robustness.
        urls = [
            f"https://raw.githubusercontent.com/{_REPO}/master/{_PATH_SHIPWRECK_DAE}",
            f"https://raw.githubusercontent.com/{_REPO}/main/{_PATH_SHIPWRECK_DAE}",
        ]
        last_err: Exception | None = None
        for url in urls:
            try:
                _download(url, shipwreck_dae)
                last_err = None
                break
            except (urllib.error.URLError, TimeoutError) as err:
                last_err = err
        if last_err is not None:
            raise RuntimeError(f"Failed to download shipwreck DAE from {_REPO}") from last_err

    if not shipwreck_obj.exists():
        mesh = load_collada_triangles(shipwreck_dae)
        if mesh.vertices.size == 0 or mesh.faces.size == 0:
            raise RuntimeError("Parsed empty shipwreck mesh from Collada")
        mesh_to_obj(mesh, output_obj=shipwreck_obj, center=True, max_faces=int(max_faces))

    return UUVSimHerkulesAssets(
        scene_id=_SCENE_ID,
        cache_dir=cache_dir,
        shipwreck_obj=shipwreck_obj,
        shipwreck_dae=shipwreck_dae,
        sources_md=sources_md,
    )

