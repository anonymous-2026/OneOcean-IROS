from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import urllib.error
import urllib.request

import numpy as np
import trimesh


@dataclass(frozen=True)
class PolyHavenModelAssets:
    provider: str
    asset_id: str
    cache_dir: Path
    gltf_path: Path
    obj_path: Path
    sources_md: Path
    resolution: str
    max_faces: int


_API_FILES = "https://api.polyhaven.com/files/{asset_id}?t=models"


def _http_get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "oneocean_sim_s3/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "oneocean_sim_s3/1.0"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = resp.read()
    output.write_bytes(data)


def _simplify_faces(mesh: trimesh.Trimesh, max_faces: int) -> trimesh.Trimesh:
    if max_faces <= 0 or len(mesh.faces) <= max_faces:
        return mesh
    idx = np.linspace(0, len(mesh.faces) - 1, num=int(max_faces), dtype=np.int64)
    faces = mesh.faces[idx]
    used = np.unique(faces.reshape(-1))
    remap = {int(old): int(i) for i, old in enumerate(used.tolist())}
    vertices = mesh.vertices[used]
    faces_remap = np.vectorize(lambda x: remap[int(x)], otypes=[np.int64])(faces).astype(np.int64)
    out = trimesh.Trimesh(vertices=vertices, faces=faces_remap, process=False)
    return out


def _scene_to_single_mesh(scene: object) -> trimesh.Trimesh:
    if isinstance(scene, trimesh.Trimesh):
        return scene
    if isinstance(scene, trimesh.Scene):
        meshes = []
        for g in scene.geometry.values():
            if isinstance(g, trimesh.Trimesh) and len(g.faces):
                meshes.append(g)
        if not meshes:
            raise RuntimeError("PolyHaven glTF contained no triangle geometry")
        return trimesh.util.concatenate(meshes)
    raise TypeError(f"Unsupported trimesh load type: {type(scene)}")


def ensure_polyhaven_model_obj(
    *,
    asset_id: str,
    cache_root: Path,
    resolution: str = "1k",
    max_faces: int = 12000,
    center: bool = True,
    scale: float = 1.0,
) -> PolyHavenModelAssets:
    provider = "polyhaven"
    cache_dir = (cache_root / provider / asset_id).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    sources_md = cache_dir / "SOURCES.md"
    if not sources_md.exists():
        sources_md.write_text(
            "\n".join(
                [
                    "# External Scene Sources — Poly Haven models",
                    "",
                    f"- Provider: `{provider}`",
                    f"- Asset id: `{asset_id}`",
                    "- License: CC0 (per Poly Haven asset policy).",
                    "- Policy: assets cached locally; not committed to Git.",
                    "",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    files = _http_get_json(_API_FILES.format(asset_id=asset_id))
    gltf_info = (files.get("gltf") or {}).get(str(resolution)) or {}
    gltf_entry = gltf_info.get("gltf")
    if not isinstance(gltf_entry, dict) or "url" not in gltf_entry:
        raise RuntimeError(f"PolyHaven files API did not return glTF for {asset_id} at {resolution}")

    gltf_url = str(gltf_entry["url"])
    gltf_path = cache_dir / f"{asset_id}_{resolution}.gltf"
    if not gltf_path.exists():
        _download(gltf_url, gltf_path)

    include = gltf_entry.get("include") or {}
    if isinstance(include, dict):
        for rel, meta in include.items():
            if not isinstance(meta, dict) or "url" not in meta:
                continue
            rel_path = Path(str(rel))
            out = cache_dir / rel_path
            if out.exists():
                continue
            try:
                _download(str(meta["url"]), out)
            except (urllib.error.URLError, TimeoutError):
                # Textures are not required for geometry export; skip if transient.
                continue

    obj_path = cache_dir / f"{asset_id}_{resolution}_{int(max_faces)}f.obj"
    if not obj_path.exists():
        loaded = trimesh.load(gltf_path, force="scene")
        mesh = _scene_to_single_mesh(loaded)
        mesh = _simplify_faces(mesh, int(max_faces))
        if center:
            bounds = mesh.bounds
            center_pt = 0.5 * (bounds[0] + bounds[1])
            mesh = mesh.copy()
            mesh.vertices = mesh.vertices - center_pt[None, :]
        if abs(float(scale) - 1.0) > 1e-9:
            mesh = mesh.copy()
            mesh.vertices = mesh.vertices * float(scale)
        obj_text = trimesh.exchange.obj.export_obj(mesh)
        obj_path.write_text(obj_text, encoding="utf-8")

    return PolyHavenModelAssets(
        provider=provider,
        asset_id=str(asset_id),
        cache_dir=cache_dir,
        gltf_path=gltf_path,
        obj_path=obj_path,
        sources_md=sources_md,
        resolution=str(resolution),
        max_faces=int(max_faces),
    )
