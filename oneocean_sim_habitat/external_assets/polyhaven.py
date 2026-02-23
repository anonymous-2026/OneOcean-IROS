from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import ssl
import urllib.request
from typing import Any, Mapping


POLYHAVEN_API = "https://api.polyhaven.com"
POLYHAVEN_ASSET_PAGE = "https://polyhaven.com/a"
DEFAULT_USER_AGENT = "OneOcean-IROS-S2/1.0"


def _ssl_context() -> ssl.SSLContext | None:
    try:
        import certifi  # type: ignore

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None


def _request_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
    ctx = _ssl_context()
    kwargs: dict[str, Any] = {"timeout": 60}
    if ctx is not None:
        kwargs["context"] = ctx
    with urllib.request.urlopen(req, **kwargs) as resp:  # nosec - controlled URL
        return json.load(resp)


def _download(url: str, dst: Path, *, expected_md5: str | None = None, overwrite: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        if expected_md5 and dst.is_file():
            md5 = hashlib.md5(dst.read_bytes()).hexdigest()  # nosec - integrity check only
            if md5 == expected_md5:
                return
        else:
            return

    req = urllib.request.Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
    ctx = _ssl_context()
    kwargs: dict[str, Any] = {"timeout": 120}
    if ctx is not None:
        kwargs["context"] = ctx
    with urllib.request.urlopen(req, **kwargs) as resp:  # nosec - controlled URL
        data = resp.read()
    dst.write_bytes(data)

    if expected_md5:
        md5 = hashlib.md5(data).hexdigest()  # nosec - integrity check only
        if md5 != expected_md5:
            raise RuntimeError(f"MD5 mismatch for {dst.name}: expected {expected_md5}, got {md5}")


@dataclass(frozen=True)
class PolyHavenAsset:
    asset_id: str
    asset_type: str
    license: str
    source_url: str
    local_root: Path
    manifest_path: Path
    primary_path: Path


def download_texture_maps(
    *,
    asset_id: str,
    out_dir: Path,
    resolution: str = "1k",
    maps: tuple[str, ...] = ("Diffuse",),
    file_format: str = "png",
    overwrite: bool = False,
) -> dict[str, Path]:
    files = _request_json(f"{POLYHAVEN_API}/files/{asset_id}")
    out: dict[str, Path] = {}
    for key in maps:
        if key not in files:
            raise KeyError(f"PolyHaven texture missing key={key} for asset={asset_id}")
        if resolution not in files[key]:
            raise KeyError(f"PolyHaven texture missing resolution={resolution} for key={key} asset={asset_id}")
        if file_format not in files[key][resolution]:
            raise KeyError(f"PolyHaven texture missing format={file_format} for key={key} asset={asset_id}")
        entry = files[key][resolution][file_format]
        url = str(entry["url"])
        md5 = str(entry.get("md5") or "")
        name = Path(url).name
        dst = out_dir / asset_id / name
        _download(url, dst, expected_md5=md5 or None, overwrite=overwrite)
        out[key] = dst
    return out


def download_gltf_model(
    *,
    asset_id: str,
    out_dir: Path,
    resolution: str = "1k",
    overwrite: bool = False,
) -> Path:
    files = _request_json(f"{POLYHAVEN_API}/files/{asset_id}")
    if "gltf" not in files:
        raise KeyError(f"PolyHaven model has no gltf entry: {asset_id}")
    if resolution not in files["gltf"]:
        raise KeyError(f"PolyHaven model missing resolution={resolution}: {asset_id}")
    gltf_entry = files["gltf"][resolution]["gltf"]
    gltf_url = str(gltf_entry["url"])
    gltf_md5 = str(gltf_entry.get("md5") or "")

    root = out_dir / asset_id
    gltf_path = root / Path(gltf_url).name
    _download(gltf_url, gltf_path, expected_md5=gltf_md5 or None, overwrite=overwrite)

    includes: Mapping[str, Mapping[str, Any]] = gltf_entry.get("include") or {}
    for rel_path, meta in includes.items():
        url = str(meta["url"])
        md5 = str(meta.get("md5") or "")
        dst = root / rel_path
        _download(url, dst, expected_md5=md5 or None, overwrite=overwrite)

    return gltf_path


def write_polyhaven_manifest(
    *,
    asset_id: str,
    local_root: Path,
    primary_path: Path,
    manifest_path: Path,
) -> None:
    info = _request_json(f"{POLYHAVEN_API}/info/{asset_id}")
    payload = {
        "asset_id": asset_id,
        "name": info.get("name", asset_id),
        "type": info.get("type"),
        "categories": info.get("categories"),
        "tags": info.get("tags"),
        "license": "CC0 (Poly Haven)",
        "source_url": f"{POLYHAVEN_ASSET_PAGE}/{asset_id}",
        "downloaded_root": str(local_root),
        "primary_path": str(primary_path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ensure_underwater_asset_pack(
    *,
    out_dir: Path,
    resolution: str = "1k",
    sand_texture_id: str = "aerial_sand",
    rock_model_id: str = "rock_07",
    overwrite: bool = False,
) -> dict[str, PolyHavenAsset]:
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sand_maps = download_texture_maps(
        asset_id=sand_texture_id,
        out_dir=out_dir,
        resolution=resolution,
        maps=("Diffuse", "nor_gl"),
        file_format="png",
        overwrite=overwrite,
    )
    sand_primary = sand_maps["Diffuse"]
    sand_manifest = out_dir / sand_texture_id / "polyhaven_manifest.json"
    write_polyhaven_manifest(
        asset_id=sand_texture_id,
        local_root=out_dir / sand_texture_id,
        primary_path=sand_primary,
        manifest_path=sand_manifest,
    )

    rock_gltf = download_gltf_model(
        asset_id=rock_model_id,
        out_dir=out_dir,
        resolution=resolution,
        overwrite=overwrite,
    )
    rock_manifest = out_dir / rock_model_id / "polyhaven_manifest.json"
    write_polyhaven_manifest(
        asset_id=rock_model_id,
        local_root=out_dir / rock_model_id,
        primary_path=rock_gltf,
        manifest_path=rock_manifest,
    )

    return {
        "sand_texture": PolyHavenAsset(
            asset_id=sand_texture_id,
            asset_type="texture",
            license="CC0 (Poly Haven)",
            source_url=f"{POLYHAVEN_ASSET_PAGE}/{sand_texture_id}",
            local_root=out_dir / sand_texture_id,
            manifest_path=sand_manifest,
            primary_path=sand_primary,
        ),
        "rock_model": PolyHavenAsset(
            asset_id=rock_model_id,
            asset_type="model_gltf",
            license="CC0 (Poly Haven)",
            source_url=f"{POLYHAVEN_ASSET_PAGE}/{rock_model_id}",
            local_root=out_dir / rock_model_id,
            manifest_path=rock_manifest,
            primary_path=rock_gltf,
        ),
    }


def copy_texture_to_stage_dir(texture_path: Path, stage_dir: Path, *, overwrite: bool = False) -> Path:
    stage_dir.mkdir(parents=True, exist_ok=True)
    dst = stage_dir / texture_path.name
    if dst.exists() and not overwrite:
        return dst
    shutil.copyfile(texture_path, dst)
    return dst
