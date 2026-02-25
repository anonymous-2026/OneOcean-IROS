from __future__ import annotations

import argparse
import json
import shutil
import ssl
import urllib.request
from dataclasses import dataclass
from pathlib import Path


OCEANSIM_SHA_DEFAULT = "1d66f6bf2d98dacef6ab3f91b2fa8743541361df"
RAW_BASE = "https://raw.githubusercontent.com/umfieldrobotics/OceanSim"
DEFAULT_DEST = Path("/home/shuaijun/isaacsim/extsUser/OceanSim")


@dataclass(frozen=True)
class OceanSimFile:
    path: str
    binary: bool = False


MINIMAL_EXTENSION_FILES: tuple[OceanSimFile, ...] = (
    OceanSimFile("config/extension.toml"),
    OceanSimFile("config/register_asset_path.py"),
    OceanSimFile("data/icon.png", binary=True),
    OceanSimFile("data/preview.png", binary=True),
    OceanSimFile("isaacsim/oceansim/modules/SensorExample_python/__init__.py"),
    OceanSimFile("isaacsim/oceansim/modules/SensorExample_python/extension.py"),
    OceanSimFile("isaacsim/oceansim/modules/SensorExample_python/global_variables.py"),
    OceanSimFile("isaacsim/oceansim/modules/SensorExample_python/scenario.py"),
    OceanSimFile("isaacsim/oceansim/modules/SensorExample_python/ui_builder.py"),
    OceanSimFile("isaacsim/oceansim/modules/colorpicker_python/__init__.py"),
    OceanSimFile("isaacsim/oceansim/modules/colorpicker_python/extension.py"),
    OceanSimFile("isaacsim/oceansim/modules/colorpicker_python/global_variables.py"),
    OceanSimFile("isaacsim/oceansim/modules/colorpicker_python/scenario.py"),
    OceanSimFile("isaacsim/oceansim/modules/colorpicker_python/ui_builder.py"),
    OceanSimFile("isaacsim/oceansim/sensors/BarometerSensor.py"),
    OceanSimFile("isaacsim/oceansim/sensors/DVLsensor.py"),
    OceanSimFile("isaacsim/oceansim/sensors/ImagingSonarSensor.py"),
    OceanSimFile("isaacsim/oceansim/sensors/UW_Camera.py"),
    OceanSimFile("isaacsim/oceansim/utils/ImagingSonar_kernels.py"),
    OceanSimFile("isaacsim/oceansim/utils/MultivariateNormal.py"),
    OceanSimFile("isaacsim/oceansim/utils/MultivariateUniform.py"),
    OceanSimFile("isaacsim/oceansim/utils/UWrenderer_utils.py"),
    OceanSimFile("isaacsim/oceansim/utils/asset_path.json"),
    OceanSimFile("isaacsim/oceansim/utils/assets_utils.py"),
    OceanSimFile("isaacsim/oceansim/utils/keyboard_cmd.py"),
)


def _ssl_context() -> ssl.SSLContext | None:
    try:
        import certifi  # type: ignore

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None


def _download(url: str, *, binary: bool) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "OneOcean-IROS-H5/1.0"})
    ctx = _ssl_context()
    kwargs = {"timeout": 120}
    if ctx is not None:
        kwargs["context"] = ctx
    with urllib.request.urlopen(req, **kwargs) as resp:  # nosec - controlled URL
        data = resp.read()
    if not binary:
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            pass
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch a minimal OceanSim extension snapshot into IsaacSim extsUser.")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--sha", type=str, default=OCEANSIM_SHA_DEFAULT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dest: Path = args.dest.expanduser().resolve()
    sha: str = args.sha.strip()
    if not sha:
        raise SystemExit("--sha is required")

    if dest.exists():
        if not args.overwrite:
            raise SystemExit(f"Destination exists: {dest} (use --overwrite to replace)")
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    fetched: list[dict[str, object]] = []
    for entry in MINIMAL_EXTENSION_FILES:
        url = f"{RAW_BASE}/{sha}/{entry.path}"
        data = _download(url, binary=entry.binary)
        out_path = dest / entry.path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        fetched.append({"path": entry.path, "url": url, "bytes": len(data), "binary": entry.binary})
        print(f"[ok] {entry.path} ({len(data)} bytes)")

    manifest = {"upstream": "umfieldrobotics/OceanSim", "sha": sha, "files": fetched}
    (dest / "ONEOCEAN_FETCH_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[ok] Wrote manifest: {dest / 'ONEOCEAN_FETCH_MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

