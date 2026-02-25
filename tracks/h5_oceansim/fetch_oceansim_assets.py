from __future__ import annotations

import argparse
import codecs
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import requests


# From OceanSim official installation guide:
# https://raw.githubusercontent.com/umfieldrobotics/OceanSim/main/docs/subsections/installation.md
OCEANSIM_ASSETS_ROOT_FOLDER_ID = "1qg4-Y_GMiybnLc1BFjx0DsWfR0AgeZzA"


@dataclass(frozen=True)
class DriveItem:
    id: str
    parent_id: str
    name: str
    mime: str

    @property
    def is_folder(self) -> bool:
        return self.mime == "application/vnd.google-apps.folder"


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print(f"[cmd] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, check=True)


def _decode_drive_ivd_from_html(html: str, *, folder_id: str) -> str:
    m = re.search(r"window\['_DRIVE_ivd'\]\s*=\s*'(?P<s>.*?)';if \(window\['_DRIVE_ivdc'\]\)", html, re.S)
    if not m:
        head = re.sub(r"\s+", " ", html[:300])
        raise RuntimeError(
            "Could not find window['_DRIVE_ivd'] payload in Google Drive folder HTML "
            f"(folder_id={folder_id}). head={head!r}"
        )
    # The payload uses \xNN escapes and some escaped slashes.
    return codecs.decode(m.group("s"), "unicode_escape")


def _list_folder_items(folder_id: str) -> list[DriveItem]:
    url = f"https://drive.google.com/drive/folders/{folder_id}?usp=sharing"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    ivd = _decode_drive_ivd_from_html(r.text, folder_id=folder_id)

    pat = re.compile(
        r'\["(?P<id>[A-Za-z0-9_-]{10,})",\["(?P<parent>[A-Za-z0-9_-]{10,})"\],"(?P<name>[^"]+)","(?P<mime>[^"]+)"'
    )
    items = []
    for fid, parent, name, mime in pat.findall(ivd):
        items.append(
            DriveItem(
                id=fid,
                parent_id=parent,
                name=name,
                mime=mime.replace("\\/", "/"),
            )
        )
    return [it for it in items if it.parent_id == folder_id]


def _download_file(file_id: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base = f"https://drive.google.com/uc?export=download&id={file_id}"
    tmp = out_path.with_suffix(out_path.suffix + ".part")

    def _curl_download(url: str) -> None:
        _run(
            [
                "curl",
                "-L",
                "--fail",
                "--retry",
                "5",
                "--retry-delay",
                "2",
                "-o",
                str(tmp),
                url,
            ]
        )

    _curl_download(base)

    # If we got an HTML confirmation page, extract confirm token and re-download.
    try:
        with open(tmp, "rb") as f:
            head = f.read(4096).decode("utf-8", errors="ignore")
        if "<html" in head.lower() or "confirm=" in head:
            m = re.search(r"confirm=([0-9A-Za-z_]+)", head)
            if m:
                token = m.group(1)
                _curl_download(base + f"&confirm={token}")
    except Exception:
        pass

    tmp.replace(out_path)


def _sync_folder(folder_id: str, out_dir: Path, *, depth: int = 0, max_depth: int = 10) -> None:
    if depth > max_depth:
        raise RuntimeError(f"Max recursion depth exceeded at folder {folder_id}")

    items = _list_folder_items(folder_id)
    for it in items:
        dst = out_dir / it.name
        if it.is_folder:
            dst.mkdir(parents=True, exist_ok=True)
            _sync_folder(it.id, dst, depth=depth + 1, max_depth=max_depth)
        else:
            if dst.exists() and dst.stat().st_size > 0:
                continue
            print(f"[info] downloading {it.name} ({it.mime})")
            _download_file(it.id, dst)


def _register_asset_path(oceansim_ext_root: Path, asset_root: Path) -> None:
    register = oceansim_ext_root / "config" / "register_asset_path.py"
    if not register.is_file():
        raise FileNotFoundError(f"register_asset_path.py not found: {register}")
    _run(["python3", str(register), str(asset_root)])


def main() -> int:
    parser = argparse.ArgumentParser(description="Download + register OceanSim assets for local use (no gdown).")
    parser.add_argument(
        "--download_dir",
        type=Path,
        default=Path("runs/_cache/oceansim_assets"),
        help="Where to download (local-only; keep out of git).",
    )
    parser.add_argument(
        "--oceansim_ext_root",
        type=Path,
        default=Path("/home/shuaijun/isaacsim/extsUser/OceanSim"),
        help="OceanSim extension root (extsUser/OceanSim).",
    )
    parser.add_argument(
        "--root_folder_id",
        type=str,
        default=OCEANSIM_ASSETS_ROOT_FOLDER_ID,
        help="Google Drive folder id for OceanSim assets root.",
    )
    parser.add_argument(
        "--include",
        type=str,
        default="collected_MHL,collected_rock",
        help="Comma-separated top-level subfolders to download (keeps download small).",
    )
    parser.add_argument("--force", action="store_true", help="Delete download_dir/OceanSim_assets before downloading.")
    args = parser.parse_args()

    download_dir = args.download_dir.expanduser().resolve()
    oceansim_ext_root = args.oceansim_ext_root.expanduser().resolve()
    asset_root = download_dir / "OceanSim_assets"

    if args.force and asset_root.exists():
        print(f"[warn] removing existing assets dir: {asset_root}")
        shutil.rmtree(asset_root)
    asset_root.mkdir(parents=True, exist_ok=True)

    if not oceansim_ext_root.is_dir():
        raise FileNotFoundError(f"OceanSim extension root not found: {oceansim_ext_root}")

    include = [s.strip() for s in args.include.split(",") if s.strip()]

    # Map include names to folder ids by listing root.
    root_items = _list_folder_items(args.root_folder_id)
    by_name = {it.name: it for it in root_items if it.is_folder}

    missing = [name for name in include if name not in by_name]
    if missing:
        raise FileNotFoundError(
            f"Missing expected folders in OceanSim assets root: {missing}. "
            f"Available: {sorted(by_name.keys())}"
        )

    for name in include:
        it = by_name[name]
        out_sub = asset_root / name
        out_sub.mkdir(parents=True, exist_ok=True)
        _sync_folder(it.id, out_sub)

    # Basic validation when MHL is requested.
    mhl = asset_root / "collected_MHL" / "mhl_scaled.usd"
    if "collected_MHL" in include and not mhl.is_file():
        raise FileNotFoundError(f"Expected MHL scene USD not found: {mhl}")

    _register_asset_path(oceansim_ext_root, asset_root)
    print("[ok] OceanSim assets configured.")
    print(f"[ok] asset_root={asset_root}")
    if mhl.is_file():
        print(f"[ok] mhl_usd={mhl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
