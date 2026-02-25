from __future__ import annotations

import argparse
import os
import ssl
import time
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


URL_DEFAULT = "https://codeload.github.com/Marine-RL/MarineGym/zip/refs/heads/main"


def _ssl_context() -> ssl.SSLContext | None:
    try:
        import certifi  # type: ignore

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None


def _download_with_resume(url: str, dst: Path, *, max_attempts: int = 20) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(dst.suffix + ".part")

    ctx = _ssl_context()

    for attempt in range(1, max_attempts + 1):
        downloaded = part.stat().st_size if part.exists() else 0
        headers = {
            "User-Agent": "OneOcean-IROS-H6/1.0",
            "Accept-Encoding": "identity",
            "Connection": "close",
        }
        if downloaded > 0:
            headers["Range"] = f"bytes={downloaded}-"

        try:
            req = Request(url, headers=headers)
            kwargs = {"timeout": 60}
            if ctx is not None:
                kwargs["context"] = ctx
            with urlopen(req, **kwargs) as resp:  # nosec - controlled URL
                status = getattr(resp, "status", None)
                mode = "ab" if downloaded > 0 and status == 206 else "wb"
                if mode == "wb":
                    downloaded = 0
                with open(part, mode) as f:
                    last_print = time.time()
                    while True:
                        chunk = resp.read(256 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if time.time() - last_print > 2.0:
                            print(f"[download] attempt={attempt} mb={downloaded/1024/1024:.2f}", flush=True)
                            last_print = time.time()

            # validate zip
            try:
                with zipfile.ZipFile(part) as zf:
                    bad = zf.testzip()
                    if bad is None:
                        os.replace(part, dst)
                        return
            except zipfile.BadZipFile:
                pass

        except HTTPError as e:
            if e.code != 416:
                print(f"[download] HTTPError attempt={attempt} code={e.code} reason={e.reason}")
        except (URLError, TimeoutError, OSError) as e:
            print(f"[download] net_error attempt={attempt} err={type(e).__name__}: {e}")

        time.sleep(min(30, 2 * attempt))

    raise RuntimeError(f"Failed to download a valid zip: {dst}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch MarineGym source zip into runs/_cache (gitignored).")
    parser.add_argument("--url", type=str, default=URL_DEFAULT)
    parser.add_argument(
        "--cache_root",
        type=Path,
        default=Path("runs/_cache/external_scenes/marinegym"),
        help="Destination cache root (inside repo; gitignored).",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cache_root: Path = args.cache_root.expanduser().resolve()
    zip_path = cache_root / "marinegym_src.zip"
    out_dir = cache_root / "MarineGym-main"

    if out_dir.exists():
        if not args.overwrite:
            print(f"[fetch] exists: {out_dir} (use --overwrite to refresh)")
            return 0
        for p in (out_dir, zip_path):
            if p.is_dir():
                for child in sorted(p.rglob("*"), reverse=True):
                    if child.is_file() or child.is_symlink():
                        child.unlink()
                    elif child.is_dir():
                        child.rmdir()
                p.rmdir()
            elif p.exists():
                p.unlink()

    cache_root.mkdir(parents=True, exist_ok=True)

    print(f"[fetch] downloading: {args.url}")
    _download_with_resume(args.url, zip_path)
    print(f"[fetch] downloaded zip: {zip_path} ({zip_path.stat().st_size/1024/1024:.1f} MB)")

    tmp_extract = cache_root / "_extract_tmp"
    if tmp_extract.exists():
        for child in sorted(tmp_extract.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        tmp_extract.rmdir()
    tmp_extract.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp_extract)

    # Zip root is "MarineGym-main/"; move it into cache_root directly.
    extracted_root = tmp_extract / "MarineGym-main"
    if not extracted_root.exists():
        # fallback: detect first folder
        first_dirs = [p for p in tmp_extract.iterdir() if p.is_dir()]
        raise RuntimeError(f"Unexpected zip structure. Found dirs: {first_dirs}")

    extracted_root.rename(out_dir)
    # cleanup temp dir
    for child in sorted(tmp_extract.rglob("*"), reverse=True):
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            child.rmdir()
    tmp_extract.rmdir()

    print(f"[fetch] extracted: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

