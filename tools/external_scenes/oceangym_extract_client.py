import argparse
import subprocess
import sys
import zipfile
from pathlib import Path


def _client_root_prefix(zip_names: list[str]) -> str | None:
    for name in zip_names:
        p = name.replace("\\", "/")
        parts = [x for x in p.split("/") if x]
        if len(parts) >= 2 and parts[1] == "client":
            return parts[0]
    return None


def extract_client(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        prefix = _client_root_prefix(names)
        if not prefix:
            raise RuntimeError("Could not find <root>/client/ in the zip. Inspect the zip contents first.")

        extracted = 0
        for member in names:
            p = member.replace("\\", "/")
            parts = [x for x in p.split("/") if x]
            if len(parts) >= 2 and parts[0] == prefix and parts[1] == "client":
                zf.extract(member, out_dir)
                extracted += 1

    client_dir = out_dir / prefix / "client"
    if not client_dir.exists():
        raise RuntimeError(f"Client dir not found after extraction: {client_dir}")
    if not (client_dir / "setup.py").exists() and not (client_dir / "pyproject.toml").exists():
        raise RuntimeError(f"Extracted client dir does not look installable: {client_dir}")

    print(f"[oceangym] extracted {extracted} members into {out_dir}")
    return client_dir


def pip_install(client_dir: Path, python_exe: str) -> None:
    cmd = [python_exe, "-m", "pip", "install", "."]
    print("[oceangym] pip install:", " ".join(cmd), "(cwd:", str(client_dir), ")")
    subprocess.check_call(cmd, cwd=str(client_dir))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="Path to OceanGym_small.zip or OceanGym_large.zip")
    ap.add_argument(
        "--out",
        default="runs/_cache/external_scenes/oceangym/extracted",
        help="Extraction directory (local cache; not committed)",
    )
    ap.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for pip install (recommend a dedicated venv)",
    )
    ap.add_argument("--install", action="store_true", help="Run pip install after extraction")
    args = ap.parse_args()

    zip_path = Path(args.zip).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    client_dir = extract_client(zip_path, out_dir)
    print("[oceangym] client_dir:", client_dir)

    if args.install:
        pip_install(client_dir, args.python)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

