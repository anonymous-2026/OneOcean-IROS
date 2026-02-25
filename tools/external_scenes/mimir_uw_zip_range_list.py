#!/usr/bin/env python3
"""
List the contents of a remote .zip using HTTP Range requests (no full download).

Use case (H1 MIMIR-UW track):
- Determine whether Zenodo environment zips contain packaged Unreal/AirSim projects,
  offline dataset folders, or something else, without downloading multi-GB files.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


EOCD_SIG = b"PK\x05\x06"  # end of central directory
CD_SIG = b"PK\x01\x02"  # central directory file header


@dataclass(frozen=True)
class ZipEntry:
    filename: str
    compress_size: int
    file_size: int


def _http_range(url: str, start: int, end_inclusive: int) -> bytes:
    if start < 0 or end_inclusive < start:
        raise ValueError(f"invalid range: {start}-{end_inclusive}")
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end_inclusive}"})
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                status = getattr(resp, "status", None)
                if status not in (200, 206):
                    raise RuntimeError(f"unexpected HTTP status for range request: {status}")
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 502, 503, 504):
                retry_after = e.headers.get("Retry-After")
                sleep_s = int(retry_after) if retry_after and retry_after.isdigit() else min(60, 2 ** attempt)
                print(
                    f"[mimir-uw-zip-probe] HTTP {e.code} for range {start}-{end_inclusive}; retry in {sleep_s}s (attempt {attempt+1}/8)",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(sleep_s)
                continue
            raise
    raise RuntimeError(f"range request failed after retries: {start}-{end_inclusive}")


def _http_range_chunked(url: str, start: int, end_inclusive: int, *, chunk_bytes: int = 1_048_576) -> bytes:
    data = bytearray()
    pos = start
    total = end_inclusive - start + 1
    while pos <= end_inclusive:
        chunk_end = min(end_inclusive, pos + chunk_bytes - 1)
        chunk = _http_range(url, pos, chunk_end)
        data.extend(chunk)
        downloaded = len(data)
        if total >= 4_000_000:
            print(
                f"[mimir-uw-zip-probe] downloaded {downloaded}/{total} bytes ({downloaded/total:.1%})",
                file=sys.stderr,
                flush=True,
            )
        pos = chunk_end + 1
    return bytes(data)


def _http_head(url: str) -> dict[str, str]:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=120) as resp:
        headers = {k.lower(): v for k, v in resp.headers.items()}
    return headers


def _get_size(url: str) -> int:
    headers = _http_head(url)
    if "content-length" in headers:
        return int(headers["content-length"])
    raise RuntimeError("could not determine file size via HEAD (missing content-length)")


def _find_eocd(tail: bytes) -> int:
    # Search from the end to find the EOCD signature.
    # EOCD record is at least 22 bytes; search for signature occurrences and pick the last valid one.
    idx = tail.rfind(EOCD_SIG)
    if idx < 0:
        raise RuntimeError("EOCD signature not found in tail; increase tail size")
    return idx


def _parse_eocd(tail: bytes, eocd_offset_in_tail: int) -> tuple[int, int]:
    """
    Return (central_directory_size, central_directory_offset).
    EOCD layout (little-endian):
      4  signature
      2  disk number
      2  disk with CD start
      2  CD records on this disk
      2  total CD records
      4  CD size
      4  CD offset
      2  comment length
      n  comment
    """
    base = eocd_offset_in_tail
    if base + 22 > len(tail):
        raise RuntimeError("EOCD truncated")
    cd_size = struct.unpack_from("<I", tail, base + 12)[0]
    cd_offset = struct.unpack_from("<I", tail, base + 16)[0]
    return cd_size, cd_offset


def _parse_central_directory(cd: bytes) -> list[ZipEntry]:
    entries: list[ZipEntry] = []
    i = 0
    n = len(cd)
    while i + 46 <= n:
        if cd[i : i + 4] != CD_SIG:
            # Stop on first non-header; central directory may have ended.
            break

        # Central directory header fixed size is 46 bytes.
        # Offsets (little-endian):
        # 28: filename length (2)
        # 30: extra length (2)
        # 32: comment length (2)
        # 20: compressed size (4)
        # 24: uncompressed size (4)
        filename_len = struct.unpack_from("<H", cd, i + 28)[0]
        extra_len = struct.unpack_from("<H", cd, i + 30)[0]
        comment_len = struct.unpack_from("<H", cd, i + 32)[0]
        compress_size = struct.unpack_from("<I", cd, i + 20)[0]
        file_size = struct.unpack_from("<I", cd, i + 24)[0]

        name_start = i + 46
        name_end = name_start + filename_len
        if name_end > n:
            break
        filename = cd[name_start:name_end].decode("utf-8", errors="replace")

        entries.append(ZipEntry(filename=filename, compress_size=compress_size, file_size=file_size))

        i = name_end + extra_len + comment_len
    return entries


def _top_levels(paths: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for p in paths:
        p = p.lstrip("/").replace("\\", "/")
        top = p.split("/", 1)[0] if p else ""
        counts[top] = counts.get(top, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="Direct download URL for a .zip (supports HTTP Range)")
    ap.add_argument("--out-dir", required=True, help="Output directory for listings/summary JSON")
    ap.add_argument("--tail-bytes", type=int, default=262144, help="Bytes to fetch from end to locate EOCD")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    url = args.url
    print("[mimir-uw-zip-probe] HEAD -> content-length ...", file=sys.stderr, flush=True)
    size = _get_size(url)

    tail_n = min(args.tail_bytes, size)
    tail_start = size - tail_n
    print(
        f"[mimir-uw-zip-probe] GET tail bytes={tail_start}-{size-1} ({tail_n} bytes) ...",
        file=sys.stderr,
        flush=True,
    )
    tail = _http_range(url, tail_start, size - 1)
    eocd_idx = _find_eocd(tail)
    cd_size, cd_offset = _parse_eocd(tail, eocd_idx)

    if cd_size > 0:
        print(
            f"[mimir-uw-zip-probe] GET central_directory bytes={cd_offset}-{cd_offset + cd_size - 1} ({cd_size} bytes) ...",
            file=sys.stderr,
            flush=True,
        )
        cd = _http_range_chunked(url, cd_offset, cd_offset + cd_size - 1)
    else:
        cd = b""
    entries = _parse_central_directory(cd)

    # Write human-readable listing.
    listing_path = out_dir / "zip_entries.txt"
    with listing_path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(f"{e.file_size}\t{e.compress_size}\t{e.filename}\n")

    # Write a compact summary.
    summary = {
        "url": url,
        "zip_size_bytes": size,
        "central_directory": {"size_bytes": cd_size, "offset_bytes": cd_offset},
        "num_entries": len(entries),
        "top_levels": _top_levels(e.filename for e in entries),
    }
    summary_path = out_dir / "zip_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote: {listing_path}")
    print(f"Wrote: {summary_path}")

    # Print a quick hint to stdout for decision-making.
    top_levels = list(summary["top_levels"].keys())[:20]
    print("Top-level entries (by count):", ", ".join(top_levels))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
