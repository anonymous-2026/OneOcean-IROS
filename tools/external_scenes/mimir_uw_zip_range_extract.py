#!/usr/bin/env python3
"""
Selectively extract files from a remote .zip using HTTP Range requests (no full download).

This complements `mimir_uw_zip_range_list.py`:
- list remote zip entries (central directory)
- extract a small subset of files for inspection/prototyping

Supported compression:
- method 0 (stored)
- method 8 (deflate)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path


EOCD_SIG = b"PK\x05\x06"
CD_SIG = b"PK\x01\x02"
LFH_SIG = b"PK\x03\x04"
ZIP64_LOC_SIG = b"PK\x06\x07"
ZIP64_EOCD_SIG = b"PK\x06\x06"


@dataclass(frozen=True)
class CentralDirEntry:
    filename: str
    compress_size: int
    file_size: int
    method: int
    flags: int
    local_header_offset: int


def _http_head(url: str) -> dict[str, str]:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return {k.lower(): v for k, v in resp.headers.items()}


def _get_size(url: str) -> int:
    headers = _http_head(url)
    if "content-length" not in headers:
        raise RuntimeError("missing content-length on HEAD")
    return int(headers["content-length"])


def _http_range(url: str, start: int, end_inclusive: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end_inclusive}"})
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                status = getattr(resp, "status", None)
                if status not in (200, 206):
                    raise RuntimeError(f"unexpected HTTP status: {status}")
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 502, 503, 504):
                retry_after = e.headers.get("Retry-After")
                sleep_s = int(retry_after) if retry_after and retry_after.isdigit() else min(60, 2**attempt)
                print(
                    f"[mimir-uw-zip-extract] HTTP {e.code} for range {start}-{end_inclusive}; retry in {sleep_s}s",
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
        data.extend(_http_range(url, pos, chunk_end))
        downloaded = len(data)
        if total >= 4_000_000:
            print(
                f"[mimir-uw-zip-extract] downloaded {downloaded}/{total} bytes ({downloaded/total:.1%})",
                file=sys.stderr,
                flush=True,
            )
        pos = chunk_end + 1
    return bytes(data)


def _find_eocd(tail: bytes) -> int:
    idx = tail.rfind(EOCD_SIG)
    if idx < 0:
        raise RuntimeError("EOCD not found in tail; increase tail-bytes")
    return idx


def _parse_eocd(tail: bytes, eocd_offset_in_tail: int) -> tuple[int, int]:
    base = eocd_offset_in_tail
    if base + 22 > len(tail):
        raise RuntimeError("EOCD truncated")
    cd_size = struct.unpack_from("<I", tail, base + 12)[0]
    cd_offset = struct.unpack_from("<I", tail, base + 16)[0]
    return cd_size, cd_offset


def _parse_zip64_eocd(url: str, tail: bytes) -> tuple[int, int]:
    loc_idx = tail.rfind(ZIP64_LOC_SIG)
    if loc_idx < 0:
        raise RuntimeError("zip64 locator not found in tail; increase tail-bytes")
    zip64_eocd_offset = struct.unpack_from("<Q", tail, loc_idx + 8)[0]
    hdr = _http_range(url, zip64_eocd_offset, zip64_eocd_offset + 56 - 1)
    if hdr[:4] != ZIP64_EOCD_SIG:
        raise RuntimeError("zip64 EOCD signature not found at locator offset")
    cd_size = struct.unpack_from("<Q", hdr, 40)[0]
    cd_offset = struct.unpack_from("<Q", hdr, 48)[0]
    return int(cd_size), int(cd_offset)


def _parse_central_directory(cd: bytes) -> list[CentralDirEntry]:
    entries: list[CentralDirEntry] = []
    i = 0
    n = len(cd)
    while i + 46 <= n:
        if cd[i : i + 4] != CD_SIG:
            break
        flags = struct.unpack_from("<H", cd, i + 8)[0]
        method = struct.unpack_from("<H", cd, i + 10)[0]
        compress_size = struct.unpack_from("<I", cd, i + 20)[0]
        file_size = struct.unpack_from("<I", cd, i + 24)[0]
        filename_len = struct.unpack_from("<H", cd, i + 28)[0]
        extra_len = struct.unpack_from("<H", cd, i + 30)[0]
        comment_len = struct.unpack_from("<H", cd, i + 32)[0]
        local_header_offset = struct.unpack_from("<I", cd, i + 42)[0]

        name_start = i + 46
        name_end = name_start + filename_len
        if name_end > n:
            break
        filename = cd[name_start:name_end].decode("utf-8", errors="replace")

        entries.append(
            CentralDirEntry(
                filename=filename,
                compress_size=compress_size,
                file_size=file_size,
                method=method,
                flags=flags,
                local_header_offset=local_header_offset,
            )
        )
        i = name_end + extra_len + comment_len
    return entries


def _local_file_data_offset(url: str, local_header_offset: int) -> tuple[int, int, int]:
    """
    Return (data_offset, method, flags) by reading the local file header.
    Local file header fixed size is 30 bytes.
    """
    hdr = _http_range(url, local_header_offset, local_header_offset + 30 - 1)
    if hdr[:4] != LFH_SIG:
        raise RuntimeError(f"bad local header signature at offset {local_header_offset}")
    flags = struct.unpack_from("<H", hdr, 6)[0]
    method = struct.unpack_from("<H", hdr, 8)[0]
    filename_len = struct.unpack_from("<H", hdr, 26)[0]
    extra_len = struct.unpack_from("<H", hdr, 28)[0]
    data_offset = local_header_offset + 30 + filename_len + extra_len
    return data_offset, method, flags


def _decompress(method: int, compressed: bytes, expected_size: int) -> bytes:
    if method == 0:
        out = compressed
    elif method == 8:
        out = zlib.decompress(compressed, wbits=-15)
    else:
        raise RuntimeError(f"unsupported zip compression method: {method}")
    if expected_size >= 0 and len(out) != expected_size:
        # Do not hard-fail: some zips use data descriptors; size mismatch can happen.
        print(
            f"[mimir-uw-zip-extract] warning: size mismatch after decompress: got={len(out)} expected={expected_size}",
            file=sys.stderr,
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="Remote zip URL (Zenodo content endpoint supports Range)")
    ap.add_argument("--out-dir", required=True, help="Output directory for extracted files + manifest")
    ap.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="Regex pattern to select entries (may be provided multiple times).",
    )
    ap.add_argument("--max-files", type=int, default=50, help="Safety cap for number of extracted files")
    ap.add_argument("--tail-bytes", type=int, default=262144, help="Tail bytes for EOCD search")
    args = ap.parse_args()

    if not args.pattern:
        print("error: at least one --pattern is required", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    url = args.url
    print("[mimir-uw-zip-extract] HEAD -> content-length ...", file=sys.stderr, flush=True)
    size = _get_size(url)

    tail_n = min(args.tail_bytes, size)
    tail_start = size - tail_n
    print(
        f"[mimir-uw-zip-extract] GET tail bytes={tail_start}-{size-1} ({tail_n} bytes) ...",
        file=sys.stderr,
        flush=True,
    )
    tail = _http_range(url, tail_start, size - 1)
    eocd_idx = _find_eocd(tail)
    cd_size, cd_offset = _parse_eocd(tail, eocd_idx)
    if cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF:
        cd_size, cd_offset = _parse_zip64_eocd(url, tail)
    print(
        f"[mimir-uw-zip-extract] GET central_directory bytes={cd_offset}-{cd_offset + cd_size - 1} ({cd_size} bytes) ...",
        file=sys.stderr,
        flush=True,
    )
    cd = _http_range_chunked(url, cd_offset, cd_offset + cd_size - 1)
    entries = _parse_central_directory(cd)

    regexes = [re.compile(p) for p in args.pattern]
    selected = [
        e for e in entries if (not e.filename.endswith("/")) and any(r.search(e.filename) for r in regexes)
    ]
    selected = selected[: args.max_files]

    manifest = {
        "url": url,
        "zip_size_bytes": size,
        "central_directory": {"size_bytes": cd_size, "offset_bytes": cd_offset},
        "patterns": args.pattern,
        "selected_count": len(selected),
        "extracted": [],
    }

    for e in selected:
        print(f"[mimir-uw-zip-extract] extracting: {e.filename}", file=sys.stderr, flush=True)
        data_offset, lfh_method, _flags = _local_file_data_offset(url, e.local_header_offset)
        if lfh_method != e.method:
            print(
                f"[mimir-uw-zip-extract] warning: method mismatch cd={e.method} lfh={lfh_method} for {e.filename}",
                file=sys.stderr,
            )
        compressed = _http_range_chunked(url, data_offset, data_offset + e.compress_size - 1)
        out = _decompress(e.method, compressed, e.file_size)

        out_path = out_dir / e.filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(out)

        manifest["extracted"].append(
            {
                "filename": e.filename,
                "output_path": str(out_path),
                "file_size": e.file_size,
                "compress_size": e.compress_size,
                "method": e.method,
                "local_header_offset": e.local_header_offset,
                "data_offset": data_offset,
            }
        )

    (out_dir / "extract_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote: {out_dir / 'extract_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
