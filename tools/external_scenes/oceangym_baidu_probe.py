"""
Probe OceanGym Baidu share links (no login) and extract:
- file name
- file fs_id
- file size

This does NOT reliably download the file: Baidu often blocks automated downloads (errno=118).
Use it to verify the link/password and to generate actionable diagnostics for manual download.
"""

import argparse
import re
from dataclasses import dataclass

import requests


@dataclass(frozen=True)
class BaiduShareFile:
    filename: str
    fs_id: int
    size: int
    shareid: str
    share_uk: str


def _normalize_surl(share_url: str) -> str:
    surl = share_url.split("/s/")[1]
    return surl[1:] if surl.startswith("1") else surl


def probe_share(share_url: str, pwd: str, session: requests.Session) -> BaiduShareFile:
    ua = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/121.0 Safari/537.36"
    )
    headers = {"User-Agent": ua, "Referer": share_url, "X-Requested-With": "XMLHttpRequest"}

    surl = _normalize_surl(share_url)
    verify = session.post(
        f"https://pan.baidu.com/share/verify?surl={surl}", data={"pwd": pwd}, headers=headers, timeout=20
    ).json()
    if verify.get("errno") != 0:
        raise RuntimeError(f"verify failed: {verify}")

    html = session.get(share_url, headers=headers, timeout=20).text

    shareid = re.search(r'shareid:\\"(\\d+)\\"', html).group(1)
    share_uk = re.search(r'share_uk:\\"(\\d+)\\"', html).group(1)

    m_fs = re.search(r'"fs_id"\\s*:\\s*(\\d+)', html)
    m_name = re.search(r'"server_filename"\\s*:\\s*"([^"]+)"', html)
    m_size = re.search(r'"size"\\s*:\\s*(\\d+)', html)
    if not (m_fs and m_name and m_size):
        raise RuntimeError("Could not parse file_list from share HTML (layout changed?)")

    return BaiduShareFile(
        filename=m_name.group(1),
        fs_id=int(m_fs.group(1)),
        size=int(m_size.group(1)),
        shareid=shareid,
        share_uk=share_uk,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="Baidu share URL")
    ap.add_argument("--pwd", required=True, help="Extraction password")
    args = ap.parse_args()

    s = requests.Session()
    info = probe_share(args.url, args.pwd, s)

    print("[baidu] filename:", info.filename)
    print("[baidu] size_bytes:", info.size)
    print("[baidu] fs_id:", info.fs_id)
    print("[baidu] shareid:", info.shareid)
    print("[baidu] share_uk:", info.share_uk)
    print("[baidu] cookies_keys:", list(s.cookies.get_dict().keys()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

