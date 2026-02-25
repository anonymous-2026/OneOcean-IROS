from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable, Optional

import imageio.v2 as imageio


def decode_png(png_bytes: bytes):
    return imageio.imread(io.BytesIO(png_bytes))


class Mp4Writer:
    def __init__(self, path: str | Path, fps: int = 20):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio.get_writer(self.path, fps=fps)

    def append_png_bytes(self, png_bytes: bytes) -> None:
        frame = decode_png(png_bytes)
        self._writer.append_data(frame)

    def close(self) -> None:
        self._writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

