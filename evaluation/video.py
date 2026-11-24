"""The main-camera video of one episode: one frame per simulator step, upright, H.264.

The encoder is `imageio` with `imageio-ffmpeg`, which every simulator environment declares. Imported when the
first video is written, so the rest of the evaluation runs without it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import numpy as np

FPS = 20  # the simulator's control frequency
QUALITY = 6  # imageio's scale 0-10


class VideoWriter:
    def __init__(self, path: Path) -> None:
        self._path, self._frames = Path(path), []  # type: Path, List[np.ndarray]

    def add(self, image: Any) -> None:
        """`image` is the main camera's array as the simulator returns it; it is stored upright."""
        self._frames.append(np.ascontiguousarray(np.asarray(image)[::-1, ::-1]))

    def save(self) -> None:
        import imageio

        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_name(self._path.stem + ".tmp" + self._path.suffix)
        imageio.mimwrite(str(temporary), self._frames, fps=FPS, quality=QUALITY, macro_block_size=1)
        temporary.replace(self._path)
