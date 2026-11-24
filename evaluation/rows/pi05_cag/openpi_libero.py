"""What the openpi-family method rows (pi05_cag, pi05_cag_va, pi05_rss) share: the request openpi's LIBERO client
builds from a simulator observation, the noise of a query, and the digests a report needs.

The request is the one LIBERO-CF's client (`eval/main_cf.py` L201-232, at 8280a500) and RSS's client
(`examples/libero_shortcut/main_origin.py` L119-145, at f51a0a3f) both build, line for line the same as openpi's
`examples/libero/main.py`:

* each camera image rotated by 180 degrees (`[::-1, ::-1]`), then `image_tools.resize_with_pad(img, 224, 224)` and
  `image_tools.convert_to_uint8` (openpi-client's own functions, imported from the row's environment);
* `observation/state` = end-effector position (3), the end-effector quaternion as axis-angle (3), the two gripper
  joints (2);
* `prompt` = the instruction as given.

The noise of a query follows the shared rule (`interface.noise_generator`: a function of the noise seed, the
episode's task and start state, and the query index), drawn as `standard_normal((10, 32))`, float32, as the
openpi rows draw it, so that rows on one model get the same noise in the same episode.

NumPy only at import time; openpi-client is imported where the request is built.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np

RESIZE = 224  # resize_size of both clients (main_cf.py L37, main_origin.py L32)
ACTION_HORIZON = 10  # pi05_libero / pi05_cfg_libero_caption_infer: action_horizon=10
ACTION_DIM = 32  # openpi's padded action dimension (Pi0Config.action_dim default)


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """(x, y, z, w) -> axis-angle (3): robosuite's `transform_utils.quat2axisangle`, which both clients copy."""
    quat = np.array(quat, dtype=np.float64)
    quat[3] = min(1.0, max(-1.0, quat[3]))
    den = math.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def rotate_180(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(image[::-1, ::-1])


def libero_request(observation, resize=None) -> Dict[str, object]:
    """The dict openpi's LIBERO client sends, from a shared `Observation`.

    `resize(image, size) -> uint8 image` defaults to openpi-client's `resize_with_pad` + `convert_to_uint8`.
    """
    if resize is None:
        from openpi_client import image_tools

        def resize(image, size):
            return image_tools.convert_to_uint8(image_tools.resize_with_pad(image, size, size))

    state = observation.state
    return {
        "observation/image": resize(rotate_180(np.asarray(observation.images["main"])), RESIZE),
        "observation/wrist_image": resize(rotate_180(np.asarray(observation.images["wrist"])), RESIZE),
        "observation/state": np.concatenate(
            (np.asarray(state["eef_pos"], np.float64), quat2axisangle(state["eef_quat"]),
             np.asarray(state["gripper_qpos"], np.float64))
        ),
        "prompt": str(observation.instruction),
    }


def query_noise(episode, query: int, horizon: int = ACTION_HORIZON, dim: int = ACTION_DIM) -> np.ndarray:
    from evaluation import interface

    return interface.noise_generator(episode, query).standard_normal((horizon, dim)).astype(np.float32)


NOISE_SENTENCE = (
    "The initial noise of every query: interface.noise_generator(episode, query).standard_normal((10, 32)), float32 "
    "(the shared rule: noise seed, task index, start state, query index); the same sample enters both samplings "
    "(conditioned and unconditioned) of the query."
)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_blob(path: Path) -> str:
    """The git blob id of a file, as GitHub's tree API lists it."""
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def tree_sha256(root: Path, skip: Optional[tuple] = None) -> Dict[str, str]:
    """The checkpoint digest in the form of the config record: sha256 over the sorted lines
    '<file sha256>  ./<relative path>' of every file under `root` (each line ending in a newline)."""
    root = Path(root)
    lines = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if skip and rel.startswith(skip):
            continue
        lines.append("%s  ./%s" % (file_sha256(path), rel))
    digest = hashlib.sha256("".join(line + "\n" for line in sorted(lines)).encode()).hexdigest()
    return {"tree": digest, "files": str(len(lines)),
            "computed_as": "sha256 over the sorted lines '<file sha256>  ./<relative path>' of every file, each ending in a newline"}


def versions(*names: str) -> Dict[str, str]:
    out = {}
    for name in names:
        try:
            from importlib import metadata

            out[name] = metadata.version(name)
        except Exception:  # not installed under that distribution name
            out[name] = "not found"
    return out


def local_checkpoint(address: str) -> Path:
    """A gs:// checkpoint through openpi's own cache (`openpi.shared.download.maybe_download`)."""
    from openpi.shared import download

    return Path(download.maybe_download(address))


def hub_snapshot(repo_id: str, revision: str, allow=None) -> Path:
    """A Hugging Face checkpoint at a pinned revision, through the hub's own cache.

    The cache is read first (`local_files_only`): a machine that holds the snapshot, or runs with HF_HUB_OFFLINE,
    never asks the hub (huggingface_hub lists the remote tree for `allow_patterns` otherwise, which offline mode
    refuses). Only when the snapshot is missing and the machine is online is it downloaded.
    """
    import os

    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(repo_id=repo_id, revision=revision, allow_patterns=allow, local_files_only=True))
    except Exception:
        if os.environ.get("HF_HUB_OFFLINE", "") not in ("", "0"):
            raise
    return Path(snapshot_download(repo_id=repo_id, revision=revision, allow_patterns=allow))


def python_version() -> str:
    import platform as _p

    return _p.python_version()
