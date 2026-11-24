"""OpenVLA-OFT behind the shared interface: the release's LIBERO evaluation, query by query.

The release is github.com/moojink/openvla-oft @ e4287e94541f459edc4feabc4e181f537cd569a8 (row.json), its LIBERO
evaluation `experiments/robot/libero/run_libero_eval.py` ("R" below), `experiments/robot/libero/libero_utils.py`
("LU"), `experiments/robot/openvla_utils.py` ("OU"), `experiments/robot/robot_utils.py` ("RU"),
`prismatic/vla/constants.py` ("C"). The checkpoint is the one policy for the four suites,
moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10.

What the release does to one observation, and where this adapter does it:

| step | release | here |
|---|---|---|
| rotate both images 180 degrees, `img[::-1, ::-1]` | LU 33-44 (`get_libero_image`, `get_libero_wrist_image`) | `_rotate` (LU imports the simulator, so the one line is written here) |
| JPEG round trip and Lanczos resize to 224, `resize_image_for_policy` | OU 520-543, called at R 250-251 | the release's function |
| centre crop to 90 % area, PIL, processor, image order [main, wrist] | OU 682-770 inside `get_vla_action` | the release's function |
| state = eef position, quaternion -> axis-angle, gripper joints (8) | R 257-259, LU 63-87 | `_state` (LU imports the simulator; the conversion is robosuite's, written out, with the release's clip of w) |
| proprio normalisation (BOUNDS_Q99, the suite's key) | OU 774-778 | inside `get_vla_action` |
| prompt "In: What action should the robot take to {instruction.lower()}?\\nOut:" | OU 757 | inside `get_vla_action` |
| un-normalise dims 0-5 with the suite's q01/q99, gripper raw | the model's `predict_action` | inside `get_vla_action` |
| gripper [0, 1] -> [-1, 1], then sign (binarize) | RU 149-177 via R 268 | the release's `normalize_gripper_action` |
| gripper sign flip (model 1 = open -> env -1 = open) | RU 180-199 via R 272-273 | the release's `invert_gripper_action` |
| 8 actions per query, all executed (queue refilled when empty) | R 100, 302-344; C 27 | `execute = 8`, the chunk's length |
| un-normalisation key: the suite's `<suite>_no_noops` | R 179-192 `check_unnorm_key` | `_unnorm_key`, from `Episode.suite` |

Nothing of the release is copied but the two lines named above. The shared side (settling steps, step limit,
environment seed, start states) is not this adapter's: the release seeds `env.seed(0)` per task (LU 24, R 384);
the evaluation seeds 7.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from evaluation.interface import Actions, Adapter, Episode, Observation

#: GenerateConfig's values for the released LIBERO evaluation (R 85-107), the ones `get_vla*` read.
RELEASE_CONFIG = {
    "model_family": "openvla",
    "use_l1_regression": True,
    "use_diffusion": False,
    "num_diffusion_steps_train": 50,
    "num_diffusion_steps_inference": 50,
    "use_film": False,
    "num_images_in_input": 2,
    "use_proprio": True,
    "center_crop": True,
    "num_open_loop_steps": 8,
    "lora_rank": 32,
    "load_in_8bit": False,
    "load_in_4bit": False,
}
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
POLICY_SIZE = 224  # RU 32-35 `get_image_resize_size` for model_family openvla
PROPRIO_DIM = 8  # R 157


def _rotate(image: np.ndarray) -> np.ndarray:
    """LU 33-44: the release rotates both cameras by 180 degrees before anything else."""
    return np.ascontiguousarray(image[::-1, ::-1])


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """LU 63-87 (robosuite's transform_utils.quat2axisangle): (x, y, z, w) -> axis * angle. Like the release,
    w is clipped to [-1, 1]; unlike it, on a copy, so the shared observation is not changed."""
    q = np.array(quat, dtype=np.float64, copy=True)
    q[3] = min(max(q[3], -1.0), 1.0)
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def _state(observation: Observation) -> np.ndarray:
    """R 257-259: concatenate(eef_pos, quat2axisangle(eef_quat), gripper_qpos), 8 numbers."""
    s = observation.state
    out = np.concatenate([np.asarray(s["eef_pos"], dtype=np.float64), quat2axisangle(s["eef_quat"]),
                          np.asarray(s["gripper_qpos"], dtype=np.float64)])
    assert out.shape == (PROPRIO_DIM,), out.shape
    return out


def _unnorm_key(suite: str, norm_stats: Dict[str, Any]) -> str:
    """R 179-192: the suite's name, or `<suite>_no_noops` when only that is in the checkpoint's statistics."""
    if suite not in SUITES:
        raise ValueError("OpenVLA-OFT needs the task's LIBERO suite (one of %s), got %r" % (SUITES, suite))
    key = suite if suite in norm_stats else suite + "_no_noops"
    if key not in norm_stats:
        raise ValueError("un-normalisation key %r is not in the checkpoint's statistics %s" % (key, sorted(norm_stats)))
    return key


class OpenVlaOft(Adapter):
    """The release's functions are passed in (`release`), so the glue can be read and tested without them."""

    def __init__(self, row: Dict[str, Any], release: Dict[str, Callable], vla: Any, processor: Any,
                 action_head: Any, proprio_projector: Any, loaded: Dict[str, Any]):
        self.row, self.release = row, release
        self.vla, self.processor, self.action_head, self.proprio_projector = vla, processor, action_head, proprio_projector
        self.loaded = loaded
        self.cfg = SimpleNamespace(pretrained_checkpoint=row["checkpoint"]["hub_id"], unnorm_key="",
                                   task_suite_name="", **RELEASE_CONFIG)
        self.episode: Optional[Episode] = None

    # -- the interface --------------------------------------------------------------------------------------
    def report(self) -> Dict[str, Any]:
        c = self.cfg
        return {
            "model": {"name": self.row["name"],
                      "checkpoint": {"address": self.row["checkpoint"]["address"],
                                     "revision": self.loaded["checkpoint_revision"],
                                     "sha256": self.loaded["checkpoint_sha256"],
                                     "licence": self.row["checkpoint"]["licence"]},
                      "code": {"address": self.row["code"]["address"], "revision": self.row["code"]["revision"]}},
            "inference": {
                "action_head": "L1 regression, parallel decoding (use_l1_regression %s, use_diffusion %s, use_film %s)"
                               % (c.use_l1_regression, c.use_diffusion, c.use_film),
                "chunk": self.loaded["NUM_ACTIONS_CHUNK"], "executed_per_query": c.num_open_loop_steps,
                "action_dim": self.loaded["ACTION_DIM"], "platform_constants": self.loaded["platform"],
                "images": {"cameras": ["main", "wrist"], "order": "main then wrist (OU 746-748)",
                           "steps": ["rotate 180 degrees", "JPEG round trip and Lanczos resize to %d (tf)" % POLICY_SIZE,
                                     "centre crop 90 %% area: %s" % c.center_crop, "processor (DINOv2 and SigLIP stats)"]},
                "state": "read: eef position, axis-angle from the quaternion, gripper joints (8); BOUNDS_Q99 with the "
                         "suite's statistics; one token",
                "prompt": "In: What action should the robot take to {instruction.lower()}?\\nOut:",
                "unnormalisation": "BOUNDS_Q99 with the key <suite>_no_noops, chosen per episode from its suite; "
                                   "gripper not un-normalised",
                "gripper": "2x - 1, then sign (binarized), then x -1: env -1 open, +1 close",
                "precision": self.loaded["dtype"], "quantisation": "none (8-bit %s, 4-bit %s)" % (c.load_in_8bit, c.load_in_4bit),
                "norm_stats_keys": self.loaded["norm_stats_keys"], "lora_rank_for_film_only": c.lora_rank,
                "release_render_size": 256, "release_environment_seed": "env.seed(0) once per task (not used here)"},
            "method": {"name": "none"},
            "noise_seed_controls": "nothing: the L1-regression head is deterministic (no sampling); the seed only "
                                   "seeds torch, numpy and random at the start of the episode",
            "environment": {"name": self.row["environment"], "python": sys.version.split()[0],
                            "packages": self.loaded["packages"]},
            "requires": {"camera_height": 256, "camera_width": 256},
        }

    def begin_episode(self, episode: Episode) -> None:
        suite = getattr(episode, "suite", None)
        self.cfg.task_suite_name = suite
        self.cfg.unnorm_key = _unnorm_key(suite, self.vla.norm_stats)
        self.release["set_seed_everywhere"](episode.noise_seed)
        self.episode = episode

    def act(self, observation: Observation) -> Actions:
        if self.episode is None:
            raise RuntimeError("act() before begin_episode()")
        obs = {
            "full_image": self.release["resize_image_for_policy"](_rotate(observation.images["main"]), POLICY_SIZE),
            "wrist_image": self.release["resize_image_for_policy"](_rotate(observation.images["wrist"]), POLICY_SIZE),
            "state": _state(observation),
        }
        chunk = self.release["get_vla_action"](self.cfg, self.vla, self.processor, obs, observation.instruction,
                                               action_head=self.action_head, proprio_projector=self.proprio_projector,
                                               noisy_action_projector=None, use_film=self.cfg.use_film)
        actions = np.stack([self._to_env(np.asarray(a, dtype=np.float64)) for a in chunk])
        return Actions(actions=actions, execute=min(self.cfg.num_open_loop_steps, len(actions)))

    def _to_env(self, action: np.ndarray) -> np.ndarray:
        """R 265-275 `process_action` for model_family openvla."""
        action = self.release["normalize_gripper_action"](action, binarize=True)
        return self.release["invert_gripper_action"](action)


# -------------------------------------------------------------------------------------------------- loading ---
def _sha256_tree(folder: str) -> Dict[str, str]:
    out = {}
    for root, _, files in os.walk(folder):
        for name in sorted(files):
            path = os.path.join(root, name)
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(1 << 24), b""):
                    h.update(block)
            out[os.path.relpath(path, folder)] = h.hexdigest()
    return dict(sorted(out.items()))


def load(row: Dict[str, Any]) -> Adapter:
    """Load as the release does (`get_vla`, `get_processor`, `get_proprio_projector`, `get_action_head`), from
    the hub id, after checking that the id resolves to the pinned revision: the release's loaders pass no
    revision (OU 282, 362, 425, 507), so a moved `main` would load other weights."""
    import importlib.metadata as md

    from huggingface_hub import HfApi, snapshot_download

    hub_id, pinned = row["checkpoint"]["hub_id"], row["checkpoint"]["revision"]
    resolved = HfApi().model_info(hub_id).sha
    if resolved != pinned:
        raise RuntimeError("%s resolves to %s, not the pinned %s: the release loads `main`" % (hub_id, resolved, pinned))
    folder = snapshot_download(hub_id, revision=pinned)

    # C 49-60 picks the platform from sys.argv at import; check what it picked.
    from prismatic.vla import constants as C
    from experiments.robot import openvla_utils as OU
    from experiments.robot import robot_utils as RU

    if (C.NUM_ACTIONS_CHUNK, C.ACTION_DIM, C.PROPRIO_DIM) != (8, 7, PROPRIO_DIM):
        raise RuntimeError("prismatic chose another platform (chunk %d, action %d, proprio %d) from sys.argv %r"
                           % (C.NUM_ACTIONS_CHUNK, C.ACTION_DIM, C.PROPRIO_DIM, sys.argv))
    cfg = SimpleNamespace(pretrained_checkpoint=hub_id, unnorm_key="", task_suite_name="", **RELEASE_CONFIG)
    vla = OU.get_vla(cfg)
    processor = OU.get_processor(cfg)
    proprio_projector = OU.get_proprio_projector(cfg, vla.llm_dim, proprio_dim=PROPRIO_DIM)
    action_head = OU.get_action_head(cfg, vla.llm_dim)
    packages = {}
    for name in ("torch", "transformers", "tensorflow", "timm", "tokenizers", "numpy", "openvla-oft"):
        try:
            packages[name] = md.version(name)
        except md.PackageNotFoundError:
            packages[name] = "not installed"
    loaded = {
        "checkpoint_revision": resolved,
        "checkpoint_sha256": {"files": _sha256_tree(folder), "how": "sha256 of every file of the snapshot loaded"},
        "NUM_ACTIONS_CHUNK": int(C.NUM_ACTIONS_CHUNK), "ACTION_DIM": int(C.ACTION_DIM),
        "platform": str(getattr(C, "ROBOT_PLATFORM", "unknown")),
        "dtype": str(next(vla.parameters()).dtype), "norm_stats_keys": sorted(vla.norm_stats),
        "packages": packages,
    }
    release = {"resize_image_for_policy": OU.resize_image_for_policy, "get_vla_action": OU.get_vla_action,
               "normalize_gripper_action": RU.normalize_gripper_action,
               "invert_gripper_action": RU.invert_gripper_action, "set_seed_everywhere": RU.set_seed_everywhere}
    return OpenVlaOft(row, release, vla, processor, action_head, proprio_projector, loaded)
