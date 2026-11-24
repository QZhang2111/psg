"""TurboVLA behind the shared interface: the release's LIBERO evaluation, query by query.

The release is github.com/H-EmbodVis/TurboVLA @ 650a288f829f549bea18e2173d3f51e77d3fbb74 and the checkpoint
H-EmbodVis/TurboVLA @ 9cdb576ed9c56039b9c1149142ed654e876bad76, file checkpoints/libero/turbovla_libero.pth (one
model for the four suites). "P" is turbovla/evaluation/policy.py, "SP" turbovla/evaluation/suite_policy.py, "R"
third_party/vla_adapter/vla_adapter/rollout.py.

| step | release | here |
|---|---|---|
| rotate both images 180 degrees | P 555-556 inside `predict_env_action_chunk_from_obs` (R 254-255 in the loop) | the release's method |
| 256 x 256 only (training images are 256, the processor's resize and crop are off) | P 22, 131-139 | the row asks for 256 |
| /255, DINOv3 mean and std, CHW; [main, wrist] | P 107-110, 141-163; SP 77-86 | the release's method |
| state = eef position, axis-angle, gripper joints (8); (s - mean) / (std + 1e-6), libero_all4_no_noops | P 170-187; SP 50-54, 92 | the release's method, the suite policy's statistics |
| instruction: raw, BERT (bert-base-uncased), padded to the embedded layout's length (21 default) | R 204, 262; text_encoder.py 61-159 | the release's method |
| one deterministic pass: 12 ACT queries, tanh | action_head.py 34-53 | the release's method |
| dims 0-5: 0.5 (a + 1)(max - min) + min; gripper sign (> 0 -> +1, < 0 -> -1, 0 -> +1) | SP 55-60, 96-103; P 205-210 | the release's method |
| 12 actions per query, all executed, no ensembling | R 55, 77, 242, 258-268 | `execute` = the chunk's length |
| bf16 for the whole model | P 370-375, 441-447, 523-526; the release's command | `precision="bf16"` |
| seed: `set_seed_everywhere(seed)` at the start and per task | R 57, 212, 306 | per episode, with the noise seed |

The class used is the suite policy (SP), as the release's evaluation does: the base class (P) holds other,
hard-coded statistics (P 31-83). Nothing of the release is copied here.
"""

from __future__ import annotations

import hashlib
import sys
from typing import Any, Callable, Dict, Optional

import numpy as np

from evaluation.interface import Actions, Adapter, Episode, Observation

SIZE = 256
STATS_KEY = "libero_all4_no_noops"


def release_observation(observation: Observation) -> Dict[str, np.ndarray]:
    """The LIBERO observation keys `predict_env_action_chunk_from_obs` reads (P 514-517, 549-557), filled
    from the shared observation as the simulator returned it: images not rotated (the release rotates)."""
    s = observation.state
    return {"agentview_image": observation.images["main"], "robot0_eye_in_hand_image": observation.images["wrist"],
            "robot0_eef_pos": np.asarray(s["eef_pos"], dtype=np.float64),
            "robot0_eef_quat": np.asarray(s["eef_quat"], dtype=np.float64),
            "robot0_gripper_qpos": np.asarray(s["gripper_qpos"], dtype=np.float64)}


class TurboVla(Adapter):
    def __init__(self, row: Dict[str, Any], release: Dict[str, Callable], loaded: Dict[str, Any]):
        self.row, self.release, self.loaded = row, release, loaded
        self.episode: Optional[Episode] = None

    def report(self) -> Dict[str, Any]:
        L = self.loaded
        return {
            "model": {"name": self.row["name"],
                      "checkpoint": {"address": self.row["checkpoint"]["address"], "revision": L["checkpoint_revision"],
                                     "sha256": L["checkpoint_sha256"], "licence": self.row["checkpoint"]["licence"]},
                      "code": {"address": self.row["code"]["address"], "revision": self.row["code"]["revision"]}},
            "inference": {
                "decoding": "one forward pass, ACT queries, tanh; no sampling", "chunk": L["chunk_size"],
                "executed_per_query": L["chunk_size"],
                "images": {"cameras": ["main", "wrist"], "size": SIZE,
                           "steps": ["rotate 180 degrees", "/255, DINOv3 mean and std from its preprocessor_config"]},
                "state": "read: eef position, axis-angle, gripper joints (8); (s - mean) / (std + 1e-6) with %s" % STATS_KEY,
                "prompt": "the instruction, BERT tokenizer (bert-base-uncased), padded to the layout's length",
                "model_config": L["model_config"],
                "unnormalisation": "dims 0-5 min-max from %s; gripper sign" % STATS_KEY,
                "precision": L["precision"], "encoders": L["encoders"]},
            "method": {"name": "none"},
            "noise_seed_controls": "nothing in the policy (deterministic); the seed seeds random, numpy and torch "
                                   "at the start of the episode, as the release does per task",
            "environment": {"name": self.row["environment"], "python": sys.version.split()[0], "packages": L["packages"]},
            "requires": {"camera_height": SIZE, "camera_width": SIZE},
        }

    def begin_episode(self, episode: Episode) -> None:
        self.release["set_seed_everywhere"](episode.noise_seed)
        self.episode = episode

    def act(self, observation: Observation) -> Actions:
        if self.episode is None:
            raise RuntimeError("act() before begin_episode()")
        for camera in ("main", "wrist"):
            if observation.images[camera].shape[:2] != (SIZE, SIZE):
                raise ValueError("TurboVLA takes %d px images only (P 131-135), got %s"
                                 % (SIZE, observation.images[camera].shape))
        chunk = np.asarray(self.release["predict"](release_observation(observation), observation.instruction),
                           dtype=np.float64)
        return Actions(actions=chunk, execute=len(chunk))


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def load(row: Dict[str, Any]) -> Adapter:
    """The weights at their pinned revisions, then the release's suite policy offline (P 86-96). The DINOv3
    directory is gated on the hub (a token with access must be present where this runs)."""
    import importlib.metadata as md

    from huggingface_hub import hf_hub_download, snapshot_download
    from turbovla.evaluation.suite_policy import TurboVLAPolicy, set_seed_everywhere

    ck = row["checkpoint"]
    ckpt = hf_hub_download(ck["hub_id"], ck["file"], revision=ck["revision"])
    stats = hf_hub_download(ck["hub_id"], ck["stats_file"], revision=ck["revision"])
    encoders = {}
    for name, spec in row["encoders"].items():
        encoders[name] = {"hub_id": spec["hub_id"], "revision": spec["revision"],
                          "path": snapshot_download(spec["hub_id"], revision=spec["revision"])}
    policy = TurboVLAPolicy(ckpt, dinov3_path=encoders["dinov3"]["path"], bert_path=encoders["bert"]["path"],
                            stats_path=stats, stats_key=STATS_KEY, precision="bf16", allow_hf_download=False)
    digest = _sha256(ckpt)
    if digest != ck["sha256"]:
        raise RuntimeError("checkpoint sha256 %s is not the pinned %s" % (digest, ck["sha256"]))
    config = getattr(getattr(policy, "model", None), "config", None)
    packages = {}
    for name in ("torch", "torchvision", "transformers", "tokenizers", "timm", "numpy", "Pillow", "turbovla"):
        try:
            packages[name] = md.version(name)
        except md.PackageNotFoundError:
            packages[name] = "not installed"
    loaded = {"checkpoint_revision": ck["revision"],
              "checkpoint_sha256": {ck["file"]: digest, ck["stats_file"]: _sha256(stats)},
              "chunk_size": int(policy.chunk_size), "precision": policy.precision,
              "model_config": config.to_dict() if hasattr(config, "to_dict") else repr(config),
              "encoders": {k: {"hub_id": v["hub_id"], "revision": v["revision"]} for k, v in encoders.items()},
              "packages": packages}
    release = {"set_seed_everywhere": set_seed_everywhere,
               "predict": lambda obs, instruction: policy.predict_env_action_chunk_from_obs(obs, instruction)}
    return TurboVla(row, release, loaded)
