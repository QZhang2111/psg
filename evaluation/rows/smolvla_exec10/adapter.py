"""SmolVLA behind the shared interface: lerobot's own LIBERO evaluation, query by query, 10 actions executed per
query. The SmolVLA row of the paper's tables.

The release is github.com/huggingface/lerobot @ e581c1d5701f846788662079d4ac08835c96def9 with the checkpoint
lerobot/smolvla_libero @ a24f1f380461b665158d52acd140152c79e8b36d (row.json). Its evaluation is
`lerobot-eval --env.type=libero --policy.path=lerobot/smolvla_libero` with the documented rename map
(docs/source/simulation.mdx L21-28). Files below are under src/lerobot/.

| step | release | here |
|---|---|---|
| state = eef position, quaternion -> axis-angle (float32, zeros near identity), gripper joints (8) | sim/integrations/libero.py L40-57 | `_axisangle` (libero.py imports the simulator; its five lines written here) |
| rotate both images 180 degrees; keys observation.images.image / image2, observation.state | sim/integrations/utils.py L28-55 `canonical_observation(..., flip_images=True)` | the release's function |
| uint8 BHWC -> float32 BCHW / 255 | envs/utils.py L134-144 `prepare_canonical_observation` | the release's function |
| task string = the instruction | scripts/lerobot_eval.py L313 | `batch["task"]` |
| rename image -> camera1, image2 -> camera2; newline; tokenizer (48, right padding); state MEAN_STD; device | the checkpoint's policy_preprocessor.json via policies/factory.py `make_pre_post_processors`, with the eval's overrides (lerobot_eval.py L870-880) | the release's pipeline, the same overrides |
| resize with pad to 512, [-1, 1], 10 Euler steps from N(0, 1) noise of shape (1, 50, 32), first 7 dims | policies/smolvla/modeling_smolvla.py L240-253, L403-413, L790-825 | `predict_action_chunk` |
| un-normalise MEAN_STD with the action statistics | the checkpoint's policy_postprocessor.json | the release's pipeline, per action as the eval applies it |
| the first n_action_steps actions of each chunk of 50 are executed, then the model is asked again; the checkpoint's n_action_steps is 50 | modeling_smolvla.py L256-288 (the queue takes `[: n_action_steps]`, L283) | the first `executed_per_query` actions, a setting stated in the row's row.json: 10 |
| no clip, no gripper conversion | lerobot_eval.py L330-341 | none |
| TF32 on, cudnn.benchmark on | lerobot_eval.py L829-832, configs/eval.py L41 | set at load |

One difference, and why: the release draws the flow's noise from torch's global stream, seeded once per run
(seed 1000), so an episode's noise depends on what ran before it. Here the noise of every query is drawn from a
generator of the shared rule (interface.noise_generator: noise seed, task, start state, query index), drawn
with NumPy and passed to `predict_action_chunk(noise=...)`, the release's own argument; the distribution is the
release's. Render size: the model was trained on 256 px frames
(config.json input_features 3x256x256); lerobot's LIBERO evaluation renders 360 (envs/configs.py L480-481). The
row asks for 360, the size the release is evaluated at; the report says so.

Executed actions per query: the number is the row's, `executed_per_query` in row.json, never the checkpoint's
default read silently. The release's default is 50 (config.json n_action_steps); this row states 10, which is
what the release's own queue does at n_action_steps = 10 (modeling_smolvla.py L283): of each chunk of 50 the first 10 are
un-normalised and executed, the other 40 are dropped, and the next query draws its own noise by the shared
rule. Where the number is not the release's default the report says so beside it.
"""

from __future__ import annotations

import hashlib
import os
import sys
from typing import Any, Callable, Dict, Optional

import numpy as np

from evaluation.interface import NOISE_RULE, Actions, Adapter, Episode, Observation, noise_generator

RENAME_MAP = {"observation.images.image": "observation.images.camera1",
              "observation.images.image2": "observation.images.camera2"}
TRAINED_SIZE = 256
RELEASE_EVAL_SIZE = 360
#: the render size asked for: 360, lerobot's evaluation size, unless PSG_SMOLVLA_CAMERA_SIZE says otherwise (256 for
#: a comparison at the training size). The config record states it.
CAMERA_SIZE = int(os.environ.get("PSG_SMOLVLA_CAMERA_SIZE", RELEASE_EVAL_SIZE))


def _axisangle(quat: np.ndarray) -> np.ndarray:
    """sim/integrations/libero.py L40-46: xyzw -> axis-angle in float32, zeros near the identity."""
    quat = np.asarray(quat, dtype=np.float32)
    w = np.clip(quat[..., 3], -1, 1)
    den = np.sqrt(np.maximum(1 - w * w, 0))
    scale = np.divide(2 * np.arccos(w), den, out=np.zeros_like(den), where=den > 1e-10)
    return (quat[..., :3] * scale[..., None]).astype(np.float32)


def raw_observation(observation: Observation) -> Dict[str, Any]:
    """The dictionary `canonical_observation` takes, as libero.py L49-57 builds it from LIBERO's observation."""
    s = observation.state
    agent_pos = np.concatenate([np.asarray(s["eef_pos"], dtype=np.float32), _axisangle(s["eef_quat"]),
                                np.asarray(s["gripper_qpos"], dtype=np.float32)]).astype(np.float32)
    return {"pixels": {"image": observation.images["main"], "image2": observation.images["wrist"]},
            "agent_pos": agent_pos}


def query_noise(episode: Episode, query: int, chunk: int, width: int) -> np.ndarray:
    """The shared rule (interface.noise_generator): the flow's initial noise for one query, N(0, 1) of shape
    (1, chunk, width), drawn first and only from the generator of (noise seed, task, start state, query)."""
    return noise_generator(episode, query).standard_normal((1, chunk, width)).astype(np.float32)


def executed_per_query(row: Dict[str, Any], loaded: Dict[str, Any]) -> int:
    """How many actions of each chunk the row executes before it asks again: `executed_per_query` of its
    row.json. A row that does not state it is refused; the checkpoint's n_action_steps is not taken in its place."""
    name, stated, chunk = row.get("row"), row.get("executed_per_query"), int(loaded["chunk_size"])
    if isinstance(stated, bool) or not isinstance(stated, int):
        raise SystemExit("row %r: row.json must state \"executed_per_query\" as a whole number, got %r" % (name, stated))
    if not 1 <= stated <= chunk:
        raise SystemExit("row %r: executed_per_query = %d is outside 1..%d, the chunk" % (name, stated, chunk))
    return stated


class SmolVla(Adapter):
    """The release's functions and objects are passed in (`release`), so the glue can be read and tested
    without lerobot."""

    def __init__(self, row: Dict[str, Any], release: Dict[str, Callable], loaded: Dict[str, Any]):
        self.row, self.release, self.loaded = row, release, loaded
        self.executed = executed_per_query(row, loaded)
        self.episode: Optional[Episode] = None

    def _beside_the_release(self) -> Dict[str, Any]:
        """What the record says beside `executed_per_query` where it is not the release's default; nothing
        where it is."""
        default = int(self.loaded["n_action_steps"])
        if self.executed == default:
            return {}
        return {
            "executed_per_query_release_default": default,
            # The wording of the config records this row wrote, kept as they hold it; the row smolvla (50 executed
            # actions, no results here) was folded into this one.
            "executed_per_query_note": "a setting of this row (row.json), not the release's default: the first %d "
            "actions of each chunk of %d are executed and the model is asked again, as the release's queue does at "
            "n_action_steps = %d (modeling_smolvla.py L283); nothing else differs from the row smolvla"
            % (self.executed, int(self.loaded["chunk_size"]), self.executed),
        }

    def report(self) -> Dict[str, Any]:
        L = self.loaded
        return {
            "model": {"name": self.row["name"],
                      "checkpoint": {"address": self.row["checkpoint"]["address"], "revision": L["checkpoint_revision"],
                                     "sha256": L["checkpoint_sha256"], "licence": self.row["checkpoint"]["licence"]},
                      "code": {"address": self.row["code"]["address"], "revision": self.row["code"]["revision"]}},
            "inference": {
                "flow": "Euler, %s steps, t from 1 to 0" % L["num_steps"], "chunk": L["chunk_size"],
                "executed_per_query": self.executed,
                **self._beside_the_release(),
                "images": {"cameras": ["main -> camera1", "wrist -> camera2"],
                           "steps": ["rotate 180 degrees", "/255", "resize with pad to %s (bilinear)" % (L["resize"],),
                                     "x2 - 1"]},
                "state": "read: eef position, axis-angle (float32), gripper joints (8); MEAN_STD; padded to %s; one token"
                         % L["max_state_dim"],
                "prompt": "the instruction + '\\n', SmolVLM2 tokenizer, %s tokens, right padding" % L["tokenizer_max_length"],
                "vlm": {"id": L["vlm_id"], "revision_resolved": L["vlm_revision"], "pinned_by_the_release": False},
                "unnormalisation": "MEAN_STD with the checkpoint's action statistics; gripper continuous, not binarized",
                "precision": L["dtype"], "tf32": True, "cudnn_benchmark": True,
                "render_size": {"asked": CAMERA_SIZE, "trained_on": TRAINED_SIZE,
                                "release_evaluation": RELEASE_EVAL_SIZE},
                "release_max_steps_spatial": 280, "release_seed": "set_seed(1000) once per run; env seed 1000 + k"},
            "method": {"name": "none"},
            "noise_seed_controls": "the flow's initial noise of every query, (1, %d, %d) float32 drawn first from the "
                                   "shared generator and passed to predict_action_chunk(noise=...): " % (L["chunk_size"], L["max_action_dim"]) + NOISE_RULE,
            "environment": {"name": self.row["environment"], "python": sys.version.split()[0], "packages": L["packages"]},
            "requires": {"camera_height": CAMERA_SIZE, "camera_width": CAMERA_SIZE},
        }

    def begin_episode(self, episode: Episode) -> None:
        self.release["reset"]()
        self.episode = episode

    def act(self, observation: Observation) -> Actions:
        if self.episode is None:
            raise RuntimeError("act() before begin_episode()")
        R = self.release
        canonical = R["canonical_observation"](raw_observation(observation), flip_images=True)
        batch = R["to_tensors"]({k: np.asarray(v)[None] for k, v in canonical.items()})
        batch["task"] = [observation.instruction]
        batch = R["preprocessor"](batch)
        noise = R["noise"](query_noise(self.episode, observation.query, self.loaded["chunk_size"], self.loaded["max_action_dim"]))
        chunk = R["predict_action_chunk"](batch, noise)  # (1, chunk, 7), normalised
        n = self.executed
        actions = np.stack([R["postprocess"](chunk, i) for i in range(n)])
        return Actions(actions=actions.astype(np.float64), execute=n)


def load(row: Dict[str, Any]) -> Adapter:
    import importlib.metadata as md

    import torch
    from huggingface_hub import HfApi, snapshot_download
    from lerobot.envs.utils import prepare_canonical_observation
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.sim.integrations.utils import canonical_observation

    hub_id, pinned = row["checkpoint"]["hub_id"], row["checkpoint"]["revision"]
    folder = snapshot_download(hub_id, revision=pinned)
    torch.backends.cudnn.benchmark = True  # lerobot_eval.py L829
    torch.backends.cuda.matmul.allow_tf32 = True  # L830-832 with configs/eval.py tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = SmolVLAPolicy.from_pretrained(hub_id, revision=pinned)
    policy.config.device = device
    policy.to(device).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=hub_id, pretrained_revision=pinned,
        preprocessor_overrides={"device_processor": {"device": device},
                                "rename_observations_processor": {"rename_map": RENAME_MAP}})
    cfg = policy.config

    def noise(array):
        return torch.from_numpy(array).to(device=device, dtype=torch.float32)

    def predict(batch, noise_tensor):
        with torch.inference_mode():
            return policy.predict_action_chunk(batch, noise=noise_tensor)

    def postprocess(chunk, i):
        return post(chunk[:, i, :]).to("cpu").numpy()[0]

    vlm_id = cfg.vlm_model_name
    packages = {}
    for name in ("lerobot", "torch", "transformers", "tokenizers", "numpy", "num2words"):
        try:
            packages[name] = md.version(name)
        except md.PackageNotFoundError:
            packages[name] = "not installed"
    files = {}
    import os
    for root, _, names in os.walk(folder):
        for name in sorted(names):
            p = os.path.join(root, name)
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for blockdata in iter(lambda: f.read(1 << 24), b""):
                    h.update(blockdata)
            files[os.path.relpath(p, folder)] = h.hexdigest()
    loaded = {"checkpoint_revision": pinned,
              "checkpoint_sha256": {"files": dict(sorted(files.items())), "how": "sha256 of every file of the snapshot"},
              "num_steps": cfg.num_steps, "chunk_size": cfg.chunk_size, "n_action_steps": cfg.n_action_steps,
              "max_action_dim": cfg.max_action_dim,
              "resize": tuple(cfg.resize_imgs_with_padding), "max_state_dim": cfg.max_state_dim,
              "tokenizer_max_length": cfg.tokenizer_max_length, "vlm_id": vlm_id,
              "vlm_revision": HfApi().model_info(vlm_id).sha, "dtype": str(next(policy.parameters()).dtype),
              "packages": packages}
    release = {"reset": policy.reset, "canonical_observation": canonical_observation,
               "to_tensors": prepare_canonical_observation, "preprocessor": pre, "noise": noise,
               "predict_action_chunk": predict, "postprocess": postprocess}
    return SmolVla(row, release, loaded)
