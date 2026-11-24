"""APT: the authors' planner (`apt/infer/planner.py::TrajPlanner`, at d875861d) behind the interface.

The release has no licence, so none of its code is copied here; the planner and its helpers are imported from
the row's environment. What its LIBERO evaluator does (`examples/libero/eval_libero_pred.py`, "the evaluator"),
and this adapter follows, by line:

* per episode: `controller.reset()`, `set_config("Libero")`, `set_prompt(<instruction>)` (L255-260); the
  planner's data config `Libero` (cameras agentview and eye_in_hand, 256 x 256, one history frame, 32 future
  states, sample_dt 1.0, state gap 1; `data_utils/datasets.py` L86-96, `data_utils/dataset_base.py` L736-768);
* per model call: one observation frame (L147-184): the end-effector pose as a 4 x 4 matrix from position and
  quaternion; the gripper as qpos[0] / 0.04 (1 open, 0 closed); both images flipped vertically only (`[::-1]`),
  with each camera's intrinsics at 256 x 256 and its extrinsics (robosuite's functions); the timestamp = policy
  steps executed so far. Then `get_action(sample_num=1)` (L265-269): 32 future absolute poses and grippers;
* 3 of them are executed per call (L57, L284-293). **Each is turned into an action against the pose after the
  previous step** (L285-291): Δposition = target − current; Δrotation = rotvec(R_target · R_current⁻¹); both
  divided by the controller's action scale; gripper = −1 if g > 0.8 (open) else +1 (L122-144). So the adapter
  returns one action per query (`execute = 1`) and calls the model at every third query.
* not followed: the evaluator's 500-step limit (L56; the shared step limit applies), its unseeded noise (DDIM's
  initial noise is `torch.randn` without a generator, `apt/ae.py` L596; here it is seeded per model call), and
  its prompt (L252: `task.language` from the file name; here the instruction the block gives).

Inference as released: DDIM, 20 steps (`apt/vla.py` L222-223: inference_timesteps = 20), fp16 (planner L274).
The planner's trajectory ensembler is built (`--ensemble 4`, `infer/remote_service.py` L85) but never applied
on this path: the call in `get_action` is commented out (planner L351-353) and the evaluator never calls
`ensemble_traj`. So: no ensembling.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from evaluation.interface import Actions, Adapter, Episode, Observation
from evaluation.rows.pi05_cag import openpi_libero as L

CALL_EVERY = 3  # SAMPLE_STEPS_PER_QUERY, eval_libero_pred.py L57
SIZE = 256  # IMAGE_RESOLUTION, L55
OPEN_QPOS, CLOSE_QPOS = 0.04, 0.0  # L154
GRIPPER_THRESHOLD = 0.8  # L141
CKPT_DIR = "apt_vla_ftlibero"
CKPT_FILE = CKPT_DIR + "/ckpt_latest.pt"
ENSEMBLE = 4  # remote_service.py L85 default; unused on this path (see above)


def rotation(quat_xyzw) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat(np.asarray(quat_xyzw, np.float64)).as_matrix()


def rotvec(matrix: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(matrix).as_rotvec()


def apt_frame(observation: Observation) -> Dict[str, Any]:
    """The planner's observation frame, as the evaluator's `obs_libero2apt` builds it (L147-184)."""
    s = observation.state
    ee_pose = np.eye(4)
    ee_pose[:3, 3] = np.asarray(s["eef_pos"], np.float64)
    ee_pose[:3, :3] = rotation(s["eef_quat"])
    gripper = (float(np.asarray(s["gripper_qpos"])[0]) - CLOSE_QPOS) / (OPEN_QPOS - CLOSE_QPOS)

    def camera(ours: str) -> Dict[str, Any]:
        image = np.asarray(observation.images[ours])
        if image.shape[:2] != (SIZE, SIZE):
            raise ValueError("APT needs %d x %d renders, got %s" % (SIZE, SIZE, image.shape))
        c = observation.cameras[ours]
        return {
            "model": "pinhole",
            "camera": {"height": SIZE, "width": SIZE, "K": np.asarray(c["intrinsics"], np.float64).flatten().tolist()},
            "data": {"color": np.ascontiguousarray(image[::-1]), "wcT": np.asarray(c["extrinsics"], np.float64),
                     "seg": None, "depth": None},
        }

    return {"ee_pose": ee_pose, "gripper": gripper, "timestamp": int(observation.step),
            "agentview": camera("main"), "eye_in_hand": camera("wrist")}


def to_libero(target_pose: np.ndarray, target_gripper: float, observation: Observation) -> np.ndarray:
    """One absolute target -> one LIBERO action against the current pose (the evaluator's L122-144)."""
    s = observation.state
    scale = np.asarray(observation.controller["action_scale"], np.float64)
    delta_pos = target_pose[:3, 3] - np.asarray(s["eef_pos"], np.float64)
    delta_ori = rotvec(target_pose[:3, :3] @ np.linalg.inv(rotation(s["eef_quat"])))
    gripper = 1.0 - 2.0 * float(target_gripper > GRIPPER_THRESHOLD)
    return np.concatenate([np.concatenate([delta_pos, delta_ori]) / scale, [gripper]])


def call_seed(episode, query: int) -> int:
    """The seed of one model call: drawn from the shared rule (`interface.noise_generator`) at the query of the call."""
    from evaluation import interface

    return int(interface.noise_generator(episode, query).integers(0, 2**31 - 1))


class Apt(Adapter):
    def __init__(self, row: Dict[str, Any], planner=None, seeder=None):
        self.row = row
        if planner is None:
            from apt.infer.planner import TrajPlanner

            ck = row["checkpoint"]
            snapshot = L.hub_snapshot(ck["address"].split("huggingface.co/")[1], ck["revision"], allow=[CKPT_DIR + "/*"])
            self._snapshot = snapshot
            planner = TrajPlanner(ckpt_path=str(snapshot / CKPT_FILE), device="cuda:0", ensemble=ENSEMBLE, use_ema=False)
        self.planner = planner
        self._seeder = seeder or seed_everything
        self._episode: Optional[Episode] = None
        self._queue: List = []
        self._calls = 0

    def begin_episode(self, episode: Episode) -> None:
        self._episode = episode
        self._queue, self._calls = [], 0
        self.planner.reset()
        self.planner.set_config("Libero")
        self.planner.set_prompt(episode.instruction)

    def _call(self, observation: Observation) -> None:
        self._seeder(call_seed(self._episode, observation.query))
        self.planner.add_obs_frame(apt_frame(observation))
        poses, grippers, times, _ = self.planner.get_action(sample_num=1)
        poses, grippers = np.asarray(poses)[0, :, 0], np.asarray(grippers)[0, :, 0]
        if int(self.planner.config.sample_state_gaps) != 1:  # the evaluator's L271-279
            from examples.libero import align

            query_time = observation.step + np.arange(len(poses) * self.planner.config.sample_state_gaps)
            data = align.align_data(query_time, np.asarray(times)[0] if np.ndim(times) > 1 else times,
                                    {"ee_pose": poses, "gripper": grippers},
                                    {"ee_pose": align.interp_SE3_sep, "gripper": align.interp_linear})
            poses, grippers = data["ee_pose"], data["gripper"]
        self._queue = [(poses[k], grippers[k]) for k in range(CALL_EVERY)]
        self._calls += 1

    def act(self, observation: Observation) -> Actions:
        if not self._queue:
            self._call(observation)
        pose, gripper = self._queue.pop(0)
        return Actions(actions=to_libero(pose, gripper, observation)[None, :], execute=1)

    def report(self) -> Dict[str, Any]:
        c = self.planner.config
        return {
            "model": {
                "name": "APT (frozen-VLM LIBERO fine-tune)",
                "checkpoint": dict(self.row["checkpoint"], file=CKPT_FILE,
                                   sha256={"ckpt_latest.pt": L.file_sha256(self._snapshot / CKPT_FILE),
                                           "config.json": L.file_sha256(self._snapshot / CKPT_DIR / "config.json")},
                                   vlm=dict(self.row["vlm"], loaded_revision=vlm_revision(self.planner))),
                "code": dict(self.row["code"]),
            },
            "inference": {
                "sampler": "DDIM",
                "sampling_steps": int(getattr(self.planner.model.actor, "inference_timesteps", -1)) if hasattr(self.planner.model, "actor") else "not read",
                "future_states_per_call": int(c.num_future_states),
                "actions_executed_per_model_call": CALL_EVERY,
                "actions_per_query": 1,
                "conversion": "each target turned into a delta action against the pose after the previous step, divided by the controller's action_scale",
                "images": "main and wrist at 256 x 256, flipped vertically only, with intrinsics and extrinsics",
                "state": "end-effector pose (4 x 4) and gripper opening qpos[0] / 0.04",
                "history": {"cameras": int(c.num_history_cameras), "states": int(c.num_history_states),
                            "camera_gap": int(c.sample_camera_gaps), "state_gap": int(c.sample_state_gaps)},
                "data_config": "Libero",
                "camera_names": list(c.camera_names),
                "gripper": "open (-1) when the predicted opening > 0.8, else closed (+1)",
                "prompt": "the instruction as given",
                "precision": "fp16 (planner _run_inference)",
                "ensembling": "none: TrajPlanner's ensembler is not applied in get_action (planner L351-353)",
            },
            "method": {"name": "none"},
            "noise_seed_controls": ("Every model call seeds torch (CPU and CUDA), NumPy's and Python's global generators with "
                                    "interface.noise_generator(episode, query of the call).integers(0, 2**31 - 1) (the shared rule: "
                                    "noise seed, task index, start state, query index); the DDIM initial noise is drawn from "
                                    "torch's generator. The release seeds nothing."),
            "environment": {"name": self.row["environment"], "python": L.python_version(),
                            "packages": L.versions("torch", "transformers", "diffusers", "scipy", "numpy", "huggingface-hub")},
            "requires": {"camera_height": SIZE, "camera_width": SIZE},
        }


def seed_everything(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def vlm_revision(planner) -> str:
    """The Qwen3-VL revision transformers actually loaded (the release names the model, not a revision)."""
    for path in ("model.vlm.config._commit_hash", "model.vlm.qwen.config._commit_hash"):
        node = planner
        try:
            for part in path.split("."):
                node = getattr(node, part)
            if node:
                return str(node)
        except AttributeError:
            continue
    return "not read"


def load(row: Dict[str, Any]) -> Adapter:
    return Apt(row)
