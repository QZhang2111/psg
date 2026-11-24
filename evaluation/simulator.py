"""The simulator, behind the few methods the rollout uses. The only module that imports the benchmark's code.

A `Scene` is one task of one block: its environment, built from the task's BDDL file with the two cameras at
the row's render size. `reset(state)` puts it at a start state the same way before every episode:

    env.seed(environment seed);  env.reset();  env.set_init_state(start state)

so an episode does not depend on the episodes run before it (the benchmark draws its fixtures' places at every
reset from NumPy's global generator, which the seed sets).

The four kinds of block, named by a block's `loader` (the word is checked; what the scene does follows the task's
own fields, `relocate` and `predicate`):

    libero      a BDDL file and one row of its init-state file
    layout      the same; LIBERO-Plus gives one start state per variant
    relocate    our scenes: the base task's start state with one object's free joint set to a new (x, y)
    conditions  LIBERO-CF: the same start, with the task's own conditions deciding the end and the success

Runs in a simulator environment (Python 3.8 and later; `libero`, `robosuite`, `mujoco`, `torch` as the
benchmark installs them). Written against the benchmarks' code as read, and checked against them on a simulator.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from evaluation import interface

ROBOT_PREFIXES = ("robot0_", "gripper0_", "mount0_")  # bodies of the robot itself
GRIPPER_PREFIX = "gripper0_"
LOADERS = ("libero", "layout", "relocate", "conditions")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_files(task: Dict[str, Any]) -> Tuple[Path, Path]:
    """The task's BDDL file and init-state file in this environment's copy of the benchmark."""
    from libero.libero import get_libero_path

    return Path(get_libero_path("bddl_files")) / task["bddl"], Path(get_libero_path("init_states")) / task["init"]


def task_digests(task: Dict[str, Any]) -> Dict[str, str]:
    """Digests of the two files as loaded; a digest pinned in the block's definition must match."""
    bddl, init = task_files(task)
    found = {"bddl_sha256": sha256(bddl), "init_sha256": sha256(init)}
    for key, value in found.items():
        if task.get(key) and task[key] != value:
            raise SystemExit("task %r: %s is %s here, the block's definition pins %s" % (task["task"], key, value, task[key]))
    return found


def start_states(task: Dict[str, Any]) -> np.ndarray:
    """The task's start states, [N, state size], as the benchmark stores them."""
    import torch

    _, init = task_files(task)
    try:
        states = torch.load(str(init), weights_only=False)
    except TypeError:  # a torch without the argument
        states = torch.load(str(init))
    states = np.asarray(states, dtype=np.float64)
    if states.ndim == 1:  # LIBERO-Plus's libero_newobj files hold one state as a vector; its loader makes it one row
        states = states.reshape(1, -1)  # (libero/libero/benchmark/__init__.py L238-239 @ 4976dc30)
    return states


def measured() -> Dict[str, Any]:
    """What this simulator environment is, for the config record."""
    try:
        from importlib import metadata
    except ImportError:  # Python 3.7
        import importlib_metadata as metadata  # type: ignore

    packages = {}
    for name in ("libero", "robosuite", "mujoco", "numpy", "torch", "bddl"):
        try:
            packages[name] = metadata.version(name)
        except Exception:
            packages[name] = "not installed as a distribution"
    return {"simulator": {"packages": packages, "renderer": os.environ.get("MUJOCO_GL", "the simulator's default")}}


def camera_intrinsics(sim: Any, camera: str, height: int, width: int) -> np.ndarray:
    """robosuite's `camera_utils.get_camera_intrinsic_matrix`, written out: that module imports `h5py`, which
    a simulator environment does not otherwise need."""
    fovy = sim.model.cam_fovy[sim.model.camera_name2id(camera)]
    focal = 0.5 * height / np.tan(fovy * np.pi / 360)
    return np.array([[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]])


def camera_extrinsics(sim: Any, camera: str) -> np.ndarray:
    """robosuite's `camera_utils.get_camera_extrinsic_matrix`, written out: the camera's pose in the world, with
    robosuite's correction of the camera's axes (y and z flipped)."""
    camera_id = sim.model.camera_name2id(camera)
    pose = np.eye(4)
    pose[:3, :3] = np.asarray(sim.data.cam_xmat[camera_id]).reshape(3, 3)
    pose[:3, 3] = np.asarray(sim.data.cam_xpos[camera_id])
    return pose @ np.diag([1.0, -1.0, -1.0, 1.0])


def _numeric(owner: Any) -> Dict[str, Any]:
    """The numeric attributes of an object, copied: arrays of numbers, numbers, booleans and None."""
    out: Dict[str, Any] = {}
    for name, value in vars(owner).items():
        if isinstance(value, np.ndarray):
            if value.dtype != object:
                out[name] = value.copy()
        elif value is None or isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
            out[name] = value
    return out


def _same_placement(a: Any, b: Any) -> bool:
    return all(x.shape == y.shape and np.allclose(x, y, rtol=0, atol=1e-9) for x, y in zip(a, b))


class Scene:
    def __init__(self, block: Dict[str, Any], task: Dict[str, Any], camera_height: int, camera_width: int) -> None:
        from libero.libero.envs import OffScreenRenderEnv

        if block["loader"] not in LOADERS:
            raise SystemExit("block %r names an unknown loader %r" % (block["block"], block["loader"]))
        self._block, self._task = block, task
        self._size = (int(camera_height), int(camera_width))
        self._seed = int(block["rollout"]["environment_seed"])
        self._states = start_states(task)
        self._bddl = str(task_files(task)[0])
        self._make = lambda: OffScreenRenderEnv(
            bddl_file_name=self._bddl, camera_heights=self._size[0], camera_widths=self._size[1]
        )
        self._env = self._make()
        self._placement: Any = None  # the static bodies' placement of this scene's first reset
        self.rebuilds = 0  # times a reset drew another placement and the environment was built anew
        self._obs: Dict[str, Any] = {}
        self._tables: Dict[str, Any] = {}
        self._robot_geoms: set = set()
        self._gripper_geoms: set = set()

    # ---------------------------------------------------------------------------------------------- the episode

    def _seeded_reset(self) -> Any:
        """Environment seed, then reset; returns where the reset placed the model's bodies (the fixtures are
        placed here and are not part of a start state)."""
        self._env.seed(self._seed)
        self._env.reset()
        model = self._env.sim.model._model
        return np.asarray(model.body_pos, np.float64).copy(), np.asarray(model.body_quat, np.float64).copy()

    def reset(self, state: int) -> None:
        """Environment seed, reset, start state: the same scene before every episode.

        The scene of an episode is the one a fresh environment's first seeded reset gives. A reset of an
        environment that has already run draws the same placement in most tasks but not in all (measured:
        LIBERO-Spatial task 4, cabinet and stove 1-2 cm elsewhere). So every reset is compared with the scene's
        first; when it differs, the environment is built anew, and its first reset has to give the first
        placement again.
        """
        start = np.array(self._states[int(state)], dtype=np.float64)
        placement = self._seeded_reset()
        if self._placement is None:
            self._placement = placement
        elif not _same_placement(placement, self._placement):
            self._env.close()
            self._env = self._make()
            self.rebuilds += 1
            if not _same_placement(self._seeded_reset(), self._placement):
                raise RuntimeError("a fresh environment's seeded reset did not reproduce the scene's placement")
        if self._task.get("relocate"):
            start = self._relocated(start)
        self._obs = self._env.set_init_state(start)
        self._read_tables()

    def _relocated(self, start: np.ndarray) -> np.ndarray:
        """The start state with one object's free joint moved to the task's (x, y); height and orientation kept."""
        move = self._task["relocate"]
        joint = self._env.env.objects_dict[move["body"]].joints[0]
        address = self._env.sim.model.get_joint_qpos_addr(joint)
        first = int(address[0] if isinstance(address, (tuple, list, np.ndarray)) else address)
        start = start.copy()
        start[1 + first : 1 + first + 2] = np.asarray(move["xy"], dtype=np.float64)  # the state is [time, qpos, qvel]
        return start

    def step(self, action: Any) -> None:
        self._obs, _, _, _ = self._env.step(np.asarray(action, dtype=np.float64).tolist())

    def status(self) -> Tuple[bool, bool]:
        """(the episode ends now, it has succeeded)."""
        predicate = self._task.get("predicate")
        if not predicate:
            success = bool(self._env.check_success())
            return success, success
        conditions = list(predicate["stop_any"])
        results = self._env.evaluate_conditions(conditions + [predicate["success"]])
        return any(bool(results[c]) for c in conditions), bool(results[predicate["success"]])

    def close(self) -> None:
        self._env.close()

    # ------------------------------------------------------------------------------- what crosses the interface

    def observation(self) -> Dict[str, Any]:
        """The fields of `interface.Observation` that the simulator provides."""
        sim, obs = self._env.sim, self._obs
        height, width = self._size
        return {
            "images": {name: np.asarray(obs[camera + "_image"]) for name, camera in interface.CAMERAS.items()},
            "state": {
                "eef_pos": np.asarray(obs["robot0_eef_pos"], np.float64),
                "eef_quat": np.asarray(obs["robot0_eef_quat"], np.float64),
                "gripper_qpos": np.asarray(obs["robot0_gripper_qpos"], np.float64),
                "joint_pos": np.asarray(obs["robot0_joint_pos"], np.float64),
            },
            "cameras": {
                name: {
                    "intrinsics": camera_intrinsics(sim, camera, height, width),
                    "extrinsics": camera_extrinsics(sim, camera),
                }
                for name, camera in interface.CAMERAS.items()
            },
            "controller": {"action_scale": self.action_scale()},
        }

    def action_scale(self) -> np.ndarray:
        """The arm controller's action scale, [6]. The controller computes it at its first action and holds None
        until then; before that it is computed here by the controller's own formula from its own bounds."""
        controller = self._env.env.robots[0].controller
        scale = controller.action_scale
        if scale is None:
            scale = np.abs(np.asarray(controller.output_max) - np.asarray(controller.output_min)) / np.abs(
                np.asarray(controller.input_max) - np.asarray(controller.input_min)
            )
        scale = np.broadcast_to(np.asarray(scale, np.float64), (6,)).copy()
        if not np.isfinite(scale).all():
            raise RuntimeError("the controller's action scale is not finite: %s" % scale)
        return scale

    def main_image(self) -> np.ndarray:
        return np.asarray(self._obs[interface.CAMERAS["main"] + "_image"])

    def oracle(self, kinds: Any) -> Dict[str, Any]:
        """Privileged state for a row whose own definition asks for it (`evaluation/oracle.py`); never
        called for a row of the paper's tables. The position is the simulator's own `body_xpos`, the array the
        trace keeps per frame as `body_pos`."""
        from evaluation import oracle as _oracle

        out: Dict[str, Any] = {}
        for kind in kinds:
            if kind != "reach_target_position":
                raise SystemExit("the simulator has no oracle %r" % kind)
            target, body, index = _oracle.target_body(self._tables, self._task.get("reach"))
            position = np.asarray(self._env.sim.data.body_xpos[index], np.float64).copy()
            out["reach_target"] = {"target": target, "body": body, "position": position}
        return out

    # ------------------------------------------------------- snapshots (for instruments; no unit takes or restores one)

    def _dynamics(self) -> Tuple[Any, Any, Any, Any]:
        import mujoco

        env = self._env.env  # the robosuite environment inside the benchmark's wrapper
        return mujoco, env, self._env.sim.model._model, self._env.sim.data._data

    def snapshot(self) -> Dict[str, Any]:
        """Everything the dynamics carry from one step to the next, so that a restored snapshot continues as the
        original did.

        MuJoCo's integration state (time, positions, velocities, actuator state, the solver's warm start, controls,
        applied forces, mocap, user data), and what the robot keeps outside MuJoCo: every numeric attribute of the
        arm controller (its goal and the kinematics it caches), of the gripper model (the Panda's finger command
        accumulates step by step) and of the robot, and the environment's step counters. Not the observables: a
        restored scene continues the dynamics; its images and cached observation are the last step's until the next
        step. Whether a snapshot is honest is measured by who uses it (the same actions from it must give the same
        frames), not assumed.
        """
        mujoco, env, model, data = self._dynamics()
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(model, spec), np.float64)
        mujoco.mj_getState(model, data, state, spec)
        robot = env.robots[0]
        controller = robot.controller
        if getattr(controller, "interpolator_pos", None) is not None or getattr(controller, "interpolator_ori", None) is not None:
            raise RuntimeError("the arm controller interpolates its goal; that state is not part of a snapshot")
        return {
            "mujoco": state,
            "controller": _numeric(controller),
            "gripper": _numeric(robot.gripper),
            "robot": _numeric(robot),
            "counters": {name: getattr(env, name) for name in ("timestep", "cur_time", "done") if hasattr(env, name)},
        }

    def restore(self, snapshot: Dict[str, Any]) -> None:
        """Puts the scene back at a snapshot of this same scene. No forward pass: it would rewrite the solver's warm
        start, which is part of the state; the next step begins with its own, as the original's did. Derived
        quantities (body positions, contacts) and the observation are stale until that step."""
        mujoco, env, model, data = self._dynamics()
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        if snapshot["mujoco"].shape != (mujoco.mj_stateSize(model, spec),):
            raise RuntimeError("the snapshot is of another model")
        mujoco.mj_setState(model, data, snapshot["mujoco"], spec)
        robot = env.robots[0]
        for owner, values in ((robot.controller, snapshot["controller"]), (robot.gripper, snapshot["gripper"]), (robot, snapshot["robot"]), (env, snapshot["counters"])):
            for name, value in values.items():
                setattr(owner, name, value.copy() if isinstance(value, np.ndarray) else value)

    # ----------------------------------------------------------------------------------------- what is recorded

    def _read_tables(self) -> None:
        import mujoco

        model = self._env.sim.model._model
        bodies = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(model.nbody)]
        geoms = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(model.ngeom)]
        geom_body = np.asarray(model.geom_bodyid, np.int32)
        objects = list(self._env.env.objects_dict)
        fixtures = list(getattr(self._env.env, "fixtures_dict", {}))
        self._robot_geoms = {g for g in range(model.ngeom) if bodies[geom_body[g]].startswith(ROBOT_PREFIXES)}
        self._gripper_geoms = {g for g in range(model.ngeom) if bodies[geom_body[g]].startswith(GRIPPER_PREFIX)}
        self._tables = {
            "body_name": np.asarray(bodies),
            "body_parent": np.asarray(model.body_parentid, np.int32),
            "geom_name": np.asarray(geoms),
            "geom_body": geom_body,
            "geom_pos": np.asarray(model.geom_pos, np.float32),
            "geom_size": np.asarray(model.geom_size, np.float32),
            "geom_type": np.asarray(model.geom_type, np.int32),
            "geom_collides": (np.asarray(model.geom_contype) != 0) | (np.asarray(model.geom_conaffinity) != 0),
            "entity_name": np.asarray(objects + fixtures),
            "entity_kind": np.asarray(["object"] * len(objects) + ["fixture"] * len(fixtures)),
            "gripper_geoms": np.asarray(sorted(self._gripper_geoms), np.int32),
        }

    def tables(self) -> Dict[str, Any]:
        return self._tables

    def frame(self) -> Dict[str, Any]:
        """The per-frame arrays of the trace: the robot, every body's pose, the gripper's contacts."""
        data, obs = self._env.sim.data, self._obs
        contacts: List[Tuple[int, int, float]] = []
        for contact in data.contact[: data.ncon]:
            first, second = int(contact.geom1), int(contact.geom2)
            for gripper, other in ((first, second), (second, first)):
                if gripper in self._gripper_geoms and other not in self._robot_geoms:
                    contacts.append((gripper, other, float(contact.dist)))
        return {
            "eef_pos": np.asarray(obs["robot0_eef_pos"], np.float32),
            "eef_quat": np.asarray(obs["robot0_eef_quat"], np.float32),
            "gripper_qpos": np.asarray(obs["robot0_gripper_qpos"], np.float32),
            "joint_pos": np.asarray(obs["robot0_joint_pos"], np.float32),
            "body_pos": np.asarray(data.body_xpos, np.float32).copy(),
            "body_quat": np.asarray(data.body_xquat, np.float32).copy(),
            "contacts": contacts,
        }
