"""The interface between the evaluation and a model: the one line every row shares.

The evaluation owns the simulator. At every query it hands the model one `Observation`, the same for every
row, and receives one `Actions`. Everything a release does to its input or output (flip, resize, crop, pad,
state or no state, prompt template, normalisation, gripper sign, action ensembling) happens inside the row's
adapter and nowhere else.

Runs under Python 3.8 and later, with NumPy only: this module is imported in every simulator environment and
in every model environment.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional

import numpy as np

#: The two cameras of every LIBERO-family benchmark, by the name this repository uses and the simulator's name.
CAMERAS = {"main": "agentview", "wrist": "robot0_eye_in_hand"}

#: An action as `env.step` takes it: end-effector translation (3) and rotation as axis-angle (3), both as
#: increments in the controller's normalised units, then the gripper command (-1 open, +1 close).
ACTION_DIM = 7

#: The render size when a row does not declare one. 256 px is what the LIBERO demonstrations were rendered at.
DEFAULT_CAMERA_SIZE = 256


@dataclasses.dataclass
class Episode:
    """What the model is told once, before the first query of an episode. `Adapter.begin_episode` receives it."""

    episode: str  # "<block>.<row>.t<task_index>.s<state>.n<seed>"
    block: str
    task_index: int
    task: str  # the task's name in its benchmark
    #: the LIBERO suite the task's scene comes from: "libero_spatial" | "libero_object" | "libero_goal" |
    #: "libero_10". A release with per-suite statistics or checkpoints chooses by it.
    suite: str
    state: int  # index of the start state
    noise_seed: int  # what it controls is the row's: the adapter says so in `report()["noise_seed_controls"]`
    instruction: str  # the text for this episode; it is also in every observation


#: The noise rule, in words, for a row's `noise_seed_controls`.
NOISE_RULE = (
    "numpy.random.default_rng(SeedSequence([noise seed, task index, start state, query index])): every episode "
    "of a unit has its own noise, rows on one model draw the same noise in the same episode, and an episode "
    "gets the same noise on any machine, in any order and under any sharding"
)


def noise_generator(episode: Episode, query: int) -> np.random.Generator:
    """The one source of a query's randomness, for every row that samples.

    A function of (noise seed, episode, query index) and of nothing else, the episode being the unit's task and
    start state. A row draws what its sampler needs from the generator, in an order it fixes and reports.
    """
    entropy = [int(episode.noise_seed), int(episode.task_index), int(episode.state), int(query)]
    return np.random.default_rng(np.random.SeedSequence(entropy))


@dataclasses.dataclass
class Observation:
    """What the simulator renders and reports at one query. Nothing privileged crosses this line: no object
    poses, no segmentation, no depth, no simulator state. The one exception is named as such: `oracle`, the last
    field, which is None for every row but an instrument whose own definition asks for it."""

    #: camera name ("main", "wrist") -> uint8 [height, width, 3], exactly the array the simulator returns
    #: (`obs["agentview_image"]`, `obs["robot0_eye_in_hand_image"]`), not flipped, not resized. The releases
    #: rotate it by 180 degrees (`[::-1, ::-1]`); that is the adapter's step.
    images: Dict[str, np.ndarray]
    #: robot state, raw: "eef_pos" [3] (m, `obs["robot0_eef_pos"]`), "eef_quat" [4] (x, y, z, w;
    #: `obs["robot0_eef_quat"]` unchanged), "gripper_qpos" [2], "joint_pos" [7].
    state: Dict[str, np.ndarray]
    #: camera name -> {"intrinsics": [3, 3], "extrinsics": [4, 4]}, of both cameras at every query (the wrist
    #: camera moves with the hand). Exactly robosuite's `camera_utils.get_camera_intrinsic_matrix(sim, <camera>,
    #: height, width)` at the render size and `camera_utils.get_camera_extrinsic_matrix(sim, <camera>)`: the
    #: camera's pose in the world, with robosuite's camera-axis correction.
    cameras: Dict[str, Dict[str, np.ndarray]]
    #: the arm controller's constants a release may need to turn a target pose into an action:
    #: {"action_scale": [6]}, the simulator's `robots[0].controller.action_scale` (how far one unit of each of
    #: the six motion dimensions moves the target per step).
    controller: Dict[str, np.ndarray]
    instruction: str
    step: int  # policy steps executed so far in this episode (settling steps are not counted)
    query: int  # index of this query in the episode, from 0
    #: None for every row of the paper's tables. For a row whose report asks for it (`requires.oracle`, from its
    #: `row.json`): privileged simulator state by kind, as `evaluation/oracle.py` defines it, e.g.
    #: {"reach_target": {"target", "body", "position": [3]}}. The unit's config record says what was passed.
    oracle: Optional[Dict[str, Any]] = None

    def to_wire(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_wire(cls, message: Dict[str, Any]) -> "Observation":
        return cls(**{f.name: message[f.name] for f in dataclasses.fields(cls)})


@dataclasses.dataclass
class Actions:
    """What the model hands back at one query."""

    #: float [n, 7], n >= 1, in `env.step`'s convention (see ACTION_DIM), ready to execute as they are.
    actions: np.ndarray
    #: how many of them the evaluation executes before it queries again, 1 <= execute <= n.
    execute: int
    #: what a method wants kept beside the actions, small arrays only (PSG: "v", "v_g", "v_d"). The evaluation
    #: stores each array per query in the episode's trace under "query/<name>" and does not read it.
    record: Optional[Dict[str, np.ndarray]] = None

    def to_wire(self) -> Dict[str, Any]:
        return {"actions": np.asarray(self.actions), "execute": int(self.execute), "record": self.record or {}}

    @classmethod
    def from_wire(cls, message: Dict[str, Any]) -> "Actions":
        actions = cls(np.asarray(message["actions"], dtype=np.float64), int(message["execute"]), message.get("record") or {})
        actions.check()
        return actions

    def check(self) -> None:
        if self.actions.ndim != 2 or self.actions.shape[1] != ACTION_DIM or self.actions.shape[0] < 1:
            raise ValueError("actions must have shape [n, 7] with n >= 1, got %s" % (self.actions.shape,))
        if not 1 <= self.execute <= self.actions.shape[0]:
            raise ValueError("execute = %d is outside 1..%d" % (self.execute, self.actions.shape[0]))
        if not np.isfinite(self.actions).all():
            raise ValueError("the model returned non-finite actions")


#: The keys `Adapter.report()` must return. The evaluation writes the report into the config record unchanged.
REPORT_KEYS = ("model", "inference", "method", "noise_seed_controls", "environment", "requires")


class Adapter:
    """One row. A row's `adapter.py` defines `load(row: dict) -> Adapter`, where `row` is its `row.json`.

    The adapter loads the release's model with the release's own code and inference settings, turns the shared
    observation into what that code expects, and turns its output into LIBERO actions.
    """

    def prepare(self, instructions: List[str]) -> None:
        """Called once, before `report`, with the instruction of every task that is about to run.

        Optional. A method whose settings depend on the instruction's text (PSG's name) works them out here, so
        that `report` can state them and an instruction it cannot handle fails before any episode.
        """

    def report(self) -> Dict[str, Any]:
        """The row's side of the config record, read from the objects actually loaded, defaults included.

        model: {"name", "checkpoint": {"address", "revision", "sha256", "licence"}, "code": {"address", "revision"}}
            `sha256` is of the files the adapter loaded (one digest per file, or one tree digest and how it is
            computed), not a value copied from a card.
        inference: every setting in effect when the model is queried: sampling or decoding steps, chunk length,
            actions executed per query, the cameras used and what is done to each image, whether the state is
            read and in which form, the prompt template, normalisation, numerical precision, ensembling.
        method: {"name": "none"} for a released checkpoint run as released; otherwise {"name", "code":
            {"address", "revision"}, "settings": {...}} with every setting of the method.
        noise_seed_controls: one sentence: what `Episode.noise_seed` determines for this row ("nothing: the
            policy is deterministic" is an answer).
        environment: {"name", "python", "packages": {name: version}} for the packages that define the model.
        requires: {"camera_height", "camera_width"}: the render size the release was trained with. An instrument
            may add "oracle": [kinds] (`evaluation/oracle.py`); no row of the paper's tables does.
        """
        raise NotImplementedError

    def begin_episode(self, episode: Episode) -> None:
        """Forget the previous episode (action queues, ensembling buffers) and seed what the noise seed controls."""
        raise NotImplementedError

    def act(self, observation: Observation) -> Actions:
        raise NotImplementedError

    def close(self) -> None:
        """Called once when the evaluation has finished."""


def check_report(report: Dict[str, Any]) -> List[str]:
    """The problems of a report, as sentences; empty when it can go into a config record."""
    problems = ["report lacks %r" % key for key in REPORT_KEYS if key not in report]
    if problems:
        return problems
    for path in (("model", "name"), ("model", "checkpoint"), ("model", "code"), ("method", "name")):
        node: Any = report
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
        if node in (None, "", {}):
            problems.append("report has no %s" % ".".join(path))
    for key in ("camera_height", "camera_width"):
        if not isinstance(report["requires"].get(key), int):
            problems.append("report.requires.%s must be an integer" % key)
    if not str(report["noise_seed_controls"]).strip():
        problems.append("report.noise_seed_controls is empty")
    return problems
