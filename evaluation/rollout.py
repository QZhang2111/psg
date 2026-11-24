"""One episode, the same for every row: reset to the start state, settle, let the model act, stop.

    1. the scene is reset to the episode's start state (the environment seed is set before every reset, so an
       episode does not depend on the episodes run before it);
    2. `settle_steps` steps with the settling action (hold still, gripper open);
    3. until the step limit: when no action is left, the model is queried with the current observation and
       returns a chunk and how many of it to execute; one action is executed; the scene says whether the
       episode ends and whether it succeeded.

The scene is anything with the methods of `simulator.Scene`; the model is anything with `begin_episode` and
`act` (`ModelClient` below, over the wire). Python 3.8 and later, NumPy only.
"""

from __future__ import annotations

import dataclasses
import socket
from typing import Any, Dict, Optional

import numpy as np

from evaluation import interface
from evaluation import trace as _trace
from evaluation import wire

SETTLE_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]  # hold still, gripper open


class ModelClient:
    """The row's model, in its own process (`evaluation/rows/serve.py`)."""

    def __init__(self, port: int) -> None:
        self._connection = socket.create_connection(("127.0.0.1", int(port)))
        self._connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def report(self, instructions: Any = ()) -> Dict[str, Any]:
        """The adapter's report; it is first told the instructions of the block's tasks (`Adapter.prepare`)."""
        return wire.call(self._connection, "report", instructions=list(instructions))["report"]

    def begin_episode(self, episode: interface.Episode) -> None:
        wire.call(self._connection, "begin_episode", episode=dataclasses.asdict(episode))

    def act(self, observation: interface.Observation) -> interface.Actions:
        return interface.Actions.from_wire(wire.call(self._connection, "act", observation=observation.to_wire()))

    def close(self) -> None:
        try:
            wire.call(self._connection, "close")
        finally:
            self._connection.close()


def run_episode(
    scene: Any,
    model: Any,
    episode: interface.Episode,
    *,
    settle_steps: int,
    max_steps: int,
    recorder: _trace.TraceWriter,
    video: Optional[Any] = None,
    oracle: Optional[Any] = None,
) -> Dict[str, Any]:
    """Runs one episode on a scene already reset to its start state; returns the outcome columns.

    `recorder` receives every frame and query; `video`, when given, every main-camera image (`video.add`).
    `oracle`: the kinds of privileged state the row's definition asks for (`evaluation/oracle.py`);
    None or empty for every row of the paper's tables, and then the scene is not asked for any.
    """

    def keep(phase: int, action: Any) -> Any:
        """Records the frame the scene is at; returns (the episode ends now, it has succeeded)."""
        status = scene.status()
        recorder.frame(phase, action, scene.frame(), status[1])
        if video is not None:
            video.add(scene.main_image())
        return status

    keep(_trace.START, None)
    for _ in range(settle_steps):
        scene.step(SETTLE_ACTION)
        keep(_trace.SETTLE, SETTLE_ACTION)

    model.begin_episode(episode)
    queue: list = []
    steps, queries, success_step, stopped = 0, 0, None, "limit"
    while steps < max_steps:
        if not queue:
            observation = interface.Observation(
                instruction=episode.instruction, step=steps, query=queries,
                oracle=scene.oracle(oracle) if oracle else None, **scene.observation(),
            )  # fmt: skip
            actions = model.act(observation)
            recorder.query(actions.execute, actions.actions, actions.record or {})
            queue = [np.asarray(action, np.float64) for action in actions.actions[: actions.execute]]
            queries += 1
        action = queue.pop(0)
        scene.step(action)
        steps += 1
        ended, succeeded = keep(_trace.POLICY, action)
        if succeeded and success_step is None:
            success_step = steps
        if ended:
            stopped = "predicate"
            break
    return {
        "success": int(success_step is not None),
        "success_step": "" if success_step is None else success_step,
        "steps": steps,
        "queries": queries,
        "stopped": stopped,
    }
