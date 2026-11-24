"""The episode loop against a hand-made scene and a model over the real wire. Standard library and NumPy."""

from __future__ import annotations

import socket
import threading

import numpy as np

from evaluation import interface
from evaluation import reach
from evaluation import rollout
from evaluation import trace as _trace
from evaluation import wire
from evaluation.rows import serve

import synthetic as s
from test_contract import FakeAdapter

EPISODE = interface.Episode("PP-O.fake.t0.s3.n7", "PP-O", 0, "task", "libero_object", 3, 7, "pick up the alphabet soup")


class Model:
    """Moves the hand toward the soup and closes on the third action of every chunk; 10 actions, 5 executed."""

    def __init__(self) -> None:
        self.episodes, self.observations = [], []

    def begin_episode(self, episode: interface.Episode) -> None:
        self.episodes.append(episode)

    def act(self, observation: interface.Observation) -> interface.Actions:
        self.observations.append(observation)
        chunk = np.zeros((10, 7))
        chunk[:, 0] = 1.0  # +x: 10 cm per step in the synthetic scene
        chunk[:, 6] = [-1, -1, 1, 1, 1, 1, 1, 1, 1, 1]
        return interface.Actions(chunk, 5, {"v": np.full((10, 10, 7), observation.query, np.float32)})


def run(scene, model, max_steps=12, settle=10):
    recorder = _trace.TraceWriter({"episode": EPISODE.episode}, scene.tables())
    scene.reset(EPISODE.state)
    outcome = rollout.run_episode(scene, model, EPISODE, settle_steps=settle, max_steps=max_steps, recorder=recorder)
    return outcome, recorder.arrays()


def test_settle_then_act_until_the_limit():
    scene, model = s.Scene(), Model()
    outcome, trace = run(scene, model)
    assert outcome == {"success": 0, "success_step": "", "steps": 12, "queries": 3, "stopped": "limit"}
    assert trace["phase"].tolist() == [0] + [1] * 10 + [2] * 12
    assert all(action.tolist() == rollout.SETTLE_ACTION for action in scene.actions[:10])
    assert trace["query_frame"].tolist() == [10, 15, 20] and trace["query_execute"].tolist() == [5, 5, 5]
    assert trace["query/v"].shape == (3, 10, 10, 7) and trace["query/v"][:, 0, 0, 0].tolist() == [0.0, 1.0, 2.0]
    assert model.episodes == [EPISODE]
    assert [(o.step, o.query, o.instruction) for o in model.observations] == [
        (0, 0, EPISODE.instruction), (5, 1, EPISODE.instruction), (10, 2, EPISODE.instruction),
    ]  # fmt: skip
    assert model.observations[0].controller["action_scale"].shape == (6,)


def test_the_episode_stops_when_the_predicate_holds():
    outcome, trace = run(s.Scene(succeeds_at=7), Model())
    assert outcome == {"success": 1, "success_step": 7, "steps": 7, "queries": 2, "stopped": "predicate"}
    assert trace["predicate"].tolist() == [False] * 17 + [True]


def test_an_episode_can_end_without_success():
    outcome, _ = run(s.Scene(ends_at=4), Model())
    assert outcome == {"success": 0, "success_step": "", "steps": 4, "queries": 1, "stopped": "predicate"}


def test_the_trace_of_a_rollout_can_be_labelled():
    # The hand starts at x = 0 and moves 10 cm per step; it closes at policy step 2, at x = 0.3, nearest the soup.
    _, trace = run(s.Scene(), Model())
    row = reach.label(trace, {"rule": "first_close", "target": ["alphabet_soup_1"]})
    assert (row["first_close_step"], row["first_close_entity"], row["reach"]) == (2, "alphabet_soup_1", 1)


def test_the_model_process_over_the_wire():
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    closed = []

    class Adapter(FakeAdapter):
        def close(self) -> None:
            closed.append(True)

    adapter = Adapter()
    thread = threading.Thread(target=serve.serve, args=(adapter, listener), daemon=True)
    thread.start()
    port = listener.getsockname()[1]

    client = rollout.ModelClient(port)
    assert client.report(["pick up the alphabet soup"])["model"]["name"] == "fake"
    outcome, trace = run(s.Scene(), client, max_steps=10, settle=2)
    assert outcome["steps"] == 10 and outcome["queries"] == 2
    assert trace["query_actions"].shape == (2, 10, 7) and trace["query/v"].shape == (2, 10, 10, 7)
    assert adapter.episodes == [EPISODE]
    client.close()

    # A second evaluation process (the next block of the same launch) finds the model still loaded.
    second = rollout.ModelClient(port)
    assert second.report([])["model"]["name"] == "fake"
    outcome, _ = run(s.Scene(), second, max_steps=5, settle=0)
    assert outcome["queries"] == 1 and adapter.episodes == [EPISODE, EPISODE] and closed == []
    second.close()

    # The entry's last word: the adapter is closed and the model process ends.
    with socket.create_connection(("127.0.0.1", port)) as connection:
        wire.call(connection, "shutdown")
    thread.join(timeout=5)
    listener.close()
    assert not thread.is_alive() and closed == [True]
