"""SmolVLA adapter, its glue on a synthetic observation, without lerobot or the weights.

Where it runs: any CPU machine with Python 3.8+, NumPy and pytest; under a second, a few MB. The release's
functions are replaced by recorders: this checks the state's order and the float32 axis-angle, that the images reach the release's canonicaliser unrotated (it
rotates them) with the right keys, the task string, the noise by the shared rule, and `execute`: the number of
actions executed per query is the row's stated setting (10 in `smolvla_exec10`; 50, the release's default, where
a test sets it).
"""

import json
import math
from pathlib import Path

import numpy as np

from evaluation.interface import Episode, Observation
from evaluation.rows.smolvla_exec10 import adapter as A

ROWS = Path(A.__file__).resolve().parents[1]


def row_file(row):
    return json.loads((ROWS / row / "row.json").read_text())


def row_at(executed):
    """The row smolvla_exec10 with another number of executed actions per query."""
    return dict(row_file("smolvla_exec10"), executed_per_query=executed)


def observation(query=0):
    rng = np.random.default_rng(0)
    return Observation(
        images={"main": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
                "wrist": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)},
        state={"eef_pos": np.array([0.1, -0.2, 1.0]), "eef_quat": np.array([0.0, 0.0, math.sin(0.25), math.cos(0.25)]),
               "gripper_qpos": np.array([0.03, -0.03]), "joint_pos": np.zeros(7)},
        cameras={}, controller={}, instruction="put the bowl on the plate", step=0, query=query)


def make(executed=50):
    calls = {}
    release = {
        "reset": lambda: calls.setdefault("reset", 0),
        "canonical_observation": lambda raw, flip_images: calls.setdefault("raw", (raw, flip_images)) and {"observation.state": raw["agent_pos"]},
        "to_tensors": lambda b: dict(b),
        "preprocessor": lambda b: calls.setdefault("batch", b) or b,
        "noise": lambda array: calls.setdefault("noise", []).append(array),
        "predict_action_chunk": lambda batch, noise: np.zeros((1, 50, 7)),
        "postprocess": lambda chunk, i: np.full(7, float(i)),
    }
    loaded = {"n_action_steps": 50, "chunk_size": 50, "max_action_dim": 32}
    return A.SmolVla(row_at(executed), release, loaded), calls


def episode():
    return Episode(episode="L-O.smolvla_exec10.t0.s0.n7", block="L-O", task_index=0, task="t", state=0, noise_seed=7,
                   instruction="i", suite="libero_object")


def test_state_and_images_reach_the_release_unrotated():
    a, calls = make()
    a.begin_episode(episode())
    o = observation()
    a.act(o)
    raw, flip = calls["raw"]
    assert flip is True
    assert raw["pixels"]["image"] is o.images["main"] and raw["pixels"]["image2"] is o.images["wrist"]
    s = raw["agent_pos"]
    assert s.dtype == np.float32 and s.shape == (8,)
    np.testing.assert_allclose(s[3:6], [0, 0, 0.5], atol=1e-6)


def test_task_string_and_execute():
    a, calls = make()
    a.begin_episode(episode())
    out = a.act(observation())
    assert calls["batch"]["task"] == ["put the bowl on the plate"]
    assert out.actions.shape == (50, 7) and out.execute == 50
    assert out.actions[3, 0] == 3.0  # postprocessed in order


def test_noise_is_the_shared_rule_per_episode_and_query():
    from evaluation.interface import noise_generator
    a, calls = make()
    a.begin_episode(episode())
    a.act(observation(0))
    a.act(observation(1))
    b, calls_b = make()
    b.begin_episode(episode())
    b.act(observation(1))
    n0, n1 = calls["noise"]
    assert n0.shape == (1, 50, 32) and n0.dtype == np.float32
    assert not np.array_equal(n0, n1)
    np.testing.assert_array_equal(n1, calls_b["noise"][0])  # same episode, same query: same noise
    np.testing.assert_array_equal(n0, noise_generator(episode(), 0).standard_normal((1, 50, 32)).astype(np.float32))


def test_axisangle_identity_is_zero():
    np.testing.assert_array_equal(A._axisangle(np.array([0, 0, 0, 1.0])), np.zeros(3, np.float32))


# ------------------------------------------------------------------- executed actions per query, a row setting


def full_loaded():
    return {"n_action_steps": 50, "chunk_size": 50, "max_action_dim": 32, "num_steps": 10, "resize": (512, 512),
            "max_state_dim": 32, "tokenizer_max_length": 48, "vlm_id": "vlm", "vlm_revision": "r", "dtype": "torch.float32",
            "checkpoint_revision": "c", "checkpoint_sha256": {"files": {}}, "packages": {}}


def test_the_new_row_executes_the_first_ten_of_each_chunk():
    a, calls = make(10)
    a.begin_episode(episode())
    out = a.act(observation())
    assert out.actions.shape == (10, 7) and out.execute == 10
    np.testing.assert_array_equal(out.actions[:, 0], np.arange(10.0))  # the first ten, in order
    assert calls["noise"][0].shape == (1, 50, 32)  # the chunk and its noise are the release's 50
    out.check()


def test_the_record_states_the_setting_and_the_release_default_beside_it():
    old = A.SmolVla(row_at(50), {}, full_loaded()).report()
    new = A.SmolVla(row_file("smolvla_exec10"), {}, full_loaded()).report()
    assert old["inference"]["executed_per_query"] == 50
    assert "executed_per_query_release_default" not in old["inference"]  # at the release's default, nothing beside it
    assert new["inference"]["executed_per_query"] == 10
    assert new["inference"]["executed_per_query_release_default"] == 50
    assert "first 10 actions of each chunk of 50" in new["inference"]["executed_per_query_note"]
    differing = {key for key in set(old["inference"]) | set(new["inference"]) if old["inference"].get(key) != new["inference"].get(key)}
    assert differing == {"executed_per_query", "executed_per_query_release_default", "executed_per_query_note"}
    for key in ("model", "method", "noise_seed_controls", "environment", "requires"):
        assert old[key] == new[key], key


def test_a_row_that_does_not_state_the_setting_is_refused():
    loaded = {"n_action_steps": 50, "chunk_size": 50, "max_action_dim": 32}
    for stated in (None, "10", 10.0, True, 0, 51):
        row = row_at(stated)
        if stated is None:
            del row["executed_per_query"]
        try:
            A.SmolVla(row, {}, loaded)
        except SystemExit as refusal:
            assert "executed_per_query" in str(refusal)
        else:
            raise AssertionError("executed_per_query = %r was accepted" % (stated,))
