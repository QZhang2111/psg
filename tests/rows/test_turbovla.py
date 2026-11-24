"""TurboVLA adapter, its glue on a synthetic observation, without the release or the weights.

Where it runs: any CPU machine with Python 3.8+,
NumPy and pytest; under a second, a few MB. Checks: the release gets the images unrotated under LIBERO's keys
(it rotates them), the raw state, the instruction; 256 px only; `execute` is the chunk's length; the seed.
"""

import numpy as np
import pytest

from evaluation.interface import Episode, Observation
from evaluation.rows.turbovla import adapter as A


def observation(size=256):
    rng = np.random.default_rng(0)
    return Observation(
        images={"main": rng.integers(0, 255, (size, size, 3), dtype=np.uint8),
                "wrist": rng.integers(0, 255, (size, size, 3), dtype=np.uint8)},
        state={"eef_pos": np.array([0.1, -0.2, 1.0]), "eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
               "gripper_qpos": np.array([0.03, -0.03]), "joint_pos": np.zeros(7)},
        cameras={}, controller={}, instruction="turn on the stove", step=0, query=0)


def make():
    calls = {}

    def predict(obs, instruction):
        calls["obs"], calls["instruction"] = obs, instruction
        return np.ones((12, 7), np.float32)

    release = {"set_seed_everywhere": lambda s: calls.setdefault("seed", s), "predict": predict}
    return A.TurboVla({"name": "TurboVLA"}, release, {}), calls


def episode():
    return Episode(episode="L-G.turbovla.t0.s0.n7", block="L-G", task_index=0, task="t", state=0, noise_seed=7,
                   instruction="i", suite="libero_goal")


def test_release_gets_raw_images_state_and_instruction():
    a, calls = make()
    a.begin_episode(episode())
    o = observation()
    out = a.act(o)
    assert calls["obs"]["agentview_image"] is o.images["main"]
    assert calls["obs"]["robot0_eye_in_hand_image"] is o.images["wrist"]
    np.testing.assert_array_equal(calls["obs"]["robot0_eef_quat"], [0, 0, 0, 1])
    assert calls["instruction"] == "turn on the stove" and calls["seed"] == 7
    assert out.actions.shape == (12, 7) and out.execute == 12


def test_only_256_px():
    a, _ = make()
    a.begin_episode(episode())
    with pytest.raises(ValueError):
        a.act(observation(224))
