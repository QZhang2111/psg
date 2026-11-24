"""OpenVLA-OFT adapter, its glue on a synthetic observation, without the release or the weights.

Where it runs: any CPU machine with Python 3.8+,
NumPy and pytest; under a second of CPU and a few MB. The release's functions are replaced by recorders, so this
checks only what the adapter itself does: the 180-degree rotation before the release's resize, the state's
order and the axis-angle, the un-normalisation key per suite, the gripper processing order, `execute`.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation.interface import Episode, Observation
from evaluation.rows.openvla_oft import adapter as A


def observation(instruction="pick up the butter and place it in the basket"):
    rng = np.random.default_rng(0)
    return Observation(
        images={"main": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
                "wrist": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)},
        state={"eef_pos": np.array([0.1, -0.2, 1.0]), "eef_quat": np.array([0.0, 0.0, math.sin(0.25), math.cos(0.25)]),
               "gripper_qpos": np.array([0.03, -0.03]), "joint_pos": np.zeros(7)},
        cameras={}, controller={}, instruction=instruction, step=0, query=0)


def make(chunk=None):
    calls = {}
    chunk = chunk if chunk is not None else [np.array([0.1, 0, 0, 0, 0, 0, 1.0])] * 4 + [np.array([0, 0, 0, 0, 0, 0, 0.0])] * 4

    def resize(img, size):
        calls.setdefault("resize", []).append(img.copy())
        return img[:size, :size]

    def get_vla_action(cfg, vla, processor, obs, task, **kw):
        calls["obs"], calls["task"], calls["key"] = obs, task, cfg.unnorm_key
        return chunk

    release = {"resize_image_for_policy": resize, "get_vla_action": get_vla_action,
               "normalize_gripper_action": lambda a, binarize=True: np.concatenate([a[:-1], [np.sign(2 * a[-1] - 1)]]),
               "invert_gripper_action": lambda a: np.concatenate([a[:-1], [-a[-1]]]),
               "set_seed_everywhere": lambda s: calls.setdefault("seed", s)}
    vla = SimpleNamespace(norm_stats={k + "_no_noops": {} for k in A.SUITES})
    row = {"name": "OpenVLA-OFT", "environment": "openvla-oft",
           "checkpoint": {"hub_id": "x", "address": "x", "licence": "MIT"}, "code": {"address": "x", "revision": "x"}}
    return A.OpenVlaOft(row, release, vla, None, None, None, loaded={}), calls


def episode(suite="libero_object"):
    return Episode(episode="PP-O.openvla_oft.t0.s0.n7", block="PP-O", task_index=0, task="t", state=0,
                   noise_seed=7, instruction="i", suite=suite)


def test_rotation_happens_before_the_release_resize():
    a, calls = make()
    a.begin_episode(episode())
    o = observation()
    a.act(o)
    assert np.array_equal(calls["resize"][0], o.images["main"][::-1, ::-1])
    assert np.array_equal(calls["resize"][1], o.images["wrist"][::-1, ::-1])


def test_state_order_and_axis_angle():
    a, calls = make()
    a.begin_episode(episode())
    a.act(observation())
    s = calls["obs"]["state"]
    assert s.shape == (8,)
    np.testing.assert_allclose(s[:3], [0.1, -0.2, 1.0])
    np.testing.assert_allclose(s[3:6], [0, 0, 0.5])  # a rotation of 0.5 rad about z
    np.testing.assert_allclose(s[6:], [0.03, -0.03])


def test_quaternion_of_the_observation_is_not_changed():
    o = observation()
    o.state["eef_quat"] = np.array([0.0, 0.0, 0.0, 1.0000001])
    A.quat2axisangle(o.state["eef_quat"])
    assert o.state["eef_quat"][3] == 1.0000001


def test_unnorm_key_per_suite_and_refusal_without_a_suite():
    a, calls = make()
    for suite in A.SUITES:
        a.begin_episode(episode(suite))
        a.act(observation())
        assert calls["key"] == suite + "_no_noops"
    with pytest.raises(ValueError):
        a.begin_episode(episode("libero_90"))


def test_gripper_open_close_and_execute_all_eight():
    a, _ = make()
    a.begin_episode(episode())
    out = a.act(observation())
    assert out.actions.shape == (8, 7) and out.execute == 8
    assert (out.actions[:4, 6] == -1).all()  # model 1 = open -> env -1 = open
    assert (out.actions[4:, 6] == 1).all()  # model 0 = close -> env +1 = close
    out.check()


def test_seed_and_instruction_pass_through():
    a, calls = make()
    a.begin_episode(episode())
    a.act(observation("put the bowl on the plate"))
    assert calls["seed"] == 7 and calls["task"] == "put the bowl on the plate"
