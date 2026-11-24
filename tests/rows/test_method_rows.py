"""The four rows pi05_cag, pi05_cag_va, pi05_rss and apt without weights, without a simulator.

Run on a CPU (Python 3.11.18, NumPy 1.26.4, SciPy 1.15.3, pytest 8.3.5), card hidden, on this repository's tree
before it was prepared for publication (the edits since are of wording, the licence and SmolVLA's cited LIBERO
number): 10 passed (`python -m pytest tests/rows/test_method_rows.py`). It needs NumPy, pytest and scipy, and
imports no model library: the releases' objects are replaced by stand-ins that record what they were given. CPU
only, under 1 GB.

What it checks: the request the openpi rows build (180-degree rotation, the 8-number state, the prompt), the
query noise (shape, determinism, independence of the row), what the CAG adapters pass to the authors' policy and
how many actions they execute, the RSS adapter's restatement of `CFGPolicy.infer` passing the noise, and APT's
frame, its per-step conversion and its call cadence (one model call per three queries).
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from evaluation import interface
from evaluation.rows.pi05_cag import openpi_libero as L


def observation(query=0, step=0, size=256, quat=(0.0, 0.0, 0.0, 1.0), pos=(0.1, -0.2, 0.9)):
    rng = np.random.default_rng(query)
    return interface.Observation(
        images={name: rng.integers(0, 255, (size, size, 3), dtype=np.uint8) for name in interface.CAMERAS},
        state={"eef_pos": np.array(pos), "eef_quat": np.array(quat), "gripper_qpos": np.array([0.03, -0.03]),
               "joint_pos": np.zeros(7)},
        cameras={name: {"intrinsics": np.eye(3) * (k + 1), "extrinsics": np.eye(4) * (k + 2)}
                 for k, name in enumerate(interface.CAMERAS)},
        controller={"action_scale": np.array([0.05] * 3 + [0.5] * 3)},
        instruction="pick up the cookie box and place it on the plate", step=step, query=query,
    )  # fmt: skip


def episode(seed=7):
    return interface.Episode(episode="CF-S.x.t0.s0.n7", block="CF-S", task_index=0, task="01-pick_up_the_cookie_box",
                             suite="libero_spatial", state=0, noise_seed=seed,
                             instruction="pick up the cookie box and place it on the plate")


def identity_resize(image, size):
    return image


# -- shared request ------------------------------------------------------------------------------------------

def test_request_rotates_and_builds_state():
    o = observation(quat=(0.0, 0.0, np.sin(0.25), np.cos(0.25)))
    r = L.libero_request(o, resize=identity_resize)
    assert np.array_equal(r["observation/image"], o.images["main"][::-1, ::-1])
    assert np.array_equal(r["observation/wrist_image"], o.images["wrist"][::-1, ::-1])
    s = r["observation/state"]
    assert s.shape == (8,)
    assert np.allclose(s[:3], o.state["eef_pos"]) and np.allclose(s[6:], o.state["gripper_qpos"])
    assert np.allclose(s[3:6], [0.0, 0.0, 0.5])  # a rotation of 0.5 rad about z
    assert r["prompt"] == o.instruction


def test_identity_quaternion_is_zero_axis_angle():
    assert np.array_equal(L.quat2axisangle([0, 0, 0, 1]), np.zeros(3))


def test_query_noise():
    import dataclasses

    e = episode(seed=7)
    a, b = L.query_noise(e, 3), L.query_noise(e, 3)
    assert a.shape == (10, 32) and a.dtype == np.float32 and np.array_equal(a, b)
    assert np.array_equal(a, interface.noise_generator(e, 3).standard_normal((10, 32)).astype(np.float32))
    assert not np.array_equal(a, L.query_noise(e, 4)) and not np.array_equal(a, L.query_noise(episode(seed=17), 3))
    assert not np.array_equal(a, L.query_noise(dataclasses.replace(e, state=1), 3))  # every episode its own noise


def test_tree_digest_is_order_free(tmp_path):
    (tmp_path / "a").write_bytes(b"1"); (tmp_path / "d").mkdir(); (tmp_path / "d" / "b").write_bytes(b"2")
    d = L.tree_sha256(tmp_path)
    assert d["files"] == "2" and len(d["tree"]) == 64


# -- CAG ------------------------------------------------------------------------------------------------------

class FakeCagPolicy:
    """Stands for CounterfactualActionGuidedPolicy: records the request and the noise."""

    def __init__(self):
        self.calls = []

    def infer(self, obs, *, noise=None):
        self.calls.append((obs, noise))
        return {"actions": np.full((10, 7), float(len(self.calls)))}


@pytest.mark.parametrize("module", ["evaluation.rows.pi05_cag.adapter", "evaluation.rows.pi05_cag_va.adapter"])
def test_cag_passes_request_and_query_noise(module, monkeypatch):
    import importlib

    m = importlib.import_module(module)
    monkeypatch.setattr(L, "libero_request", lambda o, resize=None: {"prompt": o.instruction, "q": o.query})
    policy = FakeCagPolicy()
    cls = m.CagTrainingFree if hasattr(m, "CagTrainingFree") else m.CagVisionAction
    adapter = cls({"environment": "openpi-cf"}, policy=policy)
    adapter.begin_episode(episode(seed=7))
    out = adapter.act(observation(query=2))
    assert out.actions.shape == (10, 7) and out.execute == 5
    request, noise = policy.calls[0]
    assert request == {"prompt": "pick up the cookie box and place it on the plate", "q": 2}
    assert np.array_equal(noise, L.query_noise(episode(seed=7), 2))


# -- RSS ------------------------------------------------------------------------------------------------------

def test_rss_infer_passes_noise(monkeypatch):
    jax = pytest.importorskip("jax")  # the restatement of CFGPolicy.infer uses jax.tree; skip where JAX is absent
    from evaluation.rows.pi05_rss import adapter as rss

    seen = {}

    def sample(rng, observation, noise=None, **kwargs):
        seen["noise"], seen["kwargs"] = np.asarray(noise), kwargs
        return np.zeros((1, 10, 32))

    policy = types.SimpleNamespace(
        _input_transform=lambda d: {"state": np.zeros(8), "x": np.ones(3)},
        _output_transform=lambda d: {"actions": np.asarray(d["actions"])[:, :7]},
        _rng=jax.random.key(0), _sample_actions=sample, _sample_kwargs={"num_steps": 10, "guidance_scale": 1.5},
    )
    a = rss.Rss({"environment": "rss"}, policy=policy, observation_cls=types.SimpleNamespace(from_dict=lambda d: d))
    a.begin_episode(episode(seed=7))
    monkeypatch.setattr(L, "libero_request", lambda o, resize=None: {"prompt": o.instruction})
    out = a.act(observation(query=4))
    assert out.actions.shape == (10, 7) and out.execute == 5
    assert np.array_equal(seen["noise"][0], L.query_noise(episode(seed=7), 4))
    assert seen["kwargs"] == {"num_steps": 10, "guidance_scale": 1.5}


# -- APT ------------------------------------------------------------------------------------------------------

class FakePlanner:
    def __init__(self, targets):
        self.targets, self.frames, self.calls = targets, [], 0
        self.config = types.SimpleNamespace(sample_state_gaps=1)

    def reset(self): self.frames.clear()
    def set_config(self, name): self.config_name = name
    def set_prompt(self, text): self.prompt = text
    def add_obs_frame(self, frame): self.frames.append(frame)

    def get_action(self, sample_num=1):
        self.calls += 1
        poses, grippers = self.targets
        return poses[None, :, None], grippers[None, :, None], np.arange(len(poses)) + 1.0, None


def test_apt_frame():
    pytest.importorskip("scipy")
    from evaluation.rows.apt import adapter as apt

    o = observation(step=12)
    f = apt.apt_frame(o)
    assert np.allclose(f["ee_pose"][:3, 3], o.state["eef_pos"]) and np.allclose(f["ee_pose"][:3, :3], np.eye(3))
    assert f["gripper"] == pytest.approx(0.03 / 0.04) and f["timestamp"] == 12
    assert np.array_equal(f["agentview"]["data"]["color"], o.images["main"][::-1])  # vertical flip only
    assert f["eye_in_hand"]["camera"]["K"] == (np.eye(3) * 2).flatten().tolist()
    assert np.array_equal(f["agentview"]["data"]["wcT"], np.eye(4) * 2)
    with pytest.raises(ValueError):
        apt.apt_frame(observation(size=224))


def test_apt_conversion_and_cadence():
    pytest.importorskip("scipy")
    from evaluation.rows.apt import adapter as apt

    poses = np.stack([np.eye(4) for _ in range(32)])
    poses[:, :3, 3] = [0.1, -0.2, 0.9]
    poses[0, 0, 3] += 0.01  # the first target is 1 cm along x
    grippers = np.full(32, 0.9); grippers[1] = 0.1  # open, then closed
    planner = FakePlanner((poses, grippers))
    seeds = []
    a = apt.Apt({"environment": "apt"}, planner=planner, seeder=seeds.append)
    a.begin_episode(episode(seed=7))
    assert planner.config_name == "Libero" and planner.prompt.startswith("pick up the cookie box")
    first = a.act(observation(query=0, step=0))
    assert first.execute == 1 and first.actions.shape == (1, 7)
    assert np.allclose(first.actions[0, :6], [0.01 / 0.05, 0, 0, 0, 0, 0]) and first.actions[0, 6] == -1.0
    assert a.act(observation(query=1, step=1)).actions[0, 6] == 1.0
    a.act(observation(query=2, step=2))
    assert planner.calls == 1 and len(planner.frames) == 1
    a.act(observation(query=3, step=3))
    assert planner.calls == 2 and seeds == [apt.call_seed(episode(seed=7), 0), apt.call_seed(episode(seed=7), 3)]
