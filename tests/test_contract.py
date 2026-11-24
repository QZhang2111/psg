"""The contract's own code: the wire, the interface's checks, the layout's names. No model, no simulator."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading

import numpy as np
import pytest

from evaluation import interface
from evaluation import layout
from evaluation import schema
from evaluation import wire
from evaluation.rows import serve


def observation() -> interface.Observation:
    rng = np.random.default_rng(0)
    return interface.Observation(
        images={name: rng.integers(0, 255, (256, 256, 3), dtype=np.uint8) for name in interface.CAMERAS},
        state={"eef_pos": rng.normal(size=3), "eef_quat": rng.normal(size=4), "gripper_qpos": rng.normal(size=2),
               "joint_pos": rng.normal(size=7)},
        cameras={name: {"intrinsics": np.eye(3), "extrinsics": np.eye(4)} for name in interface.CAMERAS},
        controller={"action_scale": np.array([0.05] * 3 + [0.5] * 3)},
        instruction="pick up the butter and place it in the basket", step=5, query=1,
    )  # fmt: skip


class FakeAdapter(interface.Adapter):
    def __init__(self) -> None:
        self.episodes: list = []

    def report(self) -> dict:
        return {"model": {"name": "fake", "checkpoint": {"address": "none"}, "code": {"address": "none"}},
                "inference": {"actions_executed_per_query": 5}, "method": {"name": "none"},
                "noise_seed_controls": "nothing: the policy is deterministic",
                "environment": {"name": "test", "python": sys.version.split()[0], "packages": {}},
                "requires": {"camera_height": 256, "camera_width": 256}}  # fmt: skip

    def begin_episode(self, episode: interface.Episode) -> None:
        self.episodes.append(episode)

    def act(self, observation: interface.Observation) -> interface.Actions:
        assert observation.images["main"].shape == (256, 256, 3) and observation.images["main"].dtype == np.uint8
        return interface.Actions(np.full((10, 7), observation.query, np.float32), 5, {"v": np.zeros((10, 10, 7), np.float32)})


def test_wire_carries_arrays_unchanged():
    a, b = socket.socketpair()
    message = {"kind": "act", "observation": observation().to_wire(), "nested": [1, 2.5, None, True, "π0.5"]}
    sender = threading.Thread(target=wire.send, args=(a, message))
    sender.start()
    got = wire.receive(b)
    sender.join()
    back = interface.Observation.from_wire(got["observation"])
    sent = observation()
    assert got["nested"] == message["nested"]
    for name in interface.CAMERAS:
        assert back.images[name].dtype == np.uint8 and np.array_equal(back.images[name], sent.images[name])
        assert np.array_equal(back.cameras[name]["extrinsics"], np.eye(4))
    assert all(np.array_equal(back.state[key], sent.state[key]) for key in sent.state)
    assert (back.instruction, back.step, back.query) == (sent.instruction, 5, 1)


def test_model_process_answers_the_four_requests():
    adapter = FakeAdapter()
    assert set(serve.answer(adapter, {"kind": "report"})["report"]) == set(interface.REPORT_KEYS)
    episode = interface.Episode("PP-O.fake.t6.s11.n7", "PP-O", 6, "task", "libero_object", 11, 7, "pick up the butter")
    serve.answer(adapter, {"kind": "begin_episode", "episode": episode.__dict__})
    assert adapter.episodes == [episode]
    actions = interface.Actions.from_wire(serve.answer(adapter, {"kind": "act", "observation": observation().to_wire()}))
    assert actions.actions.shape == (10, 7) and actions.execute == 5 and actions.record["v"].shape == (10, 10, 7)
    with pytest.raises(ValueError):
        serve.answer(adapter, {"kind": "something else"})


@pytest.mark.parametrize(
    ("actions", "execute"),
    [(np.zeros((10, 6)), 5), (np.zeros((4, 7)), 5), (np.zeros((4, 7)), 0), (np.full((4, 7), np.nan), 1)],
)
def test_actions_are_checked(actions, execute):
    with pytest.raises(ValueError):
        interface.Actions(actions, execute).check()


def test_an_incomplete_report_is_named():
    report = FakeAdapter().report()
    assert interface.check_report(report) == []
    del report["noise_seed_controls"]
    assert interface.check_report(report) == ["report lacks 'noise_seed_controls'"]
    report = FakeAdapter().report()
    report["requires"] = {"camera_height": 256}
    assert interface.check_report(report) == ["report.requires.camera_width must be an integer"]


def test_names_and_selections():
    assert schema.episode_id("PP-O", "pi05_psg", 6, 11, 7) == "PP-O.pi05_psg.t6.s11.n7"
    assert layout.parse_ranges("0-4,7") == [0, 1, 2, 3, 4, 7] and layout.parse_ranges(None) is None
    assert layout.parse_shard("1of2") == (1, 2) and layout.shard_name((1, 2)) == "1of2"
    with pytest.raises(SystemExit):
        layout.parse_shard("2of2")
    root = layout.output_root("/x")
    assert layout.unit_dir(root, "PP-O", "pi05", smoke=True) == root / "_smoke" / "PP-O" / "pi05"
    assert len(schema.EPISODE_COLUMNS) == len(set(schema.EPISODE_COLUMNS))


def test_shards_of_one_unit_may_differ_in_selection_only():
    a = {"unit": {"block": "PL", "row": "pi05"}, "rollout": {"max_steps": 280}, "selection": {"shard": "0of2"}}
    b = dict(a, selection={"shard": "1of2"})
    assert schema.record_differences(a, b) == []
    assert schema.record_differences(a, dict(b, rollout={"max_steps": 220})) == ["rollout"]


def test_every_row_declares_itself_and_its_environment():
    assert {"pi0", "pi05", "pi0_fast", "pi05_psg"} <= set(layout.rows())
    for row in layout.rows():
        definition = layout.load_row(row)
        assert (layout.ENVIRONMENTS / (definition["environment"] + ".json")).exists(), row


def test_the_entry_lists_rows_with_the_standard_library_only():
    out = subprocess.run([sys.executable, "-S", str(layout.REPO / "evaluation" / "run.py"), "--list"], capture_output=True, text=True)
    assert out.returncode == 0 and "pi05_psg" in out.stdout, out.stderr


def test_the_noise_of_a_query_depends_on_seed_episode_and_query_only():
    def episode(task, state, seed=7, name="PP-O.pi05.t0.s0.n7", row_block="PP-O"):
        return interface.Episode(name, row_block, task, "task", "libero_object", state, seed, "pick up the butter")

    def draw(*args, query=0, **kwargs):
        return interface.noise_generator(episode(*args, **kwargs), query).standard_normal((10, 32))

    base = draw(6, 11)
    assert np.array_equal(base, draw(6, 11, name="PP-O.pi05_psg.t6.s11.n7"))  # rows on one model are paired
    for other in (draw(6, 12), draw(5, 11), draw(6, 11, seed=17), draw(6, 11, query=1)):
        assert not np.array_equal(base, other)


CALLER = {"XLA_FLAGS": "--xla_force_host_platform_device_count=1", "HOME": "/h"}


def test_an_environments_declared_variables_reach_its_processes(tmp_path, monkeypatch):
    # The test's own (absent) machine file, not whatever `environments.local.json` lies in this checkout.
    monkeypatch.setenv("PSG_ENVIRONMENTS", str(tmp_path / "absent.json"))
    env = layout.process_environment("openpi", dict(CALLER))
    assert "--xla_gpu_deterministic_ops=true" in env["XLA_FLAGS"] and "--xla_force_host_platform_device_count=1" in env["XLA_FLAGS"]
    assert env["TF_CUDNN_DETERMINISTIC"] == "1" and env["HOME"] == "/h"
    assert env["PYTHONPATH"] == str(layout.REPO)  # the checkout alone: openpi is installed in its environment
    assert layout.process_environment("openpi", {"XLA_FLAGS": "--xla_gpu_deterministic_ops=true"})["XLA_FLAGS"] == "--xla_gpu_deterministic_ops=true"


DECLARED_FLAG, OTHER_FLAG = "--xla_gpu_deterministic_ops=true", "--xla_force_host_platform_device_count=2"
XLA_CASES = {  # what the caller gives -> what a process of the openpi environment gets
    "no flag": (None, DECLARED_FLAG),
    "only the declared flag": (DECLARED_FLAG, DECLARED_FLAG),
    "another flag alone": (OTHER_FLAG, OTHER_FLAG + " " + DECLARED_FLAG),
    "another flag with the declared one": (OTHER_FLAG + " " + DECLARED_FLAG, OTHER_FLAG + " " + DECLARED_FLAG),
    "the declared flag before another": (DECLARED_FLAG + " " + OTHER_FLAG, DECLARED_FLAG + " " + OTHER_FLAG),
    "the declared flag set to another value": (OTHER_FLAG + " --xla_gpu_deterministic_ops=false", OTHER_FLAG + " " + DECLARED_FLAG),
}


def test_a_callers_other_xla_flags_are_kept_and_the_declared_flag_stands_once(tmp_path, monkeypatch):
    monkeypatch.setenv("PSG_ENVIRONMENTS", str(tmp_path / "absent.json"))
    assert json.loads((layout.ENVIRONMENTS / "openpi.json").read_text())["variables"]["XLA_FLAGS"] == DECLARED_FLAG
    for case, (given, expected) in XLA_CASES.items():
        caller = {} if given is None else {"XLA_FLAGS": given}
        assert layout.process_environment("openpi", caller)["XLA_FLAGS"] == expected, case


def test_a_machines_variables_replace_the_callers_and_the_declared_ones_still_reach_the_process(tmp_path, monkeypatch):
    machine = tmp_path / "environments.local.json"
    monkeypatch.setenv("PSG_ENVIRONMENTS", str(machine))
    # The caller's variables, then the machine's for this environment, then the declaration's on top.
    machine.write_text(json.dumps({"openpi": {"python": "/p/python", "env": {
        "XLA_FLAGS": "--xla_gpu_strict_conv_algorithm_pick=false", "TF_CUDNN_DETERMINISTIC": "0", "MUJOCO_GL": "egl"}}}))
    env = layout.process_environment("openpi", dict(CALLER))
    assert env["XLA_FLAGS"] == "--xla_gpu_strict_conv_algorithm_pick=false --xla_gpu_deterministic_ops=true"
    assert env["TF_CUDNN_DETERMINISTIC"] == "1" and env["MUJOCO_GL"] == "egl" and env["HOME"] == "/h"
    # A machine file that names only the interpreter sets no variable.
    machine.write_text(json.dumps({"openpi": "/p/python"}))
    env = layout.process_environment("openpi", dict(CALLER))
    assert env["XLA_FLAGS"] == "--xla_force_host_platform_device_count=1 --xla_gpu_deterministic_ops=true"
    assert "MUJOCO_GL" not in env and env["TF_CUDNN_DETERMINISTIC"] == "1"


SIMULATOR_ENTRY = {"python": "/p/python", "env": {"MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": "0", "LIBERO_CONFIG_PATH": "/c"}}


def test_a_launchs_simulator_renders_on_the_launchs_card(tmp_path, monkeypatch):
    machine = tmp_path / "environments.local.json"
    monkeypatch.setenv("PSG_ENVIRONMENTS", str(machine))
    machine.write_text(json.dumps({"sim-libero-pro": SIMULATOR_ENTRY}))
    # One card named by the caller: the renderer follows it, whatever the machine's entry says.
    env = layout.simulator_environment("sim-libero-pro", {"CUDA_VISIBLE_DEVICES": "3", "HOME": "/h"})
    assert env["MUJOCO_EGL_DEVICE_ID"] == "3" and env["CUDA_VISIBLE_DEVICES"] == "3"
    assert env["MUJOCO_GL"] == "egl" and env["LIBERO_CONFIG_PATH"] == "/c" and env["HOME"] == "/h"
    # On the card the machine's entry names, the variables are what they were before the rule.
    before = layout.process_environment("sim-libero-pro", {"CUDA_VISIBLE_DEVICES": "0", "HOME": "/h"})
    assert layout.simulator_environment("sim-libero-pro", {"CUDA_VISIBLE_DEVICES": "0", "HOME": "/h"}) == before
    # No single card: the machine's entry holds, as before.
    for caller in ({"HOME": "/h"}, {"CUDA_VISIBLE_DEVICES": "0,1"}, {"CUDA_VISIBLE_DEVICES": "GPU-8f3a"}, {"CUDA_VISIBLE_DEVICES": ""}):
        assert layout.simulator_environment("sim-libero-pro", dict(caller)) == layout.process_environment("sim-libero-pro", dict(caller))
        assert layout.launch_card(caller) is None
    assert layout.launch_card({"CUDA_VISIBLE_DEVICES": " 5 "}) == "5"
    # The model's process is not touched by the rule.
    assert "MUJOCO_EGL_DEVICE_ID" not in layout.process_environment("openpi", {"CUDA_VISIBLE_DEVICES": "3"})


def test_a_machine_may_number_its_egl_devices_in_another_order(tmp_path, monkeypatch):
    machine = tmp_path / "environments.local.json"
    monkeypatch.setenv("PSG_ENVIRONMENTS", str(machine))
    machine.write_text(json.dumps({"sim-libero-pro": dict(SIMULATOR_ENTRY, egl_device_of_card={"0": 2, "1": "0", "2": 2})}))
    env = layout.simulator_environment("sim-libero-pro", {"CUDA_VISIBLE_DEVICES": "1"})
    assert env["MUJOCO_EGL_DEVICE_ID"] == "0" and "CUDA_VISIBLE_DEVICES" not in env  # robosuite refuses a device outside its cards
    env = layout.simulator_environment("sim-libero-pro", {"CUDA_VISIBLE_DEVICES": "2"})
    assert env["MUJOCO_EGL_DEVICE_ID"] == "2" and env["CUDA_VISIBLE_DEVICES"] == "2"
    env = layout.simulator_environment("sim-libero-pro", {"CUDA_VISIBLE_DEVICES": "7"})  # a card the table leaves out
    assert env["MUJOCO_EGL_DEVICE_ID"] == "7" and env["CUDA_VISIBLE_DEVICES"] == "7"
    machine.write_text(json.dumps({"sim-libero-pro": dict(SIMULATOR_ENTRY, egl_device_of_card={"1": "one"})}))
    with pytest.raises(SystemExit):
        layout.simulator_environment("sim-libero-pro", {"CUDA_VISIBLE_DEVICES": "1"})


def test_another_renderer_than_egl_is_left_alone(tmp_path, monkeypatch):
    machine = tmp_path / "environments.local.json"
    monkeypatch.setenv("PSG_ENVIRONMENTS", str(machine))
    machine.write_text(json.dumps({"sim-libero-pro": {"python": "/p/python", "env": {"MUJOCO_GL": "osmesa"}}}))
    env = layout.simulator_environment("sim-libero-pro", {"CUDA_VISIBLE_DEVICES": "3"})
    assert "MUJOCO_EGL_DEVICE_ID" not in env and env["CUDA_VISIBLE_DEVICES"] == "3"
