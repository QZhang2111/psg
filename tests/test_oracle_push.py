"""The oracle push's definition, the oracle channel through the rollout and the wire, and what collects a unit
run on named tasks. Standard library and NumPy; the sampler itself is `test_oracle_push_sampler.py` (host).

Where it runs: any machine with Python 3.8+, NumPy and pytest, from the checkout: `python -m pytest tests/test_oracle_push.py`.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

from evaluation import collect
from evaluation import interface
from evaluation import layout
from evaluation import oracle
from evaluation import rollout
from evaluation import trace as _trace
from evaluation import wire
from instruments import oracle_push

import synthetic as s
import test_collect
from test_contract import observation
from test_rollout import EPISODE, Model

ROW = "pi05_oracle_push"
IMPORTS_INSTRUMENTS = re.compile(r"^\s*(from|import)\s+instruments\b", re.MULTILINE)


def test_the_shift_points_at_the_object_and_falls_to_zero_inside_the_guard():
    np.testing.assert_allclose(oracle_push.shift([0.3, -0.4], 0.2, 0.05), [0.12, -0.16])
    np.testing.assert_allclose(oracle_push.shift([0.0, 0.025], 0.2, 0.05), [0.0, 0.1])  # half the radius, half the size
    np.testing.assert_allclose(oracle_push.shift([0.0, 0.05, 9.0], 0.2, 0.05), [0.0, 0.2])  # only x and y count
    assert not oracle_push.shift([0.0, 0.0], 0.2, 0.05).any()


def test_the_velocity_has_the_sign_that_moves_the_action_toward_the_object():
    v_o = oracle_push.velocity([0.12, -0.16], [0.8, 0.4], 32)
    assert v_o.shape == (32,) and v_o.dtype == np.float32 and not v_o[2:].any()
    # ten Euler steps with dt = -1/10 move the sample by -v_o; the output scale turns that into the action's shift
    np.testing.assert_allclose(-v_o[:2] * np.asarray([0.8, 0.4]), [0.12, -0.16], rtol=1e-6)



def test_the_rows_settings_are_checked():
    row = layout.load_row(ROW)
    with pytest.raises(SystemExit):  # a definition that names no size does not load
        oracle_push.settings(dict(row, push={"size": None, "guard_radius_m": 0.05}))
    assert oracle_push.settings(dict(row, push={"size": 0.3, "guard_radius_m": 0.05})) == {"size": 0.3, "guard_radius_m": 0.05}
    for push in ({"size": -1.0, "guard_radius_m": 0.05}, {"size": 0.3, "guard_radius_m": 0.0}, {"size": True, "guard_radius_m": 0.05}):
        with pytest.raises(SystemExit):
            oracle_push.settings(dict(row, push=push))
    with pytest.raises(SystemExit):
        oracle_push.settings(dict(row, push={"size": 0.3, "guard_radius_m": 0.05}, oracle=[]))


def test_only_the_instrument_rows_ask_for_an_oracle_and_nothing_of_the_tables_imports_an_instrument():
    asking = [row for row in layout.rows() if layout.load_row(row).get("oracle")]
    assert asking == [ROW]
    definition = layout.load_row(ROW)
    assert definition["oracle"] == [oracle_push.KIND] and definition["method"] == oracle_push.METHOD
    assert definition["checkpoint"] == layout.load_row("pi05_psg")["checkpoint"]
    for path in sorted((layout.REPO / "psg").glob("*.py")):  # the method stands apart
        assert not IMPORTS_INSTRUMENTS.search(path.read_text()) and "import oracle" not in path.read_text(), path
    instrument_rows = {row for row in layout.rows() if "instrument" in layout.load_row(row)["kind"]}
    assert ROW in instrument_rows
    for path in sorted(layout.ROWS.glob("*/*.py")) + sorted(layout.ROWS.glob("*.py")):
        if path.parent.name not in instrument_rows:
            assert not IMPORTS_INSTRUMENTS.search(path.read_text()), path


def test_the_rollout_passes_the_oracle_only_when_asked():
    def run(oracle):
        scene, model = s.Scene(), Model()
        scene.reset(EPISODE.state)
        recorder = _trace.TraceWriter({"episode": EPISODE.episode}, scene.tables())
        rollout.run_episode(scene, model, EPISODE, settle_steps=2, max_steps=6, recorder=recorder, oracle=oracle)
        return model.observations

    assert all(seen.oracle is None for seen in run(None)) and all(seen.oracle is None for seen in run([]))
    told = run(["reach_target_position"])
    assert len(told) == 2
    for seen in told:
        assert seen.oracle["reach_target"]["body"] == "alphabet_soup_1_main"
        np.testing.assert_array_equal(seen.oracle["reach_target"]["position"], s.PLACES[s.SOUP])


def test_an_observation_crosses_the_wire_with_and_without_an_oracle():
    plain = observation()
    assert plain.oracle is None
    arrays: list = []
    assert interface.Observation.from_wire(wire._decode(wire._encode(plain.to_wire(), arrays), arrays)).oracle is None
    told = interface.Observation(
        **dict(plain.to_wire(), oracle={"reach_target": {"target": "butter_1", "body": "butter_1_main", "position": np.asarray([0.1, 0.2, 0.9])}})
    )  # fmt: skip
    arrays = []
    back = interface.Observation.from_wire(wire._decode(wire._encode(told.to_wire(), arrays), arrays))
    np.testing.assert_array_equal(back.oracle["reach_target"]["position"], [0.1, 0.2, 0.9])


def test_a_unit_run_on_named_tasks_is_collected_only_when_asked_and_says_so(tmp_path):
    rows = test_collect.ALL[:2]  # task 0, its two start states
    unit = test_collect.a_unit(tmp_path / "out", ["0of1"], [rows], **{"0of1": {"tasks": "0"}})
    with pytest.raises(collect.Refused) as refused:
        collect.collect_unit(unit, tmp_path / "results")
    assert "a part of the block" in str(refused.value)
    summary = collect.collect_unit(unit, tmp_path / "results", part=True)
    assert summary == "PP-O x fake: 2 episodes, success 2, reach 2/2 (a part of the block: tasks 0)"
    record = json.loads((tmp_path / "results" / "PP-O" / "fake" / "config.json").read_text())
    assert record["selection"] == {"tasks": "0", "states": "all", "shards": 1}
    assert [task["index"] for task in record["tasks"]] == [0]


def test_shards_that_ran_different_tasks_are_refused(tmp_path):
    unit = test_collect.a_unit(
        tmp_path / "out", ["0of2", "1of2"], [test_collect.ALL[:2], test_collect.ALL[2:]],
        **{"0of2": {"tasks": "0"}, "1of2": {"tasks": "1"}},
    )  # fmt: skip
    with pytest.raises(collect.Refused) as refused:
        collect.collect_unit(unit, tmp_path / "results", part=True)
    assert "different tasks" in str(refused.value)


# ------------------------------------------------------------------------------------------- the oracle channel



def test_the_body_of_a_target():
    assert oracle.target_body(s.TABLES, {"target": ["alphabet_soup_1"]}) == ("alphabet_soup_1", "alphabet_soup_1_main", s.SOUP)
    assert oracle.target_body(s.TABLES, {"target": ["wooden_cabinet_1:middle"]})[1:] == ("wooden_cabinet_1_cabinet_middle", s.DRAWER)
    assert oracle.target_body(s.TABLES, {"target": ["wooden_cabinet_1"]})[2] == s.CABINET  # no _main: the first body
    for reach in (None, {"target": []}, {"target": ["alphabet_soup_1", "tomato_sauce_1"]}, {"target": ["plate_1"]}):
        with pytest.raises(ValueError):
            oracle.target_body(s.TABLES, reach)


def test_what_a_row_may_ask_of_the_oracle():
    assert oracle.requested({"requires": {"camera_height": 256}}) == []
    assert oracle.requested({"requires": {"oracle": ["reach_target_position"]}}) == ["reach_target_position"]
    with pytest.raises(SystemExit):
        oracle.requested({"requires": {"oracle": ["segmentation"]}})
    with pytest.raises(SystemExit):
        oracle.check_tasks([{"index": 0, "task": "open the drawer", "reach": None}])
    assert "oracle" in oracle.note(["reach_target_position"])["is"]
