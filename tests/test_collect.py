"""The record writer and the collection of a unit's shards. Standard library only; files under tmp_path."""

from __future__ import annotations

import json

import pytest

from evaluation import collect
from evaluation import record
from evaluation import schema

BLOCK = {
    "block": "PP-O", "environment": "sim-libero-pro", "base_suite": "libero_object", "loader": "libero",
    "benchmark": {"name": "LIBERO-PRO", "suite": "libero_object_swap", "perturbation": "position"},
    "rollout": {"environment_seed": 7, "settle_steps": 10, "max_steps": 280, "stop": "predicate"},
    "seeds": [7], "start_states": "0-1",
    "tasks": [
        {"index": 0, "task": "soup", "instruction": "pick up the alphabet soup", "reach": {"rule": "first_close", "target": ["alphabet_soup_1"]}},
        {"index": 1, "task": "butter", "instruction": "pick up the butter", "reach": {"rule": "first_close", "target": ["butter_1"]}},
    ],
}  # fmt: skip
REPORT = {
    "model": {"name": "fake", "checkpoint": {"address": "none"}, "code": {"address": "none"}},
    "inference": {"actions_executed_per_query": 5}, "method": {"name": "none"},
    "noise_seed_controls": "nothing", "environment": {"name": "test", "python": "3", "packages": {}},
    "requires": {"camera_height": 256, "camera_width": 256},
}  # fmt: skip
MEASURED = {"simulator": {"packages": {}, "renderer": "none"}, "observation": {"camera_height": 256, "camera_width": 256}}


def a_record(shard="0of1", max_steps=280, clean=True, tasks="all"):
    block = dict(BLOCK, rollout=dict(BLOCK["rollout"], max_steps=max_steps))
    tasks_as_run = [
        dict(task, start_states=[0, 1], bddl_sha256="x", init_sha256="y")
        for task in block["tasks"]
        if tasks == "all" or str(task["index"]) in tasks.split(",")
    ]
    return record.config_record(
        block=block, row="fake", tasks=tasks_as_run, measured=MEASURED, report=REPORT,
        selection={"tasks": tasks, "states": "all", "shard": shard}, revision="abc", clean=clean,
    )  # fmt: skip


def a_row(task, state, **changes):
    row = {
        "episode": schema.episode_id("PP-O", "fake", task, state, 7), "block": "PP-O", "row": "fake",
        "task_index": task, "task": BLOCK["tasks"][task]["task"], "state": state, "seed": 7,
        "instruction": BLOCK["tasks"][task]["instruction"], "success": 1, "success_step": 100, "steps": 100,
        "queries": 20, "stopped": "predicate", "reach": 1, "reach_rule": "first_close", "labeller": "reach-1", "error": "",
    }  # fmt: skip
    row.update(changes)
    return row


def a_unit(root, shards, rows_per_shard, **record_changes):
    unit = root / "PP-O" / "fake"
    for shard, rows in zip(shards, rows_per_shard):
        record.write_record(unit / "shards" / ("config.%s.json" % shard), a_record(shard, **record_changes.get(shard, {})))
        for row in rows:
            record.append_row(unit / "shards" / ("episodes.%s.csv" % shard), row)
    return unit


ALL = [a_row(0, 0), a_row(0, 1), a_row(1, 0), a_row(1, 1, success=0, success_step="", stopped="limit", reach=0)]


def test_the_record_has_every_section_and_nothing_of_a_machine():
    rec = a_record()
    assert set(schema.CONFIG_RECORD) <= set(rec)
    text = json.dumps(rec)
    assert "/home" not in text and "/Users" not in text
    assert rec["seeds"] == {"noise": [7], "environment": 7, "what_the_environment_seed_changes": record.ENVIRONMENT_SEED_NOTE}
    assert rec["rollout"]["max_steps"] == 280 and rec["endpoints"]["labeller"]["revision"] == "reach-2"
    assert rec["benchmark"]["simulator"]["environment"] == "sim-libero-pro"


def test_a_shard_record_is_written_once_and_a_changed_run_is_refused(tmp_path):
    path = tmp_path / "shards" / "config.0of1.json"
    record.write_record(path, a_record())
    record.write_record(path, a_record())  # the same run again: nothing to say
    with pytest.raises(SystemExit):
        record.write_record(path, a_record(max_steps=220))
    with pytest.raises(SystemExit):
        record.write_record(path, a_record(tasks="0"))


def test_the_table_keeps_its_columns(tmp_path):
    table = tmp_path / "episodes.0of1.csv"
    for row in ALL:
        record.append_row(table, row)
    rows = record.read_table(table)
    assert list(rows[0]) == schema.EPISODE_COLUMNS and len(rows) == 4 and rows[3]["success"] == "0"
    with pytest.raises(ValueError):
        record.append_row(table, dict(ALL[0], seconds=3))


def test_a_complete_unit_is_collected_in_the_blocks_order(tmp_path):
    unit = a_unit(tmp_path / "out", ["0of2", "1of2"], [[ALL[2], ALL[0]], [ALL[3], ALL[1]]])
    summary = collect.collect_unit(unit, tmp_path / "results")
    assert summary == "PP-O x fake: 4 episodes, success 3, reach 3/4"
    out = tmp_path / "results" / "PP-O" / "fake"
    assert [row["episode"] for row in record.read_table(out / "episodes.csv")] == [row["episode"] for row in ALL]
    assert json.loads((out / "config.json").read_text())["selection"] == {"tasks": "all", "states": "all", "shards": 2}


def test_a_failed_episode_that_was_run_again_counts_once(tmp_path):
    failed = a_row(0, 1, success="", success_step="", steps="", queries="", stopped="error", reach="", error="boom")
    unit = a_unit(tmp_path / "out", ["0of1"], [[ALL[0], failed, ALL[2], ALL[3], ALL[1]]])
    assert collect.collect_unit(unit, tmp_path / "results").startswith("PP-O x fake: 4 episodes")


@pytest.mark.parametrize(
    ("shards", "rows", "changes", "reason"),
    [
        (["0of2"], [ALL], {}, "shards not there: 1of2"),
        (["0of1"], [ALL[:3]], {}, "1 of 4 episodes missing"),
        (["0of1"], [ALL[:3] + [a_row(1, 1, stopped="error", error="boom")]], {}, "1 failed"),
        (["0of2", "1of2"], [ALL[:2], ALL[2:]], {"1of2": {"max_steps": 220}}, "other settings: rollout"),
        (["0of1"], [ALL], {"0of1": {"clean": False}}, "checkout with changes"),
        (["0of1"], [ALL], {"0of1": {"tasks": "0"}}, "a part of the block"),
        (["0of1"], [ALL + [a_row(1, 5)]], {}, "1 not of this block"),
    ],
)
def test_what_is_refused(tmp_path, shards, rows, changes, reason):
    unit = a_unit(tmp_path / "out", shards, rows, **changes)
    with pytest.raises(collect.Refused) as refused:
        collect.collect_unit(unit, tmp_path / "results")
    assert reason in str(refused.value)
    assert not (tmp_path / "results").exists()


def test_smoke_output_is_never_collected(tmp_path):
    a_unit(tmp_path / "out" / "_smoke", ["0of1"], [ALL])
    unit = a_unit(tmp_path / "out", ["0of1"], [ALL])
    assert collect.units(tmp_path / "out") == [unit]
    assert collect.units(tmp_path / "out", block="PL") == []


def test_a_trial_may_run_again_with_other_settings_and_sets_the_earlier_one_aside(tmp_path):
    shards = tmp_path / "shards"
    record.write_record(shards / "config.0of1.json", a_record())
    record.append_row(shards / "episodes.0of1.csv", ALL[0])
    record.write_record(shards / "config.0of1.json", a_record(max_steps=220), trial=True)
    assert json.loads((shards / "config.0of1.json").read_text())["rollout"]["max_steps"] == 220
    assert json.loads((shards / "config.0of1.json.earlier-1").read_text())["rollout"]["max_steps"] == 280
    assert (shards / "episodes.0of1.csv.earlier-1").exists() and not (shards / "episodes.0of1.csv").exists()
