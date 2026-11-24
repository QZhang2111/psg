"""The tables' command, the first-close count and the two scene tables (evaluation/tables/).

Standard library and NumPy; no card, no model, no simulator. Runs on any machine with them.
"""

from __future__ import annotations

import csv
import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from evaluation import schema
from evaluation.tables import first_close
from evaluation.tables import render
from evaluation.tables import scenes

# ----------------------------------------------------------------------------------------------------------------
# render
# ----------------------------------------------------------------------------------------------------------------


def test_a_half_is_rounded_up_from_the_exact_fraction():
    assert render.half_up(Fraction(8375, 100)) == "83.8"
    assert render.half_up(Fraction(9665, 100)) == "96.7"
    assert render.half_up(Fraction(9515, 100)) == "95.2"
    assert render.half_up(Fraction(8374, 100)) == "83.7"
    assert render.half_up(Fraction(0)) == "0.0"
    assert render.half_up(Fraction(100)) == "100.0"
    assert render.half_up(Fraction(100, 3)) == "33.3"
    assert render.half_up(Fraction(200, 3)) == "66.7"


def episode(block: str, row: str, task: int, state: int, **values: Any) -> Dict[str, str]:
    line = {name: "" for name in schema.EPISODE_COLUMNS}
    line.update(
        episode=schema.episode_id(block, row, task, state, 7), block=block, row=row, task_index=str(task),
        state=str(state), seed="7", success="0", stopped="limit", reach="0",
    )  # fmt: skip
    line.update({name: str(value) for name, value in values.items()})
    return line


def write_unit(results: Path, block: str, row: str, outcomes: Dict[int, List[str]], selection: str = "all") -> None:
    """A unit with, per task, one episode per entry: "rs" = reach and success, "r" = reach, "" = neither."""
    folder = results / block / row
    folder.mkdir(parents=True)
    lines, tasks = [], []
    for task, entries in sorted(outcomes.items()):
        tasks.append({"index": task, "start_states": list(range(len(entries)))})
        for state, entry in enumerate(entries):
            lines.append(episode(block, row, task, state, reach=int("r" in entry), success=int("s" in entry)))
    record = {"tasks": tasks, "seeds": {"noise": [7]}, "selection": {"tasks": selection, "states": "all", "shards": 1}}
    (folder / "config.json").write_text(json.dumps(record))
    with open(str(folder / "episodes.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=schema.EPISODE_COLUMNS)
        writer.writeheader()
        writer.writerows(lines)


DEFINITION = {
    "marks": {"†": "requires training", "*": "reported by the authors, not run here"},
    "tables": [
        {
            "table": 1,
            "caption": "A small table.",
            "rows": [{"label": "ours", "row": "a", "first": "measured"}, {"label": "theirs†", "row": "b", "first": "cited"}],
            "scenes": {"file": "scenes.csv", "block": "B2"},
            "columns": [
                {"group": "G", "name": "Avg", "metrics": ["success"], "mean_of_blocks": ["B1", "B2"], "or_cited": "first"},
                {"group": "H", "name": "one", "metrics": ["reach", "success"], "block": "B1"},
                {"group": "H", "name": "near", "metrics": ["reach", "success"], "block": "B2", "scene_group": "near"},
                {"group": "H", "name": "Avg", "metrics": ["reach", "success"], "mean_of_blocks": ["B1", "B2"]},
            ],
        }
    ],
}


@pytest.fixture()
def small(tmp_path: Path) -> render.Source:
    results = tmp_path / "results"
    # B1: 3 episodes, B2: 2 tasks (scenes) of 1 and 4 episodes
    write_unit(results, "B1", "a", {0: ["rs", "", ""]})
    write_unit(results, "B2", "a", {0: ["rs"], 1: ["r", "r", "", ""]})
    write_unit(results, "B1", "b", {0: ["r", "r", ""]})
    write_unit(results, "B2", "b", {0: [""], 1: ["", "", "", ""]})
    (tmp_path / "tables.json").write_text(json.dumps(DEFINITION))
    (tmp_path / "cited_libero.csv").write_text("row,average\nb,94.15\n")
    (tmp_path / "scenes.csv").write_text("block,task_index,group\nB2,0,training\nB2,1,near\nB9,0,near\n")
    return render.Source(results, tmp_path / "tables.json")


def test_an_average_is_the_mean_of_the_blocks_percentages_not_the_pooled_count(small: render.Source):
    header, lines = render.render(small, 1)
    assert header == [
        "Method", "row", "G | Avg | Succ.", "H | one | Reach", "H | one | Succ.", "H | near | Reach",
        "H | near | Succ.", "H | Avg | Reach", "H | Avg | Succ.",
    ]  # fmt: skip
    ours = dict(zip(header, lines[0]))
    # success: B1 1 of 3 = 33.33.., B2 1 of 5 = 20: their mean is 26.67; pooled it would be 2 of 8 = 25.0
    assert ours["G | Avg | Succ."] == "26.7"
    assert ours["H | Avg | Succ."] == "26.7"
    # reach: B1 1 of 3, B2 3 of 5 = 60: mean 46.67
    assert ours["H | Avg | Reach"] == "46.7"
    assert (ours["H | one | Reach"], ours["H | one | Succ."]) == ("33.3", "33.3")


def test_a_scene_group_counts_the_groups_scenes_only(small: render.Source):
    header, lines = render.render(small, 1)
    ours = dict(zip(header, lines[0]))
    # the group "near" of block B2 is task 1 alone: reach 2 of 4, success 0 of 4; task 0 (training) is left out
    assert (ours["H | near | Reach"], ours["H | near | Succ."]) == ("50.0", "0.0")


def test_a_cited_number_is_printed_with_its_mark_and_half_up(small: render.Source):
    header, lines = render.render(small, 1)
    theirs = dict(zip(header, lines[1]))
    assert theirs["G | Avg | Succ."] == "94.2*"
    assert theirs["H | one | Reach"] == "66.7"
    text = render.as_text(small, 1, header, lines)
    assert "†: requires training." in text and "*: reported by the authors." in text
    assert text.splitlines()[-1].split() == ["theirs†", "94.2*", "66.7", "0.0", "0.0", "0.0", "33.3", "0.0"]


def test_a_unit_that_is_not_whole_is_refused(tmp_path: Path):
    results = tmp_path / "results"
    write_unit(results, "B1", "a", {0: ["rs", ""]})
    lines = (results / "B1" / "a" / "episodes.csv").read_text().splitlines()
    (results / "B1" / "a" / "episodes.csv").write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(render.Refused, match="1 episodes, 1 of them the 2"):
        render.Unit(results, "B1", "a")

    write_unit(results, "B1", "failed", {0: ["rs", ""]})
    table = results / "B1" / "failed" / "episodes.csv"
    table.write_text(table.read_text().replace(",limit,", ",error,", 1))
    with pytest.raises(render.Refused, match="1 episodes failed"):
        render.Unit(results, "B1", "failed")

    write_unit(results, "B1", "part", {0: ["rs", ""]}, selection="0,3")
    with pytest.raises(render.Refused, match="a part of its block"):
        render.Unit(results, "B1", "part")

    with pytest.raises(render.Refused, match="no unit B1 x absent"):
        render.Unit(results, "B1", "absent")


def test_the_check_names_every_number_that_differs(small: render.Source, tmp_path: Path):
    header, lines = render.render(small, 1)
    render.write_csv(tmp_path / "printed" / "table1.csv", header, lines)
    assert render.check(1, header, lines, tmp_path / "printed") == (14, [])
    changed = [list(line) for line in lines]
    changed[0][2] = "26.6"
    equal, different = render.check(1, header, changed, tmp_path / "printed")
    assert equal == 13
    assert different == ["Table 1, ours, G | Avg | Succ.: from results/ 26.6, printed 26.7"]


def test_the_repositorys_results_give_the_tables_as_printed(capsys):
    """results/ and the rules of tables.json give every number of the paper's Tables 1-3."""
    if not render.RESULTS.exists():
        pytest.skip("this checkout has no results/")
    assert render.main(["--check"]) == 0
    out = capsys.readouterr().out
    assert "Table 1: 130 of 130 numbers equal to the printed table" in out
    assert "Table 2: 60 of 60 numbers equal to the printed table" in out
    assert "Table 3: 32 of 32 numbers equal to the printed table" in out


def test_table_2s_groups_are_the_scene_tables_distances_cut_as_tables_json_says():
    definition = json.loads((render.HERE / "tables.json").read_text(encoding="utf-8"))
    groups = [t for t in definition["tables"] if t["table"] == 2][0]["scenes"]["groups"]
    lines = render.read_csv(render.HERE / "os_g_scenes.csv")
    assert len(lines) == 21
    for line in lines:
        wanted = "training" if line["place"] == "training" else scenes.group_of(float(line["distance_cm"]), groups)
        assert line["group"] == wanted
    sizes = {name: sum(1 for line in lines if line["group"] == name) for name in ("training", "< 20", "20-30", "> 30")}
    assert sizes == {"training": 3, "< 20": 4, "20-30": 4, "> 30": 10}


# ----------------------------------------------------------------------------------------------------------------
# first_close
# ----------------------------------------------------------------------------------------------------------------

PLACES = {
    ("PP-O", 0): {"requested": "soup_1", "at_training_place": "butter_1", "requested_on_other_training_place": "0"},
    ("PP-O", 3): {"requested": "bbq_1", "at_training_place": "pudding_1", "requested_on_other_training_place": "1"},
    ("L-O", 0): {"requested": "soup_1", "at_training_place": "soup_1", "requested_on_other_training_place": "0"},
}


def closes(block: str, task: int, entities: List[Optional[str]], successes: int = 0) -> List[Dict[str, str]]:
    return [
        episode(
            block, "pi05", task, state, success=int(state < successes),
            first_close_step="" if entity is None else 40, first_close_entity="" if entity is None else entity,
        )
        for state, entity in enumerate(entities)
    ]  # fmt: skip


def test_every_episode_falls_in_one_of_the_four_places():
    episodes = closes("PP-O", 0, ["soup_1", "soup_1", "butter_1", "basket_1", "none", None], successes=2)
    episodes += closes("PP-O", 3, ["bbq_1", "pudding_1", "pudding_1", None], successes=1)
    task0, task3, total = first_close.count(episodes, PLACES)
    assert (task0["episodes"], task0["close_on_requested"], task0["close_on_body_at_training_place"]) == (6, 2, 1)
    assert (task0["close_on_anything_else"], task0["no_close"], task0["success"]) == (2, 1, 2)
    assert task0["close_on_requested_strict"] == 2
    assert (task3["close_on_requested"], task3["close_on_body_at_training_place"], task3["no_close"]) == (1, 2, 1)
    # the food of task 3 stands on the other training place: its close on the food is not in the strict count
    assert task3["close_on_requested_strict"] == ""
    assert total["task_index"] == "all"
    assert (total["episodes"], total["close_on_requested"], total["close_on_requested_strict"]) == (10, 3, 2)
    assert (total["close_on_body_at_training_place"], total["close_on_anything_else"], total["no_close"]) == (3, 2, 2)
    assert total["success"] == 3
    for line in (task0, task3, total):
        assert line["episodes"] == (
            line["close_on_requested"] + line["close_on_body_at_training_place"] + line["close_on_anything_else"] + line["no_close"]
        )


def test_tasks_can_be_picked_and_the_original_scenes_have_no_other_body():
    episodes = closes("PP-O", 0, ["soup_1"]) + closes("PP-O", 3, ["bbq_1"])
    lines = first_close.count(episodes, PLACES, tasks=[3])
    assert [line["task_index"] for line in lines] == [3, "all"]
    original = first_close.count(closes("L-O", 0, ["soup_1", "butter_1", None], successes=1), PLACES)
    assert (original[0]["close_on_requested"], original[0]["close_on_anything_else"], original[0]["no_close"]) == (1, 1, 1)
    assert original[0]["at_training_place"] == "" and original[0]["close_on_body_at_training_place"] == ""
    assert original[0]["close_on_requested_strict"] == 1
    text = first_close.as_text(original)
    assert "(the food)" in text and text.splitlines()[0] == "L-O x pi05: where the gripper first closed"


def test_an_episode_table_that_cannot_be_counted_is_refused():
    with pytest.raises(first_close.Refused, match="no line for task 5"):
        first_close.count(closes("PP-O", 5, ["soup_1"]), PLACES)
    with pytest.raises(first_close.Refused, match="no episode of the tasks"):
        first_close.count(closes("PP-O", 0, ["soup_1"]), PLACES, tasks=[0, 3])
    failed = closes("PP-O", 0, ["soup_1", "soup_1"])
    failed[1]["stopped"] = "error"
    with pytest.raises(first_close.Refused, match="1 episodes failed"):
        first_close.count(failed, PLACES)
    with pytest.raises(first_close.Refused, match="there twice"):
        first_close.count(closes("PP-O", 0, ["soup_1"]) * 2, PLACES)
    with pytest.raises(first_close.Refused, match="not one unit"):
        first_close.count(closes("PP-O", 0, ["soup_1"]) + closes("L-O", 0, ["soup_1"]), PLACES)


def test_every_object_episode_table_of_the_repository_is_counted():
    if not render.RESULTS.exists():
        pytest.skip("this checkout has no results/")
    places = first_close.load_places()
    assert len(places) == 20
    tables = sorted(render.RESULTS.glob("PP-O/*/episodes.csv")) + sorted(render.RESULTS.glob("L-O/*/episodes.csv"))
    assert tables
    for path in tables:
        episodes = first_close.read_csv(path)
        total = first_close.count(episodes, places)[-1]
        assert total["episodes"] == len(episodes)
        # Reach, where the block defines it, is the close on the requested food
        if episodes[0]["reach"] != "":
            assert total["close_on_requested"] == sum(int(e["reach"]) for e in episodes)


# ----------------------------------------------------------------------------------------------------------------
# scenes
# ----------------------------------------------------------------------------------------------------------------

BODIES = ["world", "gripper0_finger", "soup_1_main", "butter_1_main", "milk_1_main"]


def write_trace(folder: Path, block: str, task: int, state: int, xy: Dict[str, Any]) -> None:
    """A trace of 12 frames whose first query is at frame 10; the bodies stand at `xy` from frame 10 on and
    somewhere else before, so a reader of another frame is found out."""
    folder.mkdir(parents=True, exist_ok=True)
    body_pos = np.full((12, len(BODIES), 3), 9.0, np.float32)
    for name, place in xy.items():
        body_pos[10:, BODIES.index(name + "_main"), :2] = place
    np.savez(
        str(folder / ("%s.x.t%d.s%d.n7.npz" % (block, task, state))),
        meta=np.asarray(json.dumps({"block": block, "task_index": task, "state": state, "seed": 7})),
        query_frame=np.asarray([10, 11], np.int32),
        body_name=np.asarray(BODIES),
        body_pos=body_pos,
        entity_name=np.asarray(["soup_1", "butter_1", "milk_1"]),
        geom_body=np.arange(len(BODIES), dtype=np.int32),
        geom_pos=np.zeros((len(BODIES), 3), np.float32),
        gripper_geoms=np.asarray([1], np.int32),
    )


def write_block(blocks: Path, block: str, tasks: List[Dict[str, Any]]) -> None:
    (blocks / block).mkdir(parents=True)
    (blocks / block / "block.json").write_text(json.dumps({"block": block, "tasks": tasks}))


GROUPS = [{"name": "< 20", "below_cm": 20}, {"name": "20-30", "from_cm": 20, "below_cm": 30}, {"name": "> 30", "from_cm": 30}]


def test_a_scenes_distance_is_the_mean_over_its_start_states_from_the_training_scene(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(scenes, "BLOCKS", tmp_path / "blocks")
    reach = {"rule": "first_close", "target": ["soup_1"]}
    write_block(tmp_path / "blocks", "OS-X", [
        {"index": 0, "task": "soup", "place": "training", "base_task_index": 4, "reach": reach},
        {"index": 1, "task": "soup", "place": "x1y1", "base_task_index": 4, "reach": reach},
        {"index": 2, "task": "soup", "place": "x2y2", "base_task_index": 4, "reach": reach},
    ])  # fmt: skip
    traces = tmp_path / "traces"
    for state, start in ((6, [0.0, 0.0]), (7, [0.02, 0.0])):
        write_trace(traces, "OS-X", 0, state, {"soup_1": start})
        write_trace(traces, "OS-X", 1, state, {"soup_1": [0.20, 0.0]})  # 20 cm and 18 cm away: mean 19
        write_trace(traces, "OS-X", 2, state, {"soup_1": [0.0, 0.30]})  # 30 cm and 30.07 cm away
    lines = scenes.distances(traces, "OS-X", GROUPS)
    assert [(line["place"], line["distance_cm"], line["group"]) for line in lines] == [
        ("training", "0.0", "training"), ("x1y1", "19.0", "< 20"), ("x2y2", "30.0", "> 30"),
    ]  # fmt: skip
    assert (lines[1]["distance_cm_min"], lines[1]["distance_cm_max"], lines[1]["start_states"]) == ("18.0", "20.0", 2)
    assert scenes.group_of(20.0, GROUPS) == "20-30" and scenes.group_of(29.99, GROUPS) == "20-30"

    (traces / "OS-X.x.t0.s7.n7.npz").unlink()
    with pytest.raises(scenes.Refused, match="its training scene's episode is not among the traces"):
        scenes.distances(traces, "OS-X", GROUPS)


def object_blocks(blocks: Path, swapped_of_task_1: List[str]) -> None:
    names = ["pick_soup", "pick_butter"]
    write_block(blocks, "L-O", [{"index": i, "task": name, "reach": None} for i, name in enumerate(names)])
    write_block(blocks, "PP-O", [
        {"index": 0, "task": names[0], "reach": {"rule": "first_close", "target": ["soup_1"]}, "swapped": ["milk_1", "soup_1"]},
        {"index": 1, "task": names[1], "reach": {"rule": "first_close", "target": ["butter_1"]}, "swapped": swapped_of_task_1},
    ])  # fmt: skip


def object_traces(root: Path) -> None:
    a, b, c = [-0.12, -0.24], [0.05, -0.10], [0.20, 0.20]  # the two training places and a third place
    for state in (0, 1):
        # original scenes: the soup is trained on a, the butter on b; the milk stands on c
        write_trace(root / "L-O", "L-O", 0, state, {"soup_1": a, "butter_1": b, "milk_1": c})
        write_trace(root / "L-O", "L-O", 1, state, {"soup_1": a, "butter_1": b, "milk_1": c})
        # task 0: soup and milk exchanged, the soup goes to c, which is no training place
        write_trace(root / "PP-O", "PP-O", 0, state, {"soup_1": c, "butter_1": b, "milk_1": [a[0] + 0.005, a[1]]})
        # task 1: butter and soup exchanged, the butter goes to a, the other training place
        write_trace(root / "PP-O", "PP-O", 1, state, {"soup_1": b, "butter_1": [a[0], a[1] + 0.01], "milk_1": c})


def test_the_places_table_says_who_stands_at_the_training_place_and_where_the_food_went(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(scenes, "BLOCKS", tmp_path / "blocks")
    object_blocks(tmp_path / "blocks", ["butter_1", "soup_1"])
    object_traces(tmp_path)
    lines = scenes.places(tmp_path / "L-O", tmp_path / "PP-O")
    assert [(line["block"], line["task_index"]) for line in lines] == [("L-O", 0), ("L-O", 1), ("PP-O", 0), ("PP-O", 1)]
    original, _, soup, butter = lines
    assert (original["requested"], original["at_training_place"], original["requested_on_other_training_place"]) == ("soup_1", "soup_1", 0)
    assert (soup["requested"], soup["at_training_place"], soup["requested_on_other_training_place"]) == ("soup_1", "milk_1", 0)
    assert (butter["requested"], butter["at_training_place"], butter["requested_on_other_training_place"]) == ("butter_1", "soup_1", 1)
    assert (soup["training_place_x_m"], soup["training_place_y_m"]) == ("-0.12", "-0.24")
    assert (soup["requested_stands_x_m"], soup["requested_stands_y_m"]) == ("0.20", "0.20")
    assert soup["largest_offset_from_the_place_cm"] == "0.5" and soup["start_states"] == 2


def test_the_places_table_is_refused_when_the_traces_contradict_the_block(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(scenes, "BLOCKS", tmp_path / "blocks")
    object_blocks(tmp_path / "blocks", ["butter_1", "milk_1"])  # the block says milk; the traces put the soup there
    object_traces(tmp_path)
    with pytest.raises(scenes.Refused, match="milk_1 does not stand at the training place of butter_1"):
        scenes.places(tmp_path / "L-O", tmp_path / "PP-O")
