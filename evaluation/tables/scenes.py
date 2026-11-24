"""Two small tables of scenes, read from recorded episodes: where the requested object stands.

    python -m evaluation.tables.scenes distances <a unit's raw/traces of OS-G> --out evaluation/tables/os_g_scenes.csv
    python -m evaluation.tables.scenes places <a unit's raw/traces of L-O> <a unit's raw/traces of PP-O> \
        --out evaluation/tables/object_places.csv

Both read one thing from every trace: the position (x, y) of bodies at the frame of the episode's first query,
which is the start state after the settling steps, before the policy has acted. So the tables are facts of the
scenes and do not depend on the row whose traces are read; any row's traces of the block give the same file.

**distances** (Table 2's scenes). For a block whose tasks are scenes of one object at several places (each task
of `block.json` has `place` and `base_task_index`): per start state, how far the requested object (the task's
Reach target) stands from where it stands in the scene of the same base task whose place is `training`, in the
same start state. A scene's distance is the mean over its start states; the smallest and the largest are kept
beside it. Its group is the one of Table 2's groups (`tables.json`) the mean falls in.

**places** (the scenes of the first-close count). LIBERO-PRO Position exchanges two objects of an Object scene
(`block.json` of `PP-O`: each task's `swapped`), one of them the requested food. Per task the table says which
body stands at the food's training place (the other of the pair), and whether the moved food now stands on the
other training place: LIBERO-Object puts the requested food of every task on one of two places of the table, and
where the exchange puts the food on the one its own task does not use, a hand that goes to that place closes on
the food without having looked for it. The traces check the block's definition (the other of the pair stands
within 2 cm of the food's training place in every start state) and give the two places; the command refuses to
write when they disagree. The original scenes (`L-O`) get a line per task too: there the food itself stands at
its training place.

Python 3.8 and later, NumPy.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from evaluation import oracle as _oracle
from evaluation import trace as _trace

HERE = Path(__file__).resolve().parent
BLOCKS = HERE.parent / "blocks"

#: How near "at the place" is, in metres. The largest distance measured between a body and the place it took
#: over is 1.4 cm (two bodies drawn anew around one place); the two training places lie 22 cm apart.
AT_PLACE_M = 0.02

DISTANCE_COLUMNS = [
    "block", "task_index", "task", "object", "place", "start_states",
    "distance_cm", "distance_cm_min", "distance_cm_max", "group",
]  # fmt: skip
PLACE_COLUMNS = [
    "block", "task_index", "task", "requested", "at_training_place", "requested_on_other_training_place",
    "training_place_x_m", "training_place_y_m", "requested_stands_x_m", "requested_stands_y_m",
    "start_states", "largest_offset_from_the_place_cm",
]  # fmt: skip


class Refused(Exception):
    pass


def load_block(block: str) -> Dict[str, Any]:
    return json.loads((BLOCKS / block / "block.json").read_text(encoding="utf-8"))


def start_positions(traces: Path, block: str) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """{(task index, start state): {body name: (x, y) at the first query's frame}} for every trace of the block."""
    out: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
    for path in sorted(traces.glob("*.npz")):
        trace = _trace.load(path)
        meta = _trace.meta(trace)
        if str(meta["block"]) != block:
            raise Refused("%s is an episode of %s, not of %s" % (path.name, meta["block"], block))
        if len(trace["query_frame"]) == 0:
            raise Refused("%s has no query" % path.name)
        frame = int(trace["query_frame"][0])
        names = [str(name) for name in trace["body_name"]]
        key = (int(meta["task_index"]), int(meta["state"]))
        if key in out:
            raise Refused("two traces of task %d, start state %d" % key)
        positions: Dict[str, Any] = {name: trace["body_pos"][frame, i, :2].astype(np.float64) for i, name in enumerate(names)}
        positions[_TABLES] = {name: trace[name] for name in ("body_name", "entity_name", "geom_body", "geom_pos", "gripper_geoms")}
        out[key] = positions
    if not out:
        raise Refused("no trace under %s" % traces)
    return out


_TABLES = "<the scene's tables>"


def _body(positions: Dict[str, Any], entity: str) -> str:
    """The body of an object the labeller's own rule picks (`<entity>_main`, else its first body)."""
    return _oracle.target_body(positions[_TABLES], {"target": [entity]})[1]


def group_of(distance_cm: float, groups: Sequence[Dict[str, Any]]) -> str:
    for group in groups:
        if group.get("from_cm", float("-inf")) <= distance_cm < group.get("below_cm", float("inf")):
            return str(group["name"])
    raise Refused("no group of Table 2 holds %.2f cm" % distance_cm)


def distances(traces: Path, block: str, groups: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    definition = load_block(block)
    positions = start_positions(traces, block)
    tasks = {int(task["index"]): task for task in definition["tasks"]}
    training = {
        task["base_task_index"]: index for index, task in tasks.items() if task.get("place") == "training"
    }
    lines = []
    for index in sorted(tasks):
        task = tasks[index]
        if task.get("place") is None or task.get("base_task_index") not in training:
            raise Refused("task %d of %s names no place or has no training scene" % (index, block))
        entity = task["reach"]["target"][0]
        states = sorted(state for (t, state) in positions if t == index)
        values = []
        for state in states:
            here, there = positions[(index, state)], positions.get((training[task["base_task_index"]], state))
            if there is None:
                raise Refused("task %d, start state %d: its training scene's episode is not among the traces" % (index, state))
            values.append(100.0 * float(np.linalg.norm(here[_body(here, entity)] - there[_body(there, entity)])))
        if not values:
            raise Refused("no trace of task %d" % index)
        mean = float(np.mean(values))
        lines.append({
            "block": block, "task_index": index, "task": task["task"], "object": entity, "place": task["place"],
            "start_states": len(values), "distance_cm": "%.1f" % mean, "distance_cm_min": "%.1f" % min(values),
            "distance_cm_max": "%.1f" % max(values),
            "group": "training" if task["place"] == "training" else group_of(mean, groups),
        })  # fmt: skip
    return lines


def places(original: Path, moved: Path, original_block: str = "L-O", moved_block: str = "PP-O") -> List[Dict[str, Any]]:
    there_all = start_positions(original, original_block)
    here_all = start_positions(moved, moved_block)
    tasks = {int(task["index"]): task for task in load_block(moved_block)["tasks"]}
    names = {int(task["index"]): task["task"] for task in load_block(original_block)["tasks"]}
    if names != {index: task["task"] for index, task in tasks.items()}:
        raise Refused("%s and %s do not have the same tasks at the same indices" % (original_block, moved_block))

    # the requested food's training place per task: its mean position over the original scenes' start states
    trained: Dict[int, np.ndarray] = {}
    for index, task in tasks.items():
        entity = task["reach"]["target"][0]
        own = [positions for (t, _), positions in sorted(there_all.items()) if t == index]
        if not own:
            raise Refused("no trace of task %d of %s" % (index, original_block))
        trained[index] = np.mean([positions[_body(positions, entity)] for positions in own], axis=0)

    lines: List[Dict[str, Any]] = []
    for block, positions_all in ((original_block, there_all), (moved_block, here_all)):
        for index in sorted(tasks):
            task = tasks[index]
            entity = task["reach"]["target"][0]
            states = sorted(state for (t, state) in positions_all if t == index)
            if block == moved_block:
                pair = list(task.get("swapped") or [])
                if len(pair) != 2 or entity not in pair:
                    raise Refused("task %d of %s: `swapped` is not the requested object and one other" % (index, block))
                other = [name for name in pair if name != entity][0]
            else:
                other = entity
            offsets, stands = [], []
            for state in states:
                here, there = positions_all[(index, state)], there_all.get((index, state))
                if there is None:
                    raise Refused("task %d, start state %d is not among the traces of %s" % (index, state, original_block))
                offsets.append(float(np.linalg.norm(here[_body(here, other)] - there[_body(there, entity)])))
                stands.append(here[_body(here, entity)])
            if not states or max(offsets) > AT_PLACE_M:
                raise Refused(
                    "task %d of %s: %s does not stand at the training place of %s in every start state (largest "
                    "offset %.1f cm)" % (index, block, other, entity, 100 * max(offsets or [float("nan")]))
                )
            # does the food stand on a training place of the suite that is not its own task's?
            others = [place for place in trained.values() if np.linalg.norm(place - trained[index]) > AT_PLACE_M]
            on_other = [min(float(np.linalg.norm(xy - place)) for place in others) <= AT_PLACE_M for xy in stands]
            if any(on_other) != all(on_other):
                raise Refused("task %d of %s: the food stands on the other training place in some start states only" % (index, block))
            stand = np.mean(stands, axis=0)
            lines.append({
                "block": block, "task_index": index, "task": task["task"], "requested": entity,
                "at_training_place": other, "requested_on_other_training_place": int(all(on_other)),
                "training_place_x_m": "%.2f" % trained[index][0], "training_place_y_m": "%.2f" % trained[index][1],
                "requested_stands_x_m": "%.2f" % stand[0], "requested_stands_y_m": "%.2f" % stand[1],
                "start_states": len(states), "largest_offset_from_the_place_cm": "%.1f" % (100 * max(offsets)),
            })  # fmt: skip
    return lines


def write(path: Path, columns: List[str], lines: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("distances", help="Table 2's scenes: distance from the training place, and group")
    one.add_argument("traces", type=Path, help="<output root>/OS-G/<row>/raw/traces")
    one.add_argument("--block", default="OS-G")
    one.add_argument("--out", type=Path, required=True)
    two = commands.add_parser("places", help="which body stands where in the Position scenes of Object")
    two.add_argument("original", type=Path, help="<output root>/L-O/<row>/raw/traces")
    two.add_argument("moved", type=Path, help="<output root>/PP-O/<row>/raw/traces")
    two.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "distances":
            definition = json.loads((HERE / "tables.json").read_text(encoding="utf-8"))
            groups = [t for t in definition["tables"] if t["table"] == 2][0]["scenes"]["groups"]
            lines = distances(args.traces, args.block, groups)
            write(args.out, DISTANCE_COLUMNS, lines)
        else:
            lines = places(args.original, args.moved)
            write(args.out, PLACE_COLUMNS, lines)
    except Refused as reason:
        print("refused: %s" % reason)
        return 2
    print("wrote %d lines to %s" % (len(lines), args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
