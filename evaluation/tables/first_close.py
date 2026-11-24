"""Where the gripper first closed, counted per task from episode tables of the Object scenes.

    python -m evaluation.tables.first_close results/PP-O/pi05/episodes.csv --tasks 0,3,6,8,9
    python -m evaluation.tables.first_close results/PP-O/*/episodes.csv results/L-O/pi05/episodes.csv --out counts.csv

One table per episode table given (any row of the blocks `PP-O` and `L-O`: a policy, a method, an instrument's
arm). Per task, every episode falls in exactly one of four places, by the episode table's `first_close_entity`
(the object nearest the hand when the gripper first closes; no distance limit, as Reach):

    on the requested food          the object the instruction names
    on the body at its place       the body that stands where the food stood in training
                                   (LIBERO-PRO Position exchanged the two; in the original scenes this is the
                                   food itself, and the column is empty)
    on anything else               another object or the basket
    no close                       the gripper never closed

`success` is the episodes with success = 1, beside them. Which body stands where comes from
`object_places.csv` (written by `python -m evaluation.tables.scenes places` from the block's definition and
recorded start states).

**strict** is the paper's stricter count of the first column: a close on the requested food counts only in the
tasks where the moved food does not stand on the other training place (`requested_on_other_training_place` = 0).
LIBERO-Object trains the requested food of every task on one of two places; where the exchange puts the food
on the one its own task does not use, a close there does not show that the hand followed the food. Those
tasks' line says `-` and adds nothing to the sum.

`--tasks` picks tasks of the block (default: every task in the episode table). An episode table with a failed
episode, an episode twice, or a task the scene table does not know is refused.

Python 3.8 and later, standard library only.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
PLACES = HERE / "object_places.csv"

COUNT_COLUMNS = [
    "block", "row", "task_index", "requested", "at_training_place", "requested_on_other_training_place",
    "episodes", "close_on_requested", "close_on_body_at_training_place", "close_on_anything_else", "no_close",
    "success", "close_on_requested_strict",
]  # fmt: skip


class Refused(Exception):
    pass


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(str(path), newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_places(path: Path = PLACES) -> Dict[Tuple[str, int], Dict[str, str]]:
    return {(line["block"], int(line["task_index"])): line for line in read_csv(path)}


def count(
    episodes: List[Dict[str, str]], places: Dict[Tuple[str, int], Dict[str, str]], tasks: Optional[Sequence[int]] = None
) -> List[Dict[str, Any]]:
    """The count lines of one episode table: one per task, then their sum (`task_index` = "all")."""
    units = {(e["block"], e["row"]) for e in episodes}
    if len(units) != 1:
        raise Refused("not one unit's episode table: %s" % sorted(units))
    ((block, row),) = units
    names = [e["episode"] for e in episodes]
    if len(set(names)) != len(names):
        raise Refused("%s x %s: an episode is there twice" % (block, row))
    failed = [e["episode"] for e in episodes if e["stopped"] == "error"]
    if failed:
        raise Refused("%s x %s: %d episodes failed (first: %s)" % (block, row, len(failed), failed[0]))
    present = sorted({int(e["task_index"]) for e in episodes})
    chosen = present if tasks is None else list(tasks)
    missing = [task for task in chosen if task not in present]
    if missing:
        raise Refused("%s x %s has no episode of the tasks %s" % (block, row, missing))

    lines: List[Dict[str, Any]] = []
    for task in chosen:
        place = places.get((block, task))
        if place is None:
            raise Refused("the scene table has no line for task %d of %s" % (task, block))
        food, other = place["requested"], place["at_training_place"]
        strict_counts = place["requested_on_other_training_place"] == "0"
        line: Dict[str, Any] = {
            "block": block, "row": row, "task_index": task, "requested": food,
            "at_training_place": "" if other == food else other,
            "requested_on_other_training_place": place["requested_on_other_training_place"],
            "episodes": 0, "close_on_requested": 0, "close_on_body_at_training_place": 0 if other != food else "",
            "close_on_anything_else": 0, "no_close": 0, "success": 0,
        }  # fmt: skip
        for e in episodes:
            if int(e["task_index"]) != task:
                continue
            line["episodes"] += 1
            line["success"] += int(e["success"])
            if e["first_close_step"] == "":
                if e["first_close_entity"] not in ("", "none"):
                    raise Refused("%s: no step of a first close, and yet it is on %s" % (e["episode"], e["first_close_entity"]))
                line["no_close"] += 1
            elif e["first_close_entity"] == food:
                line["close_on_requested"] += 1
            elif e["first_close_entity"] == other:
                line["close_on_body_at_training_place"] += 1
            else:
                line["close_on_anything_else"] += 1
        line["close_on_requested_strict"] = line["close_on_requested"] if strict_counts else ""
        lines.append(line)

    total: Dict[str, Any] = {"block": block, "row": row, "task_index": "all", "requested": "", "at_training_place": "",
                             "requested_on_other_training_place": ""}  # fmt: skip
    for name in COUNT_COLUMNS[6:]:
        values = [line[name] for line in lines if line[name] != ""]
        total[name] = sum(values) if values else ""
    return lines + [total]


def as_text(lines: List[Dict[str, Any]]) -> str:
    block, row = lines[0]["block"], lines[0]["row"]
    head = ["task", "requested food", "body at its place", "episodes", "on food", "on that body", "else", "no close", "success", "strict"]
    table = [head]
    for line in lines:
        strict = line["close_on_requested_strict"]
        table.append([
            str(line["task_index"]), line["requested"], line["at_training_place"] or ("" if line["task_index"] == "all" else "(the food)"),
            str(line["episodes"]), str(line["close_on_requested"]),
            "" if line["close_on_body_at_training_place"] == "" else str(line["close_on_body_at_training_place"]),
            str(line["close_on_anything_else"]), str(line["no_close"]), str(line["success"]),
            "-" if strict == "" else str(strict),
        ])  # fmt: skip
    widths = [max(len(entry[i]) for entry in table) for i in range(len(head))]
    out = ["%s x %s: where the gripper first closed" % (block, row)]
    for entry in table:
        cells = [entry[i].ljust(widths[i]) if i < 3 else entry[i].rjust(widths[i]) for i in range(len(head))]
        out.append("  ".join(cells).rstrip())
    return "\n".join(out)


def parse_tasks(text: Optional[str]) -> Optional[List[int]]:
    return None if text is None else [int(part) for part in text.split(",") if part != ""]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("tables", type=Path, nargs="+", help="episode tables: results/<block>/<row>/episodes.csv")
    parser.add_argument("--tasks", help="task indices of the block, e.g. 0,3,6,8,9 (default: all in the table)")
    parser.add_argument("--places", type=Path, default=PLACES, help="the scene table (default: object_places.csv here)")
    parser.add_argument("--out", type=Path, help="also write every count line into this CSV")
    args = parser.parse_args(argv)
    places = load_places(args.places)
    everything: List[Dict[str, Any]] = []
    try:
        for path in args.tables:
            lines = count(read_csv(path), places, parse_tasks(args.tasks))
            print(as_text(lines))
            print()
            everything += lines
    except Refused as reason:
        print("refused: %s" % reason, file=sys.stderr)
        return 2
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(str(args.out), "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=COUNT_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(everything)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
