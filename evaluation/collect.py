"""Collect finished units: the two files a unit leaves, `config.json` and `episodes.csv`.

    python -m evaluation.collect <a copy of an output root> [--block <block>] [--row <row>] [--results <directory>]

For every `<root>/<block>/<row>/shards/` it checks, and refuses the unit with the reason when a check fails:

- every shard of the unit is there (`0of<n>` ... `<n-1>of<n>`), each run on the whole block (no --tasks/--states);
  with `--part`, a unit that was run on named tasks of its block (the same `--tasks` in every shard, every start
  state of them) is taken too, and its collected record keeps the tasks under `selection`: it says it is a part;
- the shards' config records are equal in every section but the selection: one unit, one set of settings;
- the records come from a clean checkout;
- every task x start state x seed of the block has exactly one row, and none of them failed.

Then it writes `<block>/<row>/config.json` and `episodes.csv` (rows in the block's order): by default into the
output root it was given, beside each unit's `shards/`, where `python -m evaluation.tables.render --results <root>`
reads them. It writes into this checkout's `results/`, which holds the paper's units, only when `--results results`
names it, and then replaces the unit's two files there. The copy can
have come from any machine by any transport; nothing here knows a host. `_smoke/` is never collected.

Python 3.8 and later, with NumPy: the command reads no array, but the module it takes its table readers from
(`record.py`) imports NumPy. A simulator environment's Python has it.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evaluation import layout
from evaluation import record
from evaluation import schema

_SHARD = re.compile(r"^config\.(\d+)of(\d+)\.json$")


class Refused(Exception):
    pass


def collect_unit(unit: Path, results: Path, part: bool = False) -> str:
    """Checks one unit directory and writes its two files under `results`; returns a one-line summary.

    `part`: also take a unit run on named tasks of its block (an instrument's run, never a table's unit)."""
    shards = unit / "shards"
    found: Dict[Tuple[int, int], Path] = {}
    for path in sorted(shards.glob("config.*.json")):
        match = _SHARD.match(path.name)
        if match:
            found[(int(match.group(1)), int(match.group(2)))] = path
    if not found:
        raise Refused("no shard record")
    counts = {n for _, n in found}
    if len(counts) != 1:
        raise Refused("shards of different splits: %s" % sorted(layout.shard_name(s) for s in found))
    (n,) = counts
    missing = [layout.shard_name((k, n)) for k in range(n) if (k, n) not in found]
    if missing:
        raise Refused("shards not there: %s" % ", ".join(missing))

    records = {shard: json.loads(path.read_text()) for shard, path in found.items()}
    first = records[(0, n)]
    for shard, rec in sorted(records.items()):
        whole = rec["selection"].get("tasks") == "all"
        if rec["selection"].get("states") != "all" or not (whole or part):
            raise Refused("shard %s ran a part of the block (%s)" % (layout.shard_name(shard), rec["selection"]))
        if rec["selection"].get("tasks") != first["selection"].get("tasks"):
            raise Refused("the shards ran different tasks of the block")
        different = schema.record_differences(first, rec)
        if different:
            raise Refused("shard %s was run with other settings: %s" % (layout.shard_name(shard), ", ".join(different)))
        if not rec["code"].get("clean"):
            raise Refused("shard %s was run from a checkout with changes" % layout.shard_name(shard))

    rows: List[Dict[str, str]] = []
    for shard in sorted(records):
        rows += record.read_table(shards / ("episodes.%s.csv" % layout.shard_name(shard)))
    latest = record.latest_rows(rows)
    block, row = first["unit"]["block"], first["unit"]["row"]
    expected = [
        schema.episode_id(block, row, task["index"], state, seed)
        for task in first["tasks"]
        for state in task["start_states"]
        for seed in first["seeds"]["noise"]
    ]
    absent = [name for name in expected if name not in latest]
    extra = sorted(set(latest) - set(expected))
    failed = [name for name in expected if name in latest and latest[name]["stopped"] == "error"]
    if absent or extra or failed:
        raise Refused(
            "%d of %d episodes missing, %d failed, %d not of this block%s"
            % (len(absent), len(expected), len(failed), len(extra), "; first missing: " + absent[0] if absent else "")
        )

    out = results / block / row
    out.mkdir(parents=True, exist_ok=True)
    tasks_run = first["selection"]["tasks"]  # "all" for a table's unit
    record.write_json(out / "config.json", dict(first, selection={"tasks": tasks_run, "states": "all", "shards": n}))
    with open(str(out / "episodes.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=schema.EPISODE_COLUMNS)
        writer.writeheader()
        writer.writerows(latest[name] for name in expected)
    successes = sum(int(latest[name]["success"] or 0) for name in expected)
    reached = [latest[name]["reach"] for name in expected if latest[name]["reach"] != ""]
    reach = "%d/%d" % (sum(int(value) for value in reached), len(reached)) if reached else "not defined"
    summary = "%s x %s: %d episodes, success %d, reach %s" % (block, row, len(expected), successes, reach)
    return summary if tasks_run == "all" else summary + " (a part of the block: tasks %s)" % tasks_run


def units(root: Path, block: Optional[str] = None, row: Optional[str] = None) -> List[Path]:
    return [
        path.parent
        for path in sorted(root.glob("*/*/shards"))
        if path.parts[-3] != "_smoke" and block in (None, path.parts[-3]) and row in (None, path.parts[-2])
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", type=Path, help="a copy of an output root: <root>/<block>/<row>/shards/")
    parser.add_argument("--block")
    parser.add_argument("--row")
    parser.add_argument(
        "--results", type=Path, help="where <block>/<row>/ goes; default: the output root given, beside each unit's shards"
    )
    parser.add_argument("--part", action="store_true", help="also take a unit run on named tasks of its block")
    args = parser.parse_args()
    refused = 0
    for unit in units(args.root, args.block, args.row):
        try:
            print("collected  " + collect_unit(unit, args.results or args.root, part=args.part))
        except Refused as reason:
            refused += 1
            print("refused    %s x %s: %s" % (unit.parts[-2], unit.parts[-1], reason))
    return 1 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
