"""The paper's Tables 1-3 from `results/`, as the paper printed them (compared with `printed/`).

    python -m evaluation.tables.render                    # prints the three tables
    python -m evaluation.tables.render --table 2          # one of them
    python -m evaluation.tables.render --out <directory>  # also writes table1.csv, table2.csv, table3.csv
    python -m evaluation.tables.render --check            # compares with evaluation/tables/printed/ and says what differs

Everything a table is made of is data beside this file, not code here:

- `tables.json`: which row of this repository is which line, which block is which column, the rule of every
  average, the three distance groups of Table 2;
- `os_g_scenes.csv`: every scene of Table 2 with its distance and its group (written by
  `python -m evaluation.tables.scenes distances` from recorded episodes);
- `cited_libero.csv`: the seven LIBERO averages the paper cites, each with its public source;
- `printed/table<n>.csv`: the tables as the paper prints them, for `--check`.

A number is a percentage of a unit's episodes, from the unit's episode table
(`results/<block>/<row>/episodes.csv`, columns `success` and `reach`), computed as an exact fraction and
rounded to one decimal with a half rounded up. A unit is read only if it is whole: every task x start state x
seed its config record names is there once and none failed; otherwise the command stops and says which unit.

The paper's bold is not printed: there it marks the best of a column among the rows that need no training,
and the rows that require training (†) are gray and not ranked.

Python 3.8 and later, standard library only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "results"
PRINTED = HERE / "printed"
METRIC_NAMES = {"reach": "Reach", "success": "Succ."}


class Refused(Exception):
    pass


def half_up(value: Fraction) -> str:
    """`value` to one decimal, a half rounded up: 83.75 -> "83.8"."""
    return "%d.%d" % divmod(math.floor(value * 10 + Fraction(1, 2)), 10)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(str(path), newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class Unit:
    """One row on one block: its episodes, checked to be the whole unit."""

    def __init__(self, results: Path, block: str, row: str) -> None:
        self.block, self.row = block, row
        folder = results / block / row
        if not (folder / "episodes.csv").exists() or not (folder / "config.json").exists():
            raise Refused("no unit %s x %s under %s" % (block, row, results))
        record = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        selection = record.get("selection", {})
        if selection.get("tasks") != "all" or selection.get("states") != "all":
            raise Refused("%s x %s is a part of its block (%s), not a table's unit" % (block, row, selection))
        expected = [
            "%s.%s.t%d.s%d.n%d" % (block, row, task["index"], state, seed)
            for task in record["tasks"]
            for state in task["start_states"]
            for seed in record["seeds"]["noise"]
        ]
        rows = read_csv(folder / "episodes.csv")
        names = [line["episode"] for line in rows]
        if sorted(names) != sorted(expected):
            raise Refused(
                "%s x %s: the episode table has %d episodes, %d of them the %d its record names"
                % (block, row, len(names), len(set(names) & set(expected)), len(expected))
            )
        failed = [line["episode"] for line in rows if line["stopped"] == "error"]
        if failed:
            raise Refused("%s x %s: %d episodes failed (first: %s)" % (block, row, len(failed), failed[0]))
        self.episodes = rows

    def count(self, metric: str, tasks: Optional[Sequence[int]] = None) -> Tuple[int, int]:
        """(episodes with the metric = 1, episodes), over the whole unit or over the named tasks."""
        chosen = [e for e in self.episodes if tasks is None or int(e["task_index"]) in tasks]
        values = [e[metric] for e in chosen]
        if not chosen or any(value not in ("0", "1") for value in values):
            raise Refused("%s x %s: %r is not 0 or 1 in every episode asked for" % (self.block, self.row, metric))
        return sum(int(value) for value in values), len(values)


class Source:
    """`results/`, the tables' definition and the two small tables it names."""

    def __init__(self, results: Path = RESULTS, tables: Path = HERE / "tables.json") -> None:
        self.results = results
        self.definition = json.loads(tables.read_text(encoding="utf-8"))
        self.folder = tables.parent
        self._units: Dict[Tuple[str, str], Unit] = {}
        self._cited: Optional[Dict[str, Dict[str, str]]] = None
        self._scenes: Dict[str, List[Dict[str, str]]] = {}

    def table(self, number: int) -> Dict[str, Any]:
        for table in self.definition["tables"]:
            if table["table"] == number:
                return table
        raise Refused("tables.json defines no Table %d" % number)

    def unit(self, block: str, row: str) -> Unit:
        if (block, row) not in self._units:
            self._units[(block, row)] = Unit(self.results, block, row)
        return self._units[(block, row)]

    def cited(self, row: str) -> Fraction:
        if self._cited is None:
            self._cited = {line["row"]: line for line in read_csv(self.folder / "cited_libero.csv")}
        if row not in self._cited:
            raise Refused("cited_libero.csv has no line for the row %s" % row)
        return Fraction(self._cited[row]["average"])

    def scene_tasks(self, table: Dict[str, Any], group: str) -> List[int]:
        scenes = table["scenes"]
        if scenes["file"] not in self._scenes:
            self._scenes[scenes["file"]] = read_csv(self.folder / scenes["file"])
        tasks = [
            int(line["task_index"])
            for line in self._scenes[scenes["file"]]
            if line["block"] == scenes["block"] and line["group"] == group
        ]
        if not tasks:
            raise Refused("%s names no scene of %s in the group %r" % (scenes["file"], scenes["block"], group))
        return tasks


def cell(source: Source, table: Dict[str, Any], row: Dict[str, str], column: Dict[str, Any], metric: str) -> str:
    """One printed number: the row's value in the column, by the column's rule."""
    if column.get("or_cited") and row.get(column["or_cited"]) == "cited":
        return half_up(source.cited(row["row"])) + "*"
    if "mean_of_blocks" in column:
        parts = []
        for block in column["mean_of_blocks"]:
            hits, episodes = source.unit(block, row["row"]).count(metric)
            parts.append(Fraction(100 * hits, episodes))
        return half_up(sum(parts, Fraction(0)) / len(parts))
    tasks = source.scene_tasks(table, column["scene_group"]) if "scene_group" in column else None
    hits, episodes = source.unit(column["block"], row["row"]).count(metric, tasks)
    return half_up(Fraction(100 * hits, episodes))


def render(source: Source, number: int) -> Tuple[List[str], List[List[str]]]:
    """(header, lines) of one table: the header's entries are "group | column | metric"."""
    table = source.table(number)
    header = ["Method", "row"]
    for column in table["columns"]:
        header += ["%s | %s | %s" % (column["group"], column["name"], METRIC_NAMES[m]) for m in column["metrics"]]
    lines = []
    for row in table["rows"]:
        line = [row["label"], row["row"]]
        for column in table["columns"]:
            line += [cell(source, table, row, column, metric) for metric in column["metrics"]]
        lines.append(line)
    return header, lines


def as_text(source: Source, number: int, header: List[str], lines: List[List[str]]) -> str:
    table = source.table(number)
    marks = source.definition["marks"]
    used = [mark for mark in marks if any(mark in value for line in lines for value in [line[0]] + line[2:])]
    title = "Table %d. %s" % (number, table["caption"])
    title += "".join(" %s: %s." % (mark, marks[mark].split(",")[0]) for mark in used)
    label = max(len(line[0]) for line in lines) + 2
    width = 8
    tops = [entry.split(" | ") for entry in header[2:]]
    out = [title]
    for level in range(2):  # the group's name, then the column's, each centred over the numbers it spans
        text, at = " " * label, 0
        while at < len(tops):
            span = 1
            while at + span < len(tops) and tops[at + span][: level + 1] == tops[at][: level + 1]:
                span += 1
            text += tops[at][level].center(width * span)
            at += span
        out.append(text.rstrip())
    out.append("Method".ljust(label) + "".join(top[2].rjust(width) for top in tops))
    for line in lines:
        out.append(line[0].ljust(label) + "".join(value.rjust(width) for value in line[2:]))
    return "\n".join(out)


def write_csv(path: Path, header: List[str], lines: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(lines)


def check(number: int, header: List[str], lines: List[List[str]], printed: Path) -> Tuple[int, List[str]]:
    """(numbers equal, one line per difference) against the table as `printed/` holds it."""
    path = printed / ("table%d.csv" % number)
    with open(str(path), newline="", encoding="utf-8") as handle:
        wanted = list(csv.reader(handle))
    if wanted[0] != header or [line[:2] for line in wanted[1:]] != [line[:2] for line in lines]:
        return 0, ["Table %d: %s has other rows or columns than tables.json" % (number, path.name)]
    equal, different = 0, []
    for mine, theirs in zip(lines, wanted[1:]):
        for name, a, b in zip(header[2:], mine[2:], theirs[2:]):
            if a == b:
                equal += 1
            else:
                different.append("Table %d, %s, %s: from results/ %s, printed %s" % (number, mine[0], name, a, b))
    return equal, different


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--table", type=int, choices=[1, 2, 3], action="append", help="default: all three")
    parser.add_argument("--results", type=Path, default=RESULTS, help="default: this checkout's results/")
    parser.add_argument("--out", type=Path, help="also write table<n>.csv into this directory")
    parser.add_argument("--check", action="store_true", help="compare with the tables as printed (evaluation/tables/printed/)")
    args = parser.parse_args(argv)
    source = Source(args.results)
    differences: List[str] = []
    try:
        for number in args.table or [1, 2, 3]:
            header, lines = render(source, number)
            print(as_text(source, number, header, lines))
            print()
            if args.out:
                write_csv(args.out / ("table%d.csv" % number), header, lines)
            if args.check:
                equal, different = check(number, header, lines, PRINTED)
                print("Table %d: %d of %d numbers equal to the printed table" % (number, equal, equal + len(different)))
                print()
                differences += different
    except Refused as reason:
        print("refused: %s" % reason, file=sys.stderr)
        return 2
    for line in differences:
        print(line)
    return 1 if differences else 0


if __name__ == "__main__":
    raise SystemExit(main())
