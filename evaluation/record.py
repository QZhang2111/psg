"""Writing what a unit leaves for git: the config record and the episode table.

The config record is assembled here from three sources and from nothing typed at a prompt: the block's
definition, what the simulator process measured when it loaded the block's files, and the adapter's report.

Python 3.8 and later, with NumPy (`interface` and `rollout` import it).
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation import interface
from evaluation import reach
from evaluation import rollout
from evaluation import schema

#: What the environment seed changes, as measured on a simulator on 2026-11-12.
ENVIRONMENT_SEED_NOTE = (
    "The seed sets NumPy's global generator in the simulator process; it is set anew before every episode's "
    "reset. The start state then overwrites the joint positions and velocities of the scene. What the seed "
    "still decides is what the reset draws that is not part of that state: the benchmark places its fixtures "
    "(cabinet, stove, rack) at every reset within the ranges of the task's file, 1-2 cm. A fresh environment's "
    "first seeded reset always gives the same placement; a later reset of the same environment gives it in "
    "most tasks but not in all. So every reset's placement is compared with the scene's first, and the "
    "environment is built anew when it differs: every episode of a task starts from the same placement, "
    "whatever was run before it."
)


def config_record(
    *,
    block: Dict[str, Any],
    row: str,
    tasks: List[Dict[str, Any]],
    measured: Dict[str, Any],
    report: Dict[str, Any],
    selection: Dict[str, Any],
    revision: str,
    clean: bool,
) -> Dict[str, Any]:
    """`tasks` are the block's task entries as run (digests as loaded, start states as resolved);
    `measured` is `simulator.measured()` plus what the first scene showed."""
    rollout_settings = block["rollout"]
    has_predicates = any(task.get("predicate") for task in tasks)
    return {
        "schema": schema.SCHEMA,
        "unit": {"block": block["block"], "row": row},
        "code": {"revision": revision, "clean": bool(clean)},
        "benchmark": dict(block["benchmark"], simulator=dict(measured["simulator"], environment=block["environment"])),
        "tasks": tasks,
        "seeds": {
            "noise": list(block["seeds"]),
            "environment": rollout_settings["environment_seed"],
            "what_the_environment_seed_changes": ENVIRONMENT_SEED_NOTE,
        },
        "rollout": {
            "settle_steps": rollout_settings["settle_steps"],
            "settle_action": rollout.SETTLE_ACTION,
            "max_steps": rollout_settings["max_steps"],
            "stop": "the episode ends at the first step at which a listed condition of the task holds"
            if has_predicates
            else "the episode ends at the first step at which the predicate holds, or at the step limit",
            "reset": "environment seed, reset, start state, before every episode; the reset's placement of the fixtures is checked against the scene's first and the environment built anew when it differs",
            "retry": "none: an episode that raises is written with stopped = error and run again by the next launch",
        },
        "observation": dict(
            measured["observation"],
            cameras=dict(interface.CAMERAS),
            image="uint8, as the simulator returns it, not flipped",
            state=["eef_pos", "eef_quat", "gripper_qpos", "joint_pos"],
            camera_parameters="intrinsics and extrinsics of both cameras at every query",
        ),
        "endpoints": {
            "success": "the instructed condition of the task holds when the episode ends"
            if has_predicates
            else "the BDDL goal of the task's file, checked after every policy step",
            "labeller": {"revision": reach.REVISION, "parameters": reach.PARAMETERS},
        },
        "model": report["model"],
        "inference": report["inference"],
        "method": report["method"],
        "noise_seed_controls": report["noise_seed_controls"],
        "model_environment": report["environment"],
        "selection": selection,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n")
    os.replace(str(temporary), str(path))


def write_record(path: Path, record: Dict[str, Any], trial: bool = False) -> None:
    """Writes a shard's record, or checks that the one already there describes the same run.

    A unit refuses a run with other settings. A trial (`--smoke`) may run again with other settings: the record
    and the episode table of the earlier trial are then set aside beside the new ones, not mixed with them.
    """
    if path.exists():
        before = json.loads(path.read_text())
        changed = schema.record_differences(before, record) + (
            ["selection"] if before.get("selection") != record.get("selection") else []
        )
        if changed and trial:
            table = path.with_name(path.name.replace("config.", "episodes.", 1).replace(".json", ".csv"))
            number = 1
            while path.with_name("%s.earlier-%d" % (path.name, number)).exists():
                number += 1
            for old in (path, table):
                if old.exists():
                    old.rename(old.with_name("%s.earlier-%d" % (old.name, number)))
            write_json(path, record)
            return
        if changed:
            raise SystemExit(
                "%s was written by a run with other settings (sections: %s).\n"
                "A unit is one set of settings: use another output root, or remove the unit's directory."
                % (path, ", ".join(changed))
            )
        return
    write_json(path, record)


def read_table(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(str(path), newline="") as handle:
        rows = list(csv.DictReader(handle))
    if rows and list(rows[0]) != schema.EPISODE_COLUMNS:
        raise SystemExit("%s does not have the episode table's columns" % path)
    return rows


def append_row(path: Path, row: Dict[str, Any]) -> None:
    unknown = set(row) - set(schema.EPISODE_COLUMNS)
    if unknown:
        raise ValueError("not columns of the episode table: %s" % sorted(unknown))
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(str(path), "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=schema.EPISODE_COLUMNS, restval="")
        if new:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def latest_rows(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    """The last row written for each episode (a failed episode is run again and written again)."""
    return {row["episode"]: row for row in rows}


def resolve_start_states(block: Dict[str, Any], task: Dict[str, Any], only: Optional[List[int]] = None) -> List[int]:
    from evaluation import layout

    states = layout.parse_ranges(str(task.get("start_states") or block["start_states"])) or []
    return [state for state in states if only is None or state in only]
