"""The evaluation process: run one row on one block, or a shard of it, against the row's model process.

    <the block's python> -m evaluation.run_unit --block <block> --row <row> --port <port> \
        --unit-dir <dir> --shard <k>of<n> [--tasks ...] [--states ...] [--smoke]

Started by `run.py`, not by hand. It asks the model process for its report, loads the block's files, writes the
shard's config record, then for every episode of the shard: reset, settle, roll out, save the trace and the
video, label Reach from the trace, append one row to the shard's episode table. Episodes already in the table
are skipped, so a stopped run goes on where it stopped; a failed episode is run again.

Python 3.8 and later, in the block's simulator environment.
"""

from __future__ import annotations

import argparse
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

from evaluation import interface
from evaluation import layout
from evaluation import oracle as _oracle
from evaluation import reach
from evaluation import record
from evaluation import rollout
from evaluation import schema
from evaluation import trace as _trace
from evaluation import video as _video

MAX_ERRORS_IN_A_ROW = 3  # then the unit stops: something is wrong with the model or the simulator, not an episode


def episodes_of(block: Dict[str, Any], tasks: List[Dict[str, Any]], only_states: Any) -> List[Tuple[Dict[str, Any], int, int]]:
    """Every (task, start state, noise seed) of the selection, in the block's order."""
    return [
        (task, state, seed)
        for task in tasks
        for state in record.resolve_start_states(block, task, only_states)
        for seed in block["seeds"]
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--block", required=True)
    parser.add_argument("--row", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--unit-dir", type=Path, required=True)
    parser.add_argument("--shard", default="0of1")
    parser.add_argument("--tasks")
    parser.add_argument("--states")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    block = layout.load_block(args.block)
    shard = layout.parse_shard(args.shard)
    only_tasks, only_states = layout.parse_ranges(args.tasks), layout.parse_ranges(args.states)
    tasks = [task for task in block["tasks"] if only_tasks is None or task["index"] in only_tasks]
    episodes = episodes_of(block, tasks, only_states)[shard[0] :: shard[1]]
    if not episodes:
        raise SystemExit("the selection holds no episode of block %r" % args.block)

    from evaluation import simulator  # the benchmark's code is imported here and nowhere earlier

    model = rollout.ModelClient(args.port)
    instructions = sorted({task["instruction"] for task in tasks})
    report = model.report(instructions)
    height, width = int(report["requires"]["camera_height"]), int(report["requires"]["camera_width"])
    oracle_kinds = _oracle.requested(report)  # empty for every row of the paper's tables
    if oracle_kinds:
        _oracle.check_tasks(tasks)

    tasks_as_run = []
    for task in tasks:
        entry = dict(task, **simulator.task_digests(task))
        entry["start_states"] = record.resolve_start_states(block, task, None)
        entry["base_suite"] = task.get("base_suite") or block.get("base_suite")
        entry["max_steps"] = int(task.get("max_steps") or block["rollout"]["max_steps"])
        tasks_as_run.append(entry)

    scene = simulator.Scene(block, episodes[0][0], height, width)
    scene.reset(episodes[0][1])
    measured = simulator.measured()
    measured["observation"] = {
        "camera_height": height,
        "camera_width": width,
        "controller": {"action_scale": [float(x) for x in scene.action_scale()]},
    }
    if oracle_kinds:  # what was passed, and that it is an oracle
        measured["observation"]["oracle"] = _oracle.note(oracle_kinds)
    selection = {"tasks": args.tasks or "all", "states": args.states or "all", "shard": args.shard}
    shards = args.unit_dir / "shards"
    record.write_record(
        shards / ("config.%s.json" % args.shard),
        record.config_record(
            block=block, row=args.row, tasks=tasks_as_run, measured=measured, report=report, selection=selection,
            revision=os.environ.get("PSG_REVISION", "unknown"), clean=os.environ.get("PSG_CLEAN") == "1",
        ),
        trial=args.smoke,
    )  # fmt: skip

    table = shards / ("episodes.%s.csv" % args.shard)
    done = {name for name, row in record.latest_rows(record.read_table(table)).items() if row["stopped"] != "error"}
    raw = args.unit_dir / "raw"
    current_task, errors_in_a_row = episodes[0][0]["index"], 0
    try:
        for task, state, seed in episodes:
            name = schema.episode_id(args.block, args.row, task["index"], state, seed)
            if name in done:
                continue
            if task["index"] != current_task:
                scene.close()
                scene = simulator.Scene(block, task, height, width)
                current_task = task["index"]
            base = {
                "episode": name, "block": args.block, "row": args.row, "task_index": task["index"],
                "task": task["task"], "state": state, "seed": seed, "instruction": task["instruction"],
            }  # fmt: skip
            try:
                started = time.monotonic()
                scene.reset(state)
                episode = interface.Episode(
                    episode=name, block=args.block, task_index=task["index"], task=task["task"],
                    suite=task.get("base_suite") or block.get("base_suite"), state=state, noise_seed=seed,
                    instruction=task["instruction"],
                )  # fmt: skip
                recorder = _trace.TraceWriter(
                    dict(base, settle_steps=block["rollout"]["settle_steps"]), scene.tables()
                )
                video = _video.VideoWriter(raw / "videos" / (name + ".mp4"))
                outcome = rollout.run_episode(
                    scene, model, episode,
                    settle_steps=int(block["rollout"]["settle_steps"]),
                    max_steps=int(task.get("max_steps") or block["rollout"]["max_steps"]),
                    recorder=recorder, video=video, oracle=oracle_kinds,
                )  # fmt: skip
                trace_path = raw / "traces" / (name + ".npz")
                digest = recorder.save(trace_path)
                video.save()
                labels = reach.label(_trace.load(trace_path), task.get("reach"))
                record.append_row(table, dict(base, **outcome, **labels, trace_sha256=digest, error=""))
                errors_in_a_row = 0
                seconds = time.monotonic() - started
                # How long an episode took is a fact of the machine, not of the unit: it goes to the logs.
                (raw / "logs").mkdir(parents=True, exist_ok=True)
                with open(str(raw / "logs" / ("timing.%s.csv" % args.shard)), "a") as log:
                    log.write("%s,%d,%d,%.2f,%d\n" % (name, outcome["steps"], outcome["queries"], seconds, scene.rebuilds))
                print(
                    "[episode] %s success=%s reach=%s steps=%s seconds=%.1f"
                    % (name, outcome["success"], labels["reach"], outcome["steps"], seconds),
                    flush=True,
                )
            except Exception:
                error = traceback.format_exc().strip().splitlines()[-1]
                (raw / "logs").mkdir(parents=True, exist_ok=True)
                with open(str(raw / "logs" / ("errors.%s.log" % args.shard)), "a") as log:
                    log.write("%s\n%s\n" % (name, traceback.format_exc()))
                record.append_row(table, dict(base, stopped="error", labeller=reach.REVISION, error=error))
                print("[episode] %s error: %s" % (name, error), flush=True)
                errors_in_a_row += 1
                if errors_in_a_row >= MAX_ERRORS_IN_A_ROW:
                    raise SystemExit("%d episodes in a row failed; see raw/logs" % errors_in_a_row)
    finally:
        scene.close()
        try:
            model.close()
        except Exception:
            pass
    failed = [name for name, row in record.latest_rows(record.read_table(table)).items() if row["stopped"] == "error"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
