#!/usr/bin/env python3
"""The one entry: run one row on one evaluation block.

    python -m evaluation.run --block PP-O --row pi05_psg
    python -m evaluation.run --block CF-S --row apt

from the repository's root.

The line has the same shape for every row. It starts the row's model in the row's environment and the block's
simulator in the block's environment, lets them talk through the interface of `evaluation/interface.py`,
and writes under the output directory the unit's config record, its episode table and the raw episodes.
Nobody types a config: the record is what the two processes report about themselves.

What a part of a unit or a smoke needs:

    --block PT-O,PT-G,PT-S       several blocks of the row, one after another, with the model loaded once
    --tasks 0,3 --states 0-4     a subset of the block's tasks and start states
    --shard 0of2                 every second episode of the unit, for a second card or machine
    --out <directory>            the output root (default: $PSG_OUTPUT, else ./outputs)
    --smoke                      write under <root>/_smoke/, allow a checkout with changes; never collected
    --list                       the blocks and rows this checkout defines

The card is chosen by the caller's CUDA_VISIBLE_DEVICES, as for any program; when it names one card, the simulator
renders on that card too (`layout.simulator_environment`), and `raw/logs/launch.<shard>.jsonl` says which card each
process was given. Python 3.8 or later, standard
library only: this file runs with whatever Python is at hand; the two environments are found by name
(`environments.local.json`, see evaluation/layout.py).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation import layout  # noqa: E402

MODEL_START_TIMEOUT_S = 3600  # loading weights from a cold disk; the model process says when it is ready


def checkout_state():
    """(commit, clean) of this checkout; ("unknown", None) where it is no git checkout (a ZIP download) or where
    git is not installed."""

    def git(*args):
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout

    try:
        return git("rev-parse", "HEAD").strip(), not git("status", "--porcelain").strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown", None


CARD_VARIABLES = ("CUDA_VISIBLE_DEVICES", "MUJOCO_GL", "MUJOCO_EGL_DEVICE_ID", "XLA_PYTHON_CLIENT_MEM_FRACTION")


def note_launch(unit, tag, block, row, revision, clean, base, model_environment, simulator_environment) -> None:
    """One line per launch in the unit's logs: the card each of the two processes was given.

    A fact of the machine, so it goes to `raw/logs/` and not into the config record, which two shards of a unit
    on two cards have to share."""
    line = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "block": block, "row": row, "shard": tag,
        "revision": revision, "clean": clean, "card": layout.launch_card(base),
        "model": {name: model_environment[name] for name in CARD_VARIABLES if name in model_environment},
        "simulator": {name: simulator_environment[name] for name in CARD_VARIABLES if name in simulator_environment},
    }  # fmt: skip
    with open(str(unit / "raw" / "logs" / ("launch.%s.jsonl" % tag)), "a") as log:
        log.write(json.dumps(line, sort_keys=True) + "\n")


def shut_down(model: subprocess.Popen, port: str) -> None:
    """Asks the model process to finish (it then closes its adapter); ends it if it does not."""
    if model.poll() is None:
        try:
            from evaluation import wire  # NumPy is needed only here; without it the process is ended

            with socket.create_connection(("127.0.0.1", int(port)), timeout=10) as connection:
                wire.call(connection, "shutdown")
            model.wait(timeout=30)
        except Exception:
            pass
    if model.poll() is None:
        model.terminate()
        try:
            model.wait(timeout=30)
        except subprocess.TimeoutExpired:
            model.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one row on one evaluation block.")
    parser.add_argument("--block", help="an evaluation block; several, separated by commas, share one loaded model")
    parser.add_argument("--row")
    parser.add_argument("--tasks", help="task indices of the block, e.g. 0,3 or 0-9 (default: all)")
    parser.add_argument("--states", help="start-state indices, e.g. 0-4 (default: the block's)")
    parser.add_argument("--shard", default="0of1", help="<k>of<n>: the k-th of n interleaved parts of the unit")
    parser.add_argument("--out", help="output root (default: $PSG_OUTPUT, else ./outputs)")
    parser.add_argument("--smoke", action="store_true", help="a trial: written under _smoke/, never collected")
    parser.add_argument("--list", action="store_true", help="list the blocks and rows and exit")
    args = parser.parse_args()

    if args.list:
        print("blocks:", ", ".join(layout.blocks()) or "none")
        print("rows:  ", ", ".join(layout.rows()) or "none")
        return 0
    if not args.block or not args.row:
        parser.error("--block and --row are required")

    names = [name for name in args.block.split(",") if name]
    blocks = [layout.load_block(name) for name in names]
    row = layout.load_row(args.row)
    shard = layout.parse_shard(args.shard)
    layout.parse_ranges(args.tasks), layout.parse_ranges(args.states)  # fail here on a malformed selection
    revision, clean = checkout_state()
    if clean is False and not args.smoke:
        raise SystemExit("this checkout has changes; commit them, or pass --smoke for a trial")
    if clean is None:
        print(
            "not a git checkout: the records say revision unknown and clean false, and collect does not take them",
            file=sys.stderr,
        )

    model_python = layout.interpreter(row["environment"])
    simulator_pythons = [layout.interpreter(block["environment"]) for block in blocks]  # all found before a start
    root = layout.output_root(args.out)
    units = [layout.unit_dir(root, name, args.row, smoke=args.smoke) for name in names]
    for unit in units:
        (unit / "raw" / "logs").mkdir(parents=True, exist_ok=True)
    tag = layout.shard_name(shard)

    base = dict(os.environ)
    base["PSG_REVISION"] = revision
    base["PSG_CLEAN"] = "1" if clean else "0"
    model_environment = layout.process_environment(row["environment"], base)

    with tempfile.TemporaryDirectory() as scratch:
        port_file = Path(scratch) / "port"
        logs = units[0] / "raw" / "logs"  # the model's log is kept with the first block's unit
        with open(logs / ("model.%s.log" % tag), "ab") as model_log:
            model = subprocess.Popen(
                [model_python, "-m", "evaluation.rows.serve", "--row", args.row, "--port-file", str(port_file)],
                cwd=str(REPO), env=model_environment, stdout=model_log, stderr=subprocess.STDOUT,
            )  # fmt: skip
        port = ""
        try:
            started = time.monotonic()
            while not port_file.exists():
                if model.poll() is not None:
                    raise SystemExit("the model process of row %r stopped while loading; see %s" % (args.row, logs))
                if time.monotonic() - started > MODEL_START_TIMEOUT_S:
                    raise SystemExit("the model process of row %r did not become ready" % args.row)
                time.sleep(0.5)
            port = port_file.read_text().strip()
            worst = 0
            for name, block, simulator_python, unit in zip(names, blocks, simulator_pythons, units):
                if model.poll() is not None:
                    raise SystemExit("the model process of row %r stopped; see %s" % (args.row, logs))
                command = [
                    simulator_python, "-m", "evaluation.run_unit",
                    "--block", name, "--row", args.row, "--port", port, "--unit-dir", str(unit), "--shard", tag,
                ]  # fmt: skip
                for flag, value in (("--tasks", args.tasks), ("--states", args.states)):
                    if value:
                        command += [flag, value]
                if args.smoke:
                    command.append("--smoke")
                environment = layout.simulator_environment(block["environment"], base)  # renders on the launch's card
                note_launch(unit, tag, name, args.row, revision, clean, base, model_environment, environment)
                worst = max(worst, subprocess.call(command, cwd=str(REPO), env=environment))
            return worst
        finally:
            shut_down(model, port)


if __name__ == "__main__":
    sys.exit(main())
