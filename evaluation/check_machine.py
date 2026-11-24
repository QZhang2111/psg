"""Check a machine before a run, without a card and without loading a model.

    python3 -m evaluation.check_machine                 every row and every block
    python3 -m evaluation.check_machine --row pi05_psg --block PP-O

Run from the checkout. For every environment a selected row or block names, it reads this machine's entry in
`environments.local.json` (or the file $PSG_ENVIRONMENTS names) the way `run.py` does, and checks:

- the environment's interpreter exists and can be executed;
- every absolute path among the environment's resolved variables exists (each entry of PYTHONPATH, the
  LIBERO config folder, the weights' caches, ...);
- every variable the environment's declaration (`evaluation/environments/<name>.json`, "variables") sets is still set
  after the machine's entry is applied: for the openpi environments the deterministic GPU operations
  (`XLA_FLAGS` holding `--xla_gpu_deterministic_ops=true`, `TF_CUDNN_DETERMINISTIC=1`). It also prints the
  memory share an environment gives each job (`XLA_PYTHON_CLIENT_MEM_FRACTION`), when one is set.

Then, for every selected row, that each checkpoint the row names is on disk at its recorded revision:
`gs://...` checkpoints under the environment's `OPENPI_DATA_HOME`, Hugging Face checkpoints in the hub cache
under its `HF_HOME` (`hub/models--<org>--<name>/snapshots/<revision>`, and the row's file or subfolder in it).
It counts files; it does not hash them (the run's config record does).

One line per check, `ok` or `FAIL`; the exit status is 1 if any check failed. Python 3.8 and later, standard
library only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from evaluation import layout

Check = Tuple[bool, str]


def _files(path: Path) -> int:
    if path.is_dir():
        return sum(1 for p in path.rglob("*") if p.is_file())
    return 1 if path.is_file() else 0


def environment_checks(environment: str) -> Iterator[Check]:
    """The interpreter, the paths among the resolved variables, the declared variables."""
    try:
        python = layout.interpreter(environment)
    except SystemExit as missing:
        yield False, "no entry for this environment in the machine file (%s)" % str(missing).splitlines()[0]
        return
    yield os.access(python, os.X_OK), "interpreter %s" % python
    resolved = layout.process_environment(environment, {})
    for name, value in sorted(resolved.items()):
        for part in value.split(os.pathsep) if name == "PYTHONPATH" else [value]:
            if part.startswith("/"):
                yield Path(part).exists(), "%s: %s" % (name, part)
    declaration = layout.ENVIRONMENTS / (environment + ".json")
    declared = json.loads(declaration.read_text()).get("variables", {}) if declaration.exists() else {}
    for name, value in sorted(declared.items()):
        if name == "XLA_FLAGS":
            missing = [flag for flag in str(value).split() if flag not in resolved.get(name, "").split()]
            yield not missing, "%s holds %s (resolved: %r)" % (name, value, resolved.get(name))
        else:
            yield resolved.get(name) == str(value), "%s=%s (resolved: %r)" % (name, value, resolved.get(name))
    if "XLA_PYTHON_CLIENT_MEM_FRACTION" in resolved:
        yield True, "memory share per job: %s of the card" % resolved["XLA_PYTHON_CLIENT_MEM_FRACTION"]


def checkpoints(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every checkpoint a row names: its own, an unconditioned one, a VLM, encoders."""
    out = [row["checkpoint"]] + [row[key] for key in ("unconditioned_checkpoint", "vlm") if key in row]
    return out + list((row.get("encoders") or {}).values())


def checkpoint_path(checkpoint: Dict[str, Any], resolved: Dict[str, str]) -> Optional[Path]:
    """Where a checkpoint lies on this machine, from the row's address and the environment's variables."""
    address = checkpoint.get("address") or "huggingface.co/" + checkpoint["hub_id"]
    if address.startswith("gs://"):
        home = resolved.get("OPENPI_DATA_HOME") or os.path.expanduser("~/.cache/openpi")
        return Path(home) / address[len("gs://"):]
    if "huggingface.co/" not in address:
        return None
    hub_id = address.split("huggingface.co/", 1)[1]
    revision = checkpoint.get("revision") or checkpoint.get("revision_at_reading")
    home = resolved.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    path = Path(home) / "hub" / ("models--" + hub_id.replace("/", "--")) / "snapshots" / str(revision)
    for part in (checkpoint.get("file"), checkpoint.get("subfolder")):
        if part:
            path = path / part
    return path


def row_checks(row_name: str) -> Iterator[Check]:
    row = layout.load_row(row_name)
    try:
        layout.interpreter(row["environment"])
    except SystemExit:
        yield False, "its environment %r has no entry in the machine file" % row["environment"]
        return
    resolved = layout.process_environment(row["environment"], {})
    for checkpoint in checkpoints(row):
        path = checkpoint_path(checkpoint, resolved)
        address = checkpoint.get("address") or checkpoint.get("hub_id")
        if path is None:
            yield False, "%s: no rule for where this address lies" % address
            continue
        count = _files(path)
        yield count > 0, "%s: %d files at %s" % (address, count, path)


def check(rows: List[str], blocks: List[str]) -> List[Tuple[str, List[Check]]]:
    environments: Dict[str, List[str]] = {}
    for name in rows:
        environments.setdefault(layout.load_row(name)["environment"], []).append("row " + name)
    for name in blocks:
        environments.setdefault(layout.load_block(name)["environment"], []).append("block " + name)
    out = []
    for environment in sorted(environments):
        title = "environment %s (%s)" % (environment, ", ".join(environments[environment]))
        out.append((title, list(environment_checks(environment))))
    for name in rows:
        out.append(("row %s: weights" % name, list(row_checks(name))))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Check this machine's environments and weights, without a card.")
    parser.add_argument("--row", help="rows, separated by commas (default: every row)")
    parser.add_argument("--block", help="blocks, separated by commas (default: every block)")
    args = parser.parse_args(argv)
    rows = args.row.split(",") if args.row else layout.rows()
    blocks = args.block.split(",") if args.block else layout.blocks()
    if args.row and not args.block:
        blocks = []
    if args.block and not args.row:
        rows = []
    failed = 0
    for title, checks in check(rows, blocks):
        print(title)
        for good, line in checks:
            failed += not good
            print("  %s %s" % ("ok  " if good else "FAIL", line))
    print("%d checks failed" % failed if failed else "all checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
