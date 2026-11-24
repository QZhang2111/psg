"""Where things are: block definitions, rows, environments, and the one output directory.

    evaluation/blocks/<block>/block.json        the test set of an evaluation block (git)
    evaluation/rows/<row>/row.json              a row: its model, its environment's name (git)
    evaluation/environments/<name>.json         how an environment is built (git)
    environments.local.json              this machine's interpreters, {environment name: python} (not in git)
    results/<block>/<row>/               config.json, episodes.csv of a finished unit (git)
    outputs/<block>/<row>/               what a run writes (not in git); the same below any output root:
        shards/config.<k>of<n>.json          the config record of one shard
        shards/episodes.<k>of<n>.csv         its episode table, appended episode by episode
        raw/traces/<episode>.npz             per-episode trace
        raw/videos/<episode>.mp4             per-episode main-camera video
        raw/logs/                            process logs and the host's own facts (name, GPU, times)
        config.json, episodes.csv            the unit, written by `collect` when every shard is complete

Python 3.8 and later, standard library only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
BLOCKS = REPO / "evaluation" / "blocks"
ROWS = REPO / "evaluation" / "rows"
ENVIRONMENTS = REPO / "evaluation" / "environments"
RESULTS = REPO / "results"
LOCAL_ENVIRONMENTS = "environments.local.json"


def blocks() -> List[str]:
    return sorted(p.parent.name for p in BLOCKS.glob("*/block.json"))


def rows() -> List[str]:
    return sorted(p.parent.name for p in ROWS.glob("*/row.json"))


def load_block(block: str) -> Dict[str, Any]:
    path = BLOCKS / block / "block.json"
    if not path.exists():
        raise SystemExit("no evaluation block %r; defined: %s" % (block, ", ".join(blocks()) or "none"))
    definition = json.loads(path.read_text())
    if definition.get("block") != block:
        raise SystemExit("%s names itself %r" % (path, definition.get("block")))
    return definition


def load_row(row: str) -> Dict[str, Any]:
    path = ROWS / row / "row.json"
    if not path.exists():
        raise SystemExit("no row %r; defined: %s" % (row, ", ".join(rows()) or "none"))
    definition = json.loads(path.read_text())
    if definition.get("row") != row:
        raise SystemExit("%s names itself %r" % (path, definition.get("row")))
    return definition


def output_root(given: Optional[str] = None) -> Path:
    """The output root: the argument, else $PSG_OUTPUT, else `outputs/` in the checkout."""
    return Path(given or os.environ.get("PSG_OUTPUT") or REPO / "outputs").expanduser().resolve()


def unit_dir(root: Path, block: str, row: str, smoke: bool = False) -> Path:
    """A smoke writes under `_smoke/`, where no tool collects from."""
    return (root / "_smoke" if smoke else root) / block / row


def shard_name(shard: Tuple[int, int]) -> str:
    return "%dof%d" % shard


def interpreter(environment: str) -> str:
    """The Python of a named environment on this machine.

    Read from $PSG_ENVIRONMENTS or `environments.local.json` in the checkout: {"<name>": "<python>"}.
    The repository names environments and says how each is built (`evaluation/environments/<name>.json`); only this
    file, which is not in git, says where one is on a machine.
    """
    path = Path(os.environ.get("PSG_ENVIRONMENTS") or REPO / LOCAL_ENVIRONMENTS)
    table = json.loads(path.read_text()) if path.exists() else {}
    if environment not in table:
        raise SystemExit(
            "environment %r has no interpreter on this machine.\n"
            "  how it is built: evaluation/environments/%s.json\n"
            "  then add {\"%s\": \"<its python>\"} to %s" % (environment, environment, environment, path)
        )
    entry = table[environment]
    return str(entry["python"] if isinstance(entry, dict) else entry)


def with_declared_flags(given: str, declared: str) -> str:
    """`XLA_FLAGS` as given by a machine or a caller, with the declared flags in force. Flags are separated by
    spaces and compared as flags, not as parts of the string.

    The other flags are kept, in their order. A declared flag stands once: where it is already given it keeps
    its place; where it is missing it is added after the others. The same flag given with another value
    (`--xla_gpu_deterministic_ops=false`) is dropped, so that the declared value is the only one."""
    wanted = declared.split()
    names = {flag.split("=", 1)[0] for flag in wanted}
    kept: List[str] = []
    for flag in given.split():
        if flag in wanted:
            if flag not in kept:
                kept.append(flag)
        elif flag.split("=", 1)[0] not in names:
            kept.append(flag)
    return " ".join(kept + [flag for flag in wanted if flag not in kept])


def process_environment(environment: str, base: Dict[str, str]) -> Dict[str, str]:
    """The variables of a process started in a named environment.

    PYTHONPATH: the checkout, then the paths the environment's declaration lists under "pythonpath" (relative
    to the checkout; no environment lists one now: `openpi` is installed in its environment). Then the
    variables this machine sets for the environment: its entry in `environments.local.json` may be
    {"python": "<its python>", "env": {"NAME": "value"}} for what only a machine knows (where the benchmark's
    files and the weights' cache are, which renderer). Last, the variables the environment's declaration lists
    under "variables": they belong to how its rows are run, are the same on every machine, and win over the
    machine's (`evaluation/environments/openpi.json` asks XLA for deterministic GPU operations, so that an episode repeats
    from launch to launch). `XLA_FLAGS` is a list of flags: a machine's or a caller's other flags are kept beside
    the declaration's, and the same flag given with another value gives way to the declared one
    (`with_declared_flags`).
    """
    declaration_path = ENVIRONMENTS / (environment + ".json")
    declaration = json.loads(declaration_path.read_text()) if declaration_path.exists() else {}
    paths = [str(REPO)] + [str(REPO / extra) for extra in declaration.get("pythonpath", [])]
    out = dict(base)
    local_path = Path(os.environ.get("PSG_ENVIRONMENTS") or REPO / LOCAL_ENVIRONMENTS)
    local = json.loads(local_path.read_text()) if local_path.exists() else {}
    entry = local.get(environment)
    if isinstance(entry, dict):
        out.update({str(name): str(value) for name, value in (entry.get("env") or {}).items()})
    for name, value in (declaration.get("variables") or {}).items():
        if name == "XLA_FLAGS":
            value = with_declared_flags(out.get("XLA_FLAGS") or "", str(value))
        out[str(name)] = str(value)
    # The checkout's own paths come first; a PYTHONPATH of the machine's entry (a benchmark's code that is not
    # installed in the environment) or of the caller follows.
    out["PYTHONPATH"] = os.pathsep.join(paths + ([out["PYTHONPATH"]] if out.get("PYTHONPATH") else []))
    return out


def launch_card(base: Dict[str, str]) -> Optional[str]:
    """The card a launch was given: the caller's CUDA_VISIBLE_DEVICES when it names exactly one card by its
    index ("3"). None when it is unset, lists several cards, or names a card another way (a UUID)."""
    value = (base.get("CUDA_VISIBLE_DEVICES") or "").strip()
    return value if value.isdigit() else None


def simulator_environment(environment: str, base: Dict[str, str]) -> Dict[str, str]:
    """The variables of a simulator process: `process_environment`, with the renderer on the launch's card.

    A launch is given its card by the caller's CUDA_VISIBLE_DEVICES; that places the model. The simulator
    renders through EGL on the device MUJOCO_EGL_DEVICE_ID names, and a machine file can only name one device
    for all launches, which would put every launch's simulator on the same card of a machine with several.
    So when the launch has one card and the renderer is EGL, MUJOCO_EGL_DEVICE_ID is set here to that card's
    EGL device, whatever the machine's entry says:

    - the card's own index, which is how the simulator (robosuite) itself reads CUDA_VISIBLE_DEVICES;
    - or, where a machine's EGL numbers its cards in another order than CUDA does, the device its entry gives:
      {"python": ..., "env": {...}, "egl_device_of_card": {"0": 2, "1": 0}}. The simulator is then started
      without CUDA_VISIBLE_DEVICES, because robosuite refuses a device index that is not among its cards; the
      simulator computes nothing on a card besides the image.

    Without a single card (CUDA_VISIBLE_DEVICES unset, or several cards) nothing is changed: the machine's
    entry holds, as before. On a machine with one card the variables are the same as before this rule.

    Which card renders does not enter an episode's definition: the scene, the camera and the renderer's
    settings are the same. Whether two cards give the same image bit for bit is a fact of the machine, shown
    by comparing trace digests, not assumed here.
    """
    out = process_environment(environment, base)
    card = launch_card(base)
    if card is None or out.get("MUJOCO_GL", "egl").strip().lower() not in ("egl", ""):
        return out
    local_path = Path(os.environ.get("PSG_ENVIRONMENTS") or REPO / LOCAL_ENVIRONMENTS)
    local = json.loads(local_path.read_text()) if local_path.exists() else {}
    entry = local.get(environment)
    table = entry.get("egl_device_of_card") if isinstance(entry, dict) else None
    device = str((table or {}).get(card, card)).strip()
    if not device.isdigit():
        raise SystemExit(
            "%s: egl_device_of_card of %r gives %r for card %s; an EGL device is a number"
            % (local_path, environment, device, card)
        )
    out["MUJOCO_EGL_DEVICE_ID"] = device
    if device != card:
        out.pop("CUDA_VISIBLE_DEVICES", None)
    return out


def parse_ranges(text: Optional[str]) -> Optional[List[int]]:
    """'0-4,7' -> [0, 1, 2, 3, 4, 7]; None stays None (everything)."""
    if text is None or text == "":
        return None
    out: List[int] = []
    for part in text.split(","):
        low, _, high = part.partition("-")
        out.extend(range(int(low), int(high or low) + 1))
    return sorted(set(out))


def parse_shard(text: str) -> Tuple[int, int]:
    k, _, n = text.partition("of")
    shard = (int(k), int(n))
    if not 0 <= shard[0] < shard[1]:
        raise SystemExit("--shard is <k>of<n> with 0 <= k < n, got %r" % text)
    return shard
