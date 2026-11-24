"""The oracle channel: privileged simulator state for a row whose own definition asks for it.

No row of the paper's tables asks: for them `Observation.oracle` is None and nothing here runs. A row that is an
instrument (`instruments/`) names in its `row.json` what it needs (`"oracle": [<kind>, ...]`); its
adapter repeats that in its report (`requires.oracle`); the evaluation then puts the named state into every
observation and writes into the unit's config record (`observation.oracle`) what was passed and that it is an
oracle. A unit that received an oracle is no measurement of a policy: the model was told where a thing is.

Kinds:

    reach_target_position   the world position (m) of the body of the task's Reach target, at the query's frame

The body of a target is chosen by `target_body` below, from the scene's tables alone, so that what reads a
trace later finds the same body as the simulator did.

Python 3.8 and later, NumPy only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from evaluation import reach as _reach

KINDS: Dict[str, str] = {
    "reach_target_position": "the world position (m) of the body of the task's Reach target at the query's frame: "
    "the body named <target>_main, or the first body of the target's entity where there is none of that name",
}


def requested(report: Dict[str, Any]) -> List[str]:
    """The kinds a row's report asks for (`requires.oracle`); empty for every row that does not ask."""
    kinds = list((report.get("requires") or {}).get("oracle") or [])
    unknown = [kind for kind in kinds if kind not in KINDS]
    if unknown:
        raise SystemExit("the row asks for an oracle this evaluation does not have: %s" % ", ".join(unknown))
    return kinds


def target_body(tables: Dict[str, Any], reach: Optional[Dict[str, Any]]) -> Tuple[str, str, int]:
    """(the target as the block names it, the body's name, the body's index) for a task's `reach` entry.

    `tables` are a scene's tables or a trace (both hold `body_name`, `entity_name`, `geom_body`, `geom_pos`,
    `gripper_geoms`). The task must name exactly one target. The entity's bodies are found by the labeller's own
    rule (`reach.Model`); of them the one named `<entity>_main`, else the first in the model's order; where the
    target names a part (`wooden_cabinet_1:middle`), only bodies with the part's word after the entity's name.
    """
    targets = list((reach or {}).get("target") or [])
    if len(targets) != 1:
        raise ValueError("the oracle needs a task with exactly one Reach target; this one names %d" % len(targets))
    entity, _, part = targets[0].partition(":")
    model = _reach.Model(tables)
    own = [
        index
        for index, owner in enumerate(model.body_entity)
        if owner == entity and (not part or part in model.body_name[index][len(entity) :])
    ]
    if not own:
        raise ValueError("no body of the scene belongs to the Reach target %r" % targets[0])
    main = [index for index in own if model.body_name[index] == entity + "_main"]
    index = (main or own)[0]
    return targets[0], model.body_name[index], index


def check_tasks(tasks: List[Dict[str, Any]]) -> None:
    """Before any episode: every task about to run names exactly one Reach target."""
    for task in tasks:
        if len(list((task.get("reach") or {}).get("target") or [])) != 1:
            raise SystemExit(
                "task %s (%s) does not name exactly one Reach target; a row that asks for the oracle "
                "'reach_target_position' runs only on tasks that do" % (task.get("index"), task.get("task"))
            )


def note(kinds: List[str]) -> Dict[str, Any]:
    """What the config record says under `observation.oracle` for a unit that received an oracle."""
    return {
        "is": "an oracle: privileged simulator state in every observation, asked for by the row's own definition "
        "(row.json: oracle). No row of the paper's tables receives one; this unit is no measurement of a policy",
        "passed": {kind: KINDS[kind] for kind in kinds},
    }
