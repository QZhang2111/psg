"""The three things a unit leaves behind, in one fixed form for every block and every row.

A unit is one row on one evaluation block. It leaves:

1. the config record (`config.json`): written by the program, never by hand; goes into git;
2. the episode table (`episodes.csv`): one row per episode, the columns below; goes into git;
3. per episode, a trace (`.npz`) and a main-camera video (`.mp4`): stay on the host, outside git.

Python 3.8 and later, standard library only.
"""

from __future__ import annotations

from typing import Any, Dict, List

#: Version of this file's formats. It is written into every config record and trace. It keeps the repository's
#: earlier package name, `pi05lab`, because the 131 config records in `results/` carry it and a new record of the
#: same format must say the same.
SCHEMA = "pi05lab-2"

# --------------------------------------------------------------------------------------------------------------
# 1. The episode table
# --------------------------------------------------------------------------------------------------------------

#: Columns of `episodes.csv`, in order. The same for every block and row; a value that does not apply is empty.
EPISODE_COLUMNS = [
    "episode",  # "<block>.<row>.t<task_index>.s<state>.n<seed>"
    "block",
    "row",
    "task_index",  # position of the task in the block's definition
    "task",  # the task's name in its benchmark
    "state",  # index of the start state
    "seed",  # the noise seed
    "instruction",  # the text the model was given
    "success",  # 1 or 0: the block's predicate held before the step limit
    "success_step",  # policy step at which it first held
    "steps",  # policy steps executed (settling steps not counted)
    "queries",  # times the model was asked for actions
    "stopped",  # "predicate" | "limit" | "error"
    "reach",  # 1 or 0, by the task's rule and target; empty when the block defines no Reach
    "reach_rule",  # "first_close" | "first_contact", from the block's definition
    "reach_target",  # the target's body-name prefixes, joined by "|"
    "first_close_step",  # policy step of the first open -> close change of the executed gripper command
    "first_close_entity",  # the object or fixture that close is on (the labeller's rule), or "none"
    "first_close_body",  # the body of that entity nearest the hand at that step
    "first_close_distance_m",  # hand to that body's nearest geom centre
    "first_contact_step",  # first policy step at which a gripper geom touches an object or a fixture
    "first_contact_entity",  # the object or fixture touched: a cabinet with its frame and drawers is one entity
    "first_contact_body",  # the body touched, as the simulator names it
    "first_part_step",  # where the target names a part of a fixture: the first contact with a body of that kind
    "first_part_body",  # (a drawer), contacts with the rest of the fixture skipped
    "labeller",  # revision of the Reach labeller that wrote the columns from "reach" on
    "trace_sha256",  # digest of the episode's trace file, so the table and the raw record can be matched
    "error",  # empty unless the episode failed; then "stopped" is "error" and success and reach are empty
]

# --------------------------------------------------------------------------------------------------------------
# 2. The trace of one episode (npz). T + 1 frames: frame 0 is the start state, frame i follows the i-th
#    simulator step (settling steps first). Q queries.
# --------------------------------------------------------------------------------------------------------------

TRACE_ARRAYS: Dict[str, str] = {
    "meta": "JSON text: schema, episode, block, row, task, state, seed, instruction, settle_steps, and the model "
    "tables' names below",
    # per frame
    "phase": "uint8 [T+1]: 0 start state, 1 settling step, 2 policy step",
    "action": "float32 [T+1, 7]: the action executed to reach this frame (NaN at frame 0)",
    "eef_pos": "float32 [T+1, 3]: gripper site position, world, m",
    "eef_quat": "float32 [T+1, 4]: gripper orientation (x, y, z, w)",
    "gripper_qpos": "float32 [T+1, 2]: finger joint positions",
    "joint_pos": "float32 [T+1, 7]: arm joint positions",
    "body_pos": "float32 [T+1, B, 3]: world position of every body of the model, fixtures and robot included",
    "body_quat": "float32 [T+1, B, 4]: world orientation of every body (w, x, y, z as the simulator stores it)",
    "predicate": "bool [T+1]: the block's success predicate at this frame",
    # contacts: one entry per (frame, gripper geom, other geom) in touch; the robot's own geoms are left out
    "contact_frame": "int32 [C]",
    "contact_gripper_geom": "int32 [C]: geom id on the gripper",
    "contact_other_geom": "int32 [C]: geom id of what it touches",
    "contact_distance": "float32 [C]: signed distance (negative = penetration)",
    # the model's tables, constant over the episode: enough to place every geom from the body poses
    "body_name": "str [B]",
    "body_parent": "int32 [B]",
    "geom_name": "str [G]",
    "geom_body": "int32 [G]: the body each geom is fixed to",
    "geom_pos": "float32 [G, 3]: geom centre in its body's frame",
    "geom_size": "float32 [G, 3]",
    "geom_type": "int32 [G]",
    "geom_collides": "bool [G]: the geom takes part in collisions (contype or conaffinity not 0)",
    "entity_name": "str [E]: the scene's objects and fixtures, as the benchmark names them",
    "entity_kind": "str [E]: 'object' | 'fixture'",
    "gripper_geoms": "int32 [Ng]: the geoms that count as the gripper",
    # per query
    "query_frame": "int32 [Q]: the frame the query's observation was taken at",
    "query_execute": "int32 [Q]: how many of the returned actions were executed",
    "query_actions": "float32 [Q, n, 7]: the returned chunk",
    "query/<name>": "float32 [Q, ...]: what the method returned for the record (PSG: v, v_g, v_d, each "
    "[Q, Euler steps, horizon, 7])",
}

# --------------------------------------------------------------------------------------------------------------
# 3. The config record
# --------------------------------------------------------------------------------------------------------------

#: The record's sections and what each holds. Sections 1-8 are written by the evaluation from the block's
#: definition and from what it measured when it loaded the files; 9-11 are the adapter's report, unchanged.
#: Scientific settings only: no host name, no path, no time.
CONFIG_RECORD: Dict[str, str] = {
    "schema": "the SCHEMA above",
    "unit": "{block, row}",
    "code": "{revision: this repository's commit, clean: bool}; a table run refuses a checkout with changes",
    # the evaluation's side (the table's fields 1-8)
    "benchmark": "1. {name, code: {address, revision}, files: {address, revision}, suite, perturbation, "
    "simulator: {environment, packages: {name: version}, renderer}}",
    "tasks": "2-4, 7, 8. one entry per task: {index, task, bddl_sha256, start_states: [indices], "
    "start_states_sha256, instruction, reach: {rule, target}, and the block's own columns (Layout: family, "
    "moves_target)}; the hashes are of the files as loaded",
    "seeds": "5. {noise: [..], environment: n, what_the_environment_seed_changes: text}",
    "rollout": "6. {settle_steps, settle_action, max_steps, stop, retry}",
    "observation": "what crosses the interface: cameras, render size, state fields, camera parameters",
    "endpoints": "8. {success: the predicate's definition, labeller: {revision, parameters}}",
    # the adapter's side (fields 9-11), from Adapter.report()
    "model": "9. {name, checkpoint: {address, revision, sha256, licence}, code: {address, revision}}",
    "inference": "10. every inference setting in effect",
    "method": "11. {name, code, settings}",
    "noise_seed_controls": "what the noise seed determines for this row",
    "model_environment": "{name, python, packages}",
    # what was asked for
    "selection": "{tasks, states, shard}: the part of the unit this record covers; 'all' for a whole unit",
}

#: Sections that may differ between the shards of one unit. Every other section must be equal, or the unit is
#: refused when its shards are collected.
SHARD_SECTIONS = ("selection",)


def record_differences(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    """The sections in which two shards' records of one unit disagree (empty: they describe the same unit)."""
    keys = sorted((set(a) | set(b)) - set(SHARD_SECTIONS))
    return [key for key in keys if a.get(key) != b.get(key)]


def episode_id(block: str, row: str, task_index: int, state: int, seed: int) -> str:
    return "%s.%s.t%d.s%d.n%d" % (block, row, task_index, state, seed)
