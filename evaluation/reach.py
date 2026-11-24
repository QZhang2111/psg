"""The Reach labeller: where the hand first went, from an episode's trace, the task's rule and the task's target.

It reads nothing else: no simulator, no model, no image. So Reach can be written again from the traces when a
rule changes, and a rule is a few lines here with its parameters in the config record.

Three events are read from every trace:

- **the first close**: the first policy step whose executed gripper command changes from open to close. It is
  "on" the object or fixture that has the geom whose centre is nearest the hand at that step.
- **the first contact**: the first policy step at which a geom of the gripper touches a geom of an object or
  fixture (the table and the floor are not counted). It is written twice: the entity touched (a cabinet with
  its frame and all its drawers is one entity; a stove with its burner and button likewise) and the body as the
  simulator names it.
- **the first part**, only where the task's target names a part of a fixture (`wooden_cabinet_1:middle`): the
  first body *of that kind* the gripper touches, contacts with the rest of the fixture (its frame) skipped. The
  kind is read from the body's name: the bodies of the entity named like the target's body with another word
  in the part's place (`..._cabinet_top`, `..._cabinet_middle`, `..._cabinet_bottom`).

Reach is then, by the task's rule:

- `first_close`: the first close is on the target (for a part: the nearest body is the named part).
- `first_contact`: the first entity touched is the target's entity, and, where the target names a part, the
  first part touched is the named one. Touching the right cabinet's frame and then the right drawer is a hit;
  the right cabinet and the wrong drawer is a miss, with the entity still saying "cabinet".

Python 3.8 and later, NumPy only.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from evaluation import trace as _trace

REVISION = "reach-2"

#: The labeller's parameters; they go into the config record with the revision.
PARAMETERS: Dict[str, Any] = {
    "close": "the first policy step whose executed gripper command is > 0 after a command <= 0 (the gripper "
    "starts open)",
    "close_is_on": "the object or fixture owning the geom whose centre is nearest the gripper site at the frame "
    "after that step; every geom of every body of the entity counts",
    "close_max_distance_m": None,  # no limit: the nearest entity is the label however far it is
    "contact": "the first policy step with a gripper geom in contact with a geom of an object or fixture",
    "contact_max_distance_m": 0.0,  # signed distance of the contact; 0 = touching or penetrating
    "contact_ignores": ["table", "floor"],  # entities whose name contains one of these are not counted
    "part": "where the target names a part: the first contact with a body of the target's entity named like the "
    "target's body with another word in the part's place; the entity's other bodies (its frame) are skipped",
    "reach_first_contact": "the first entity touched is the target's entity and, where a part is named, the "
    "first part touched is the named one",
    "entity_bodies": "the bodies whose name is the entity's name or starts with it and an underscore; a body "
    "matching several entities belongs to the longest name",
}


def _quat_rotate(quat_wxyz: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Rotate `vectors` [n, 3] by unit quaternions [n, 4] given as (w, x, y, z)."""
    w, xyz = quat_wxyz[:, :1], quat_wxyz[:, 1:]
    t = 2.0 * np.cross(xyz, vectors)
    return vectors + w * t + np.cross(xyz, t)


class Model:
    """The scene's tables of one trace: which body and geom belongs to which entity."""

    def __init__(self, trace: Dict[str, np.ndarray]) -> None:
        self.body_name = [str(name) for name in trace["body_name"]]
        self.geom_body = np.asarray(trace["geom_body"], int)
        self.geom_pos = np.asarray(trace["geom_pos"], float)
        self.entities = [str(name) for name in trace["entity_name"]]
        self.body_entity: List[Optional[str]] = [self._entity_of(name) for name in self.body_name]
        self.geom_entity = [self.body_entity[body] for body in self.geom_body]
        self.gripper_geoms = set(int(geom) for geom in trace["gripper_geoms"])

    def _entity_of(self, body: str) -> Optional[str]:
        owners = [entity for entity in self.entities if body == entity or body.startswith(entity + "_")]
        return max(owners, key=len) if owners else None

    def geom_centres(self, body_pos: np.ndarray, body_quat: np.ndarray) -> np.ndarray:
        """World position of every geom centre at one frame, from the bodies' poses: [G, 3]."""
        return body_pos[self.geom_body] + _quat_rotate(body_quat[self.geom_body], self.geom_pos)

    def parts_like(self, target: str) -> Set[str]:
        """The bodies of the kind a part target names: for `wooden_cabinet_1:middle` the cabinet's drawers.

        They are the entity's bodies named like the target's body with another word in the part's place. Where
        the body's name gives no such kind (the part's word follows the entity's name directly), the kind is the
        named part alone.
        """
        entity, _, part = target.partition(":")
        if not part:
            return set()
        own = [body for body, owner in zip(self.body_name, self.body_entity) if owner == entity]
        kind: Set[str] = set()
        for body in own:
            if part not in body[len(entity) :]:
                continue
            stem = body[: body.rfind(part)]
            if stem.rstrip("_") == entity:
                kind.add(body)
            else:
                kind.update(other for other in own if other.startswith(stem))
        return kind


def policy_frames(trace: Dict[str, np.ndarray]) -> np.ndarray:
    """Frame indices of the policy steps, in order; the k-th entry is policy step k."""
    return np.flatnonzero(np.asarray(trace["phase"]) == _trace.POLICY)


def first_close(trace: Dict[str, np.ndarray], model: Model, max_distance_m: Optional[float] = None) -> Dict[str, Any]:
    frames = policy_frames(trace)
    commands = np.asarray(trace["action"])[frames, 6]
    previous = np.concatenate([[-1.0], commands[:-1]])
    closing = np.flatnonzero((commands > 0) & (previous <= 0))
    if closing.size == 0:
        return {"step": None, "entity": None, "body": None, "distance_m": None}
    step = int(closing[0])
    frame = int(frames[step])
    centres = model.geom_centres(np.asarray(trace["body_pos"][frame], float), np.asarray(trace["body_quat"][frame], float))
    distance = np.linalg.norm(centres - np.asarray(trace["eef_pos"][frame], float), axis=1)
    owned = np.asarray([entity is not None for entity in model.geom_entity])
    if not owned.any():
        return {"step": step, "entity": "none", "body": None, "distance_m": None}
    geom = int(np.flatnonzero(owned)[np.argmin(distance[owned])])
    if max_distance_m is not None and distance[geom] > max_distance_m:
        return {"step": step, "entity": "none", "body": None, "distance_m": float(distance[geom])}
    body = int(model.geom_body[geom])
    return {"step": step, "entity": model.geom_entity[geom], "body": model.body_name[body], "distance_m": float(distance[geom])}


def _first_touch(
    trace: Dict[str, np.ndarray], model: Model, counts: Callable[[Optional[str], str], bool], max_distance_m: float
) -> Dict[str, Any]:
    """The first policy step at which a gripper geom touches a geom for which `counts(entity, body)` holds; of
    several at that step, the deepest."""
    frames = policy_frames(trace)
    step_of_frame = {int(frame): step for step, frame in enumerate(frames)}
    best: Optional[Tuple[int, float, int]] = None  # (frame, distance, other geom)
    for frame, gripper_geom, other_geom, distance in zip(
        trace["contact_frame"], trace["contact_gripper_geom"], trace["contact_other_geom"], trace["contact_distance"]
    ):
        frame, other_geom, distance = int(frame), int(other_geom), float(distance)
        if frame not in step_of_frame or int(gripper_geom) not in model.gripper_geoms or distance > max_distance_m:
            continue
        if not counts(model.geom_entity[other_geom], model.body_name[int(model.geom_body[other_geom])]):
            continue
        if best is None or (frame, distance) < best[:2]:
            best = (frame, distance, other_geom)
    if best is None:
        return {"step": None, "entity": None, "body": None}
    body = int(model.geom_body[best[2]])
    return {"step": step_of_frame[best[0]], "entity": model.geom_entity[best[2]], "body": model.body_name[body]}


def first_contact(
    trace: Dict[str, np.ndarray], model: Model, max_distance_m: float = 0.0, ignores: Sequence[str] = ("table", "floor")
) -> Dict[str, Any]:
    """The first object or fixture the gripper touches: its entity and the body touched."""

    def counts(entity: Optional[str], body: str) -> bool:
        return entity is not None and not any(word in entity for word in ignores)

    return _first_touch(trace, model, counts, max_distance_m)


def first_part(
    trace: Dict[str, np.ndarray], model: Model, targets: Sequence[str], max_distance_m: float = 0.0
) -> Dict[str, Any]:
    """The first body of the targets' named kind the gripper touches (a cabinet's first drawer), the rest of
    the fixture skipped. Empty when no target names a part."""
    kind: Set[str] = set()
    for target in targets:
        kind |= model.parts_like(target)
    if not kind:
        return {"step": None, "entity": None, "body": None}
    return _first_touch(trace, model, lambda entity, body: body in kind, max_distance_m)


def is_target(entity: Optional[str], body: Optional[str], targets: Sequence[str]) -> bool:
    for target in targets:
        name, _, part = target.partition(":")
        if entity == name and (not part or (body is not None and part in body)):
            return True
    return False


def reached_by_first_contact(contact: Dict[str, Any], part: Dict[str, Any], targets: Sequence[str]) -> bool:
    """The first entity touched is a target's entity; where that target names a part, the first part touched
    is the named one."""
    for target in targets:
        name, _, named = target.partition(":")
        if contact["entity"] != name:
            continue
        if not named or (part["body"] is not None and named in part["body"][len(name) :]):
            return True
    return False


def label(trace: Dict[str, np.ndarray], reach: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The episode table's columns from "reach" to "labeller" for one trace.

    `reach` is the task's `{"rule", "target"}` from the block's definition, or None where the block has no Reach
    (the events are still written).
    """
    model = Model(trace)
    targets = list(reach["target"]) if reach else []
    close = first_close(trace, model, PARAMETERS["close_max_distance_m"])
    contact = first_contact(trace, model, PARAMETERS["contact_max_distance_m"], PARAMETERS["contact_ignores"])
    part = first_part(trace, model, targets, PARAMETERS["contact_max_distance_m"])
    row: Dict[str, Any] = {
        "reach": "", "reach_rule": "", "reach_target": "",
        "first_close_step": close["step"], "first_close_entity": close["entity"], "first_close_body": close["body"],
        "first_close_distance_m": None if close["distance_m"] is None else round(close["distance_m"], 4),
        "first_contact_step": contact["step"], "first_contact_entity": contact["entity"],
        "first_contact_body": contact["body"],
        "first_part_step": part["step"], "first_part_body": part["body"],
        "labeller": REVISION,
    }  # fmt: skip
    if reach:
        rule = reach["rule"]
        if rule == "first_close":
            reached = is_target(close["entity"], close["body"], targets)
        elif rule == "first_contact":
            reached = reached_by_first_contact(contact, part, targets)
        else:
            raise ValueError("unknown Reach rule %r" % rule)
        row.update(reach=int(reached), reach_rule=rule, reach_target="|".join(targets))
    return {key: ("" if value is None else value) for key, value in row.items()}
