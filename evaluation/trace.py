"""The per-step trace of one episode: written during the rollout, read by the Reach labeller.

The arrays are listed in `schema.TRACE_ARRAYS`. The trace holds every body's pose at every step and every
contact of the gripper, with the model's own tables, so that any rule about where the hand went can be
evaluated from the file alone, without the simulator.

Python 3.8 and later, NumPy only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from evaluation import schema

START, SETTLE, POLICY = 0, 1, 2  # values of "phase"

#: The model's tables a scene must provide once per episode (see schema.TRACE_ARRAYS).
TABLES = (
    "body_name", "body_parent", "geom_name", "geom_body", "geom_pos", "geom_size", "geom_type", "geom_collides",
    "entity_name", "entity_kind", "gripper_geoms",
)  # fmt: skip
_PER_FRAME = ("eef_pos", "eef_quat", "gripper_qpos", "joint_pos", "body_pos", "body_quat")


class TraceWriter:
    def __init__(self, meta: Dict[str, Any], tables: Dict[str, Any]) -> None:
        missing = [name for name in TABLES if name not in tables]
        if missing:
            raise ValueError("the scene's tables lack %s" % ", ".join(missing))
        self._meta = dict(meta, schema=schema.SCHEMA)
        self._tables = {name: np.asarray(tables[name]) for name in TABLES}
        self._frames: Dict[str, List[Any]] = {name: [] for name in ("phase", "action", "predicate") + _PER_FRAME}
        self._contacts: Dict[str, List[Any]] = {name: [] for name in ("frame", "gripper_geom", "other_geom", "distance")}
        self._queries: Dict[str, List[Any]] = {"frame": [], "execute": [], "actions": []}
        self._records: Dict[str, List[np.ndarray]] = {}

    @property
    def frames(self) -> int:
        return len(self._frames["phase"])

    def frame(self, phase: int, action: Any, state: Dict[str, Any], predicate: bool) -> None:
        """One frame. `state` is `Scene.frame()`: the per-frame arrays and "contacts", a list of
        (gripper geom, other geom, distance)."""
        index = self.frames
        self._frames["phase"].append(phase)
        self._frames["action"].append(np.full(7, np.nan, np.float32) if action is None else np.asarray(action, np.float32))
        self._frames["predicate"].append(bool(predicate))
        for name in _PER_FRAME:
            self._frames[name].append(np.asarray(state[name], np.float32))
        for gripper_geom, other_geom, distance in state["contacts"]:
            self._contacts["frame"].append(index)
            self._contacts["gripper_geom"].append(int(gripper_geom))
            self._contacts["other_geom"].append(int(other_geom))
            self._contacts["distance"].append(float(distance))

    def query(self, execute: int, actions: np.ndarray, record: Dict[str, np.ndarray]) -> None:
        """One query, taken at the latest frame."""
        self._queries["frame"].append(self.frames - 1)
        self._queries["execute"].append(int(execute))
        self._queries["actions"].append(np.asarray(actions, np.float32))
        for name, array in (record or {}).items():
            self._records.setdefault(name, []).append(np.asarray(array, np.float32))

    def arrays(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {"meta": np.asarray(json.dumps(self._meta, sort_keys=True, ensure_ascii=False))}
        out["phase"] = np.asarray(self._frames["phase"], np.uint8)
        out["predicate"] = np.asarray(self._frames["predicate"], bool)
        for name in ("action",) + _PER_FRAME:
            out[name] = np.stack(self._frames[name]).astype(np.float32)
        out["contact_frame"] = np.asarray(self._contacts["frame"], np.int32)
        out["contact_gripper_geom"] = np.asarray(self._contacts["gripper_geom"], np.int32)
        out["contact_other_geom"] = np.asarray(self._contacts["other_geom"], np.int32)
        out["contact_distance"] = np.asarray(self._contacts["distance"], np.float32)
        out.update(self._tables)
        out["query_frame"] = np.asarray(self._queries["frame"], np.int32)
        out["query_execute"] = np.asarray(self._queries["execute"], np.int32)
        chunks = self._queries["actions"]
        if chunks and len({chunk.shape for chunk in chunks}) == 1:
            out["query_actions"] = np.stack(chunks)
        else:  # chunks of different lengths: padded with NaN to the longest
            longest = max([chunk.shape[0] for chunk in chunks], default=0)
            padded = np.full((len(chunks), longest, 7), np.nan, np.float32)
            for i, chunk in enumerate(chunks):
                padded[i, : chunk.shape[0]] = chunk
            out["query_actions"] = padded
        for name, arrays in self._records.items():
            if len(arrays) != len(chunks):
                raise ValueError("the method returned %r at %d of %d queries" % (name, len(arrays), len(chunks)))
            out["query/" + name] = np.stack(arrays)
        return out

    def save(self, path: Path) -> str:
        """Writes the trace and returns the file's sha256."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        with open(temporary, "wb") as handle:
            np.savez_compressed(handle, **self.arrays())
        temporary.replace(path)
        return file_sha256(path)


def load(path: Path) -> Dict[str, np.ndarray]:
    with np.load(str(path), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def meta(trace: Dict[str, np.ndarray]) -> Dict[str, Any]:
    return json.loads(str(trace["meta"]))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
