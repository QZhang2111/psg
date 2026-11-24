"""An instrument, not the method: PSG's grounding term replaced by a push along the true direction to the object.

    v~ = v + v_o - b * v_d          on the motion dimensions (0-5); the gripper dimension keeps v

v is the released policy's velocity and v_d = v - v[A -/-> l] the direct contribution, with PSG's own cut and
b = 1 (`psg/sampler.py`). PSG's grounding term a * v_g is not evaluated. In its place stands v_o: a
velocity that is the same at every Euler step and for every action of the chunk, non-zero only in the two
horizontal translation dimensions, and **defined by its effect on the actions**:

    t      the horizontal vector (m, world) from the gripper to the requested object at the query: the position
           of the Reach target's body, which the simulator tells the row (the oracle channel,
           `evaluation/oracle.py`), minus the gripper site's
    shift  = size * min(1, |t| / guard radius) * t / |t|
           the horizontal shift added to every action of the chunk, in executed-action units, toward the object.
           `size` is the row's setting. Inside the guard radius, where the direction means little, the shift
           falls linearly to zero.
    v_o    = -shift / (output scale), dimension by dimension. The sampler integrates x <- x + dt * v~ with
           dt = -1 / (Euler steps): a velocity held over all steps moves the sample by minus itself, and the
           policy's output scale turns the sample's units into the executed action's. A push along +u in v
           would move the action along -u; the sign is taken here, once.

The shift is in the action's own axes; the simulator's controller takes translation increments in the world's
axes. On this row's own traces, the readout behind the paper's direction panel (no longer in this repository) read
v_o and gave cos = +1 at every query outside the exact centre.

This row is told where the object is. It measures what a correct direction of the given size does to Reach; it
is no policy and no row of the paper's tables.

This module is the definition and needs NumPy only; the sampler is `oracle_push_sampler.py` (JAX).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

KIND = "reach_target_position"  # what the row asks of the oracle channel
METHOD = "oracle-push"
GUARD_RADIUS_M = 0.05  # chosen once, not tuned
RECORDED = ("v", "v_o", "v_d")  # what the sampler leaves for the record at every Euler step, in this order


def settings(row: Dict[str, Any]) -> Dict[str, float]:
    """The push's two settings from the row's definition (`row.json`: push), checked."""
    push = row.get("push") or {}
    size, radius = push.get("size"), push.get("guard_radius_m")
    for name, value in (("size", size), ("guard_radius_m", radius)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SystemExit(
                "row %r: push.%s is %r in row.json; it must be a number (the size is set there, in "
                "executed-action units, before a run)" % (row.get("row"), name, value)
            )
    if size < 0 or radius <= 0:
        raise SystemExit("row %r: push.size must be >= 0 and push.guard_radius_m > 0" % row.get("row"))
    if list(row.get("oracle") or []) != [KIND]:
        raise SystemExit("row %r: row.json must ask for the oracle [%r]" % (row.get("row"), KIND))
    return {"size": float(size), "guard_radius_m": float(radius)}


def shift(t: Any, size: float, guard_radius_m: float) -> np.ndarray:
    """The horizontal shift of every action of the chunk, [2], executed-action units: `size` along t, falling
    linearly to zero inside the guard radius."""
    t = np.asarray(t, np.float64)[:2]
    distance = float(np.linalg.norm(t))
    if distance == 0.0:
        return np.zeros(2)
    return float(size) * min(1.0, distance / float(guard_radius_m)) * t / distance


def velocity(shift_xy: Any, output_scale_xy: Any, action_dim: int) -> np.ndarray:
    """v_o for one query, float32 [action_dim]: the velocity that, held over all Euler steps, shifts the two
    horizontal dimensions of every executed action by `shift_xy`."""
    v_o = np.zeros(int(action_dim), np.float32)
    v_o[:2] = -np.asarray(shift_xy, np.float64) / np.asarray(output_scale_xy, np.float64)
    return v_o
