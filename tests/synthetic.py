"""A hand-made scene for the tests of the trace, the labeller and the rollout: no simulator.

Bodies: the world, one gripper finger, two foods, a cabinet with a base, a middle and a top drawer, the table.
Every body has one geom at its origin, except the cabinet's base, whose geom sits 10 cm along the body's x axis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

BODIES = [
    "world", "gripper0_finger", "alphabet_soup_1_main", "tomato_sauce_1_main", "wooden_cabinet_1_base",
    "wooden_cabinet_1_cabinet_middle", "main_table_base", "plate_10_main", "wooden_cabinet_1_cabinet_top",
]  # fmt: skip
GEOM_POS = np.zeros((len(BODIES), 3), np.float32)
GEOM_POS[4] = [0.1, 0.0, 0.0]
TABLES: Dict[str, Any] = {
    "body_name": np.asarray(BODIES),
    "body_parent": np.zeros(len(BODIES), np.int32),
    "geom_name": np.asarray([name + "_g" for name in BODIES]),
    "geom_body": np.arange(len(BODIES), dtype=np.int32),
    "geom_pos": GEOM_POS,
    "geom_size": np.full((len(BODIES), 3), 0.02, np.float32),
    "geom_type": np.zeros(len(BODIES), np.int32),
    "geom_collides": np.ones(len(BODIES), bool),
    "entity_name": np.asarray(["alphabet_soup_1", "tomato_sauce_1", "plate_1", "wooden_cabinet_1", "main_table"]),
    "entity_kind": np.asarray(["object", "object", "object", "fixture", "fixture"]),
    "gripper_geoms": np.asarray([1], np.int32),
}
SOUP, TOMATO, CABINET, DRAWER, TABLE, PLATE_10, TOP_DRAWER = 2, 3, 4, 5, 6, 7, 8

#: Where the bodies stand; a test moves the hand.
PLACES = np.asarray(
    [[0, 0, 0], [0, 0, 0.3], [0.5, 0.0, 0.0], [0.5, 0.3, 0.0], [0.0, 0.6, 0.0], [0.0, 0.6, 0.3], [0.0, 0.0, -0.5], [5, 5, 5], [0.0, 0.6, 0.6]],
    np.float32,
)  # fmt: skip
IDENTITY = np.tile(np.asarray([1.0, 0.0, 0.0, 0.0], np.float32), (len(BODIES), 1))  # (w, x, y, z)


def frame(hand: Any, contacts: Optional[List[Tuple[int, int, float]]] = None, body_quat: Any = None) -> Dict[str, Any]:
    body_pos = PLACES.copy()
    body_pos[1] = hand
    return {
        "eef_pos": np.asarray(hand, np.float32),
        "eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0], np.float32),
        "gripper_qpos": np.zeros(2, np.float32),
        "joint_pos": np.zeros(7, np.float32),
        "body_pos": body_pos,
        "body_quat": IDENTITY if body_quat is None else np.asarray(body_quat, np.float32),
        "contacts": list(contacts or []),
    }


def action(gripper: float) -> List[float]:
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(gripper)]


class Scene:
    """A scene for the rollout: the hand moves by 0.1 x the action's translation per step; the predicate holds
    from `succeeds_at` policy steps on (None: never)."""

    def __init__(self, succeeds_at: Optional[int] = None, ends_at: Optional[int] = None) -> None:
        self.succeeds_at, self.ends_at = succeeds_at, ends_at
        self.hand = np.asarray([0.0, 0.0, 0.3])
        self.actions: List[np.ndarray] = []
        self.resets: List[int] = []

    def reset(self, state: int) -> None:
        self.resets.append(state)
        self.hand = np.asarray([0.0, 0.0, 0.3])
        self.actions = []

    def tables(self) -> Dict[str, Any]:
        return TABLES

    def step(self, act: Any) -> None:
        act = np.asarray(act, np.float64)
        self.actions.append(act)
        self.hand = self.hand + 0.1 * act[:3]

    def policy_steps(self) -> int:
        return sum(1 for act in self.actions if act[6] != -1.0 or np.any(act[:6] != 0))

    def status(self) -> Tuple[bool, bool]:
        steps = self.policy_steps()
        success = self.succeeds_at is not None and steps >= self.succeeds_at
        ended = success if self.ends_at is None else steps >= self.ends_at
        return ended, success

    def observation(self) -> Dict[str, Any]:
        return {
            "images": {"main": np.zeros((256, 256, 3), np.uint8), "wrist": np.zeros((256, 256, 3), np.uint8)},
            "state": {"eef_pos": self.hand.copy(), "eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
                      "gripper_qpos": np.zeros(2), "joint_pos": np.zeros(7)},
            "cameras": {name: {"intrinsics": np.eye(3), "extrinsics": np.eye(4)} for name in ("main", "wrist")},
            "controller": {"action_scale": np.asarray([0.05] * 3 + [0.5] * 3)},
        }  # fmt: skip

    def frame(self) -> Dict[str, Any]:
        return frame(self.hand)

    def oracle(self, kinds: Any) -> Dict[str, Any]:
        """What a scene tells a row that asks for the oracle: here the soup is the target."""
        assert list(kinds) == ["reach_target_position"]
        return {"reach_target": {"target": "alphabet_soup_1", "body": BODIES[SOUP], "position": PLACES[SOUP].astype(np.float64)}}

    def main_image(self) -> np.ndarray:
        return np.zeros((256, 256, 3), np.uint8)

    def close(self) -> None:
        pass
