"""π0.5 + PSG: the released π0.5-LIBERO checkpoint, sampled with Pathway-Separated Guidance.

Everything but the sampler is the π0.5 row: the same checkpoint, the same input, the same noise, the same five
executed actions. The sampling is `psg/policy.py` (`PsgPolicy`), the class the README's example uses: PSG's
sampler (`psg/sampler.py`) with the name from `psg/name_rule.py`. PSG is on at every Euler step of every query,
for the whole episode.

The rows `pi05_psg_a5_b0` and `pi05_psg_a0_b1` are this adapter with other weights than the method's, stated
in their `row.json` (`"weights": {"a": ..., "b": ...}`): the single-term arms of the grounding figure, no rows
of the paper's tables. The row `pi05_psg` states none and runs the method's a = 5, b = 1. a and b are Python
numbers in both cases, so at the method's weights the compiled program is the method's.

This module's top level needs NumPy and the standard library only; the adapter is built in the `openpi`
environment.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from evaluation import interface
from evaluation.rows import _openpi
from psg import name_rule

#: The method name in the config record of a row that states its own weights.
WEIGHTS_METHOD = "psg-weights"


def weights(row: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """(a, b) from the row's definition, checked; None for a row that states no weights."""
    if "weights" not in row:
        return None
    given = row.get("weights") or {}
    for name in ("a", "b"):
        value = given.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SystemExit("row %r: weights.%s is %r in row.json; it must be a number" % (row.get("row"), name, value))
    return float(given["a"]), float(given["b"])


class Pi05Psg(_openpi.OpenpiRow):
    noise_seed_controls = (
        _openpi.OpenpiRow.noise_seed_controls + " The three velocity evaluations of PSG start from the same sample."
    )

    def __init__(self, row: Dict[str, Any]) -> None:
        self.weights = weights(row)  # a row.json with a malformed weight stops here, before the model loads
        super().__init__(row)
        from psg import sampler
        from psg.policy import PsgPolicy

        if not getattr(self.train_config.model, "pi05", False):
            raise SystemExit("PSG is defined on π0.5; row.json names %r" % self.train_config.name)
        a, b = self.weights or (sampler.A, sampler.B)
        self.psg = PsgPolicy(self.policy, self.train_config, a=a, b=b)

    def name_of(self, instruction: str) -> Dict[str, Any]:
        """The name of one instruction: its phrases, and the prompt tokens that spell them."""
        return self.psg.name_of(instruction)

    def prepare(self, instructions: List[str]) -> None:
        for instruction in instructions:
            self.name_of(instruction)

    def act(self, observation: interface.Observation) -> interface.Actions:
        noise = _openpi.query_noise(self.episode, observation.query, self.horizon)
        out = self.psg.infer(_openpi.request(observation), noise=noise)
        return interface.Actions(out["actions"], _openpi.EXECUTE, out["velocities"])

    # ------------------------------------------------------------------------------------------- the report

    def inference(self) -> Dict[str, Any]:
        from psg import sampler

        settings = super().inference()
        settings["sampler"] = {"num_steps": sampler.EULER_STEPS}
        settings["passes_per_query"] = (
            "2 encodings of the images and the instruction; %d action-expert evaluations at each Euler step"
            % len(sampler.RECORDED)
        )
        return settings

    def method(self) -> Dict[str, Any]:
        import os

        from psg import sampler

        method = {
            "name": "psg",
            "code": {"address": "this repository, psg", "revision": os.environ.get("PSG_REVISION", "unknown")},
            "settings": {
                "equation": "v~ = v + a * v_g - b * v_d",
                "a": sampler.A,
                "b": sampler.B,
                "v_g": "v - v[main-camera image tokens may not read the name tokens]",
                "v_d": "v - v[action tokens may not read the instruction tokens]",
                "applies_to": "action dimensions 0-%d (motion); dimension %d (gripper) keeps v"
                % (sampler.MOTION - 1, sampler.MOTION),
                "when": "every Euler step of every query, for the whole episode",
                "main_camera": sampler.MAIN_CAMERA,
                "name_rule": {"revision": name_rule.REVISION, "name_is": "every noun phrase of the instruction"},
                "name": {
                    instruction: {key: name[key] for key in ("phrases", "token_indices", "token_pieces")}
                    for instruction, name in sorted(self.psg.names.items())
                },
                "recorded_per_query": "v, v_g, v_d at every Euler step, each [Euler steps, horizon, 7], as the "
                "model's velocity (the sample moves by -v / Euler steps per step)",
            },
        }
        if self.weights is not None:
            method["name"] = WEIGHTS_METHOD
            settings = method["settings"]
            settings["a"], settings["b"] = self.weights
            settings["method_weights"] = {"a": sampler.A, "b": sampler.B}
            settings["status"] = (
                "PSG's sampler with other weights than the method's, an arm of the paper's Fig. 5 (b); no row of "
                "the paper's tables"
            )
        return method


def load(row: dict) -> Pi05Psg:
    return Pi05Psg(row)
