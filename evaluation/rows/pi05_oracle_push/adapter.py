"""π0.5 with an oracle push: an instrument for the paper's Fig. 7 (b), no row of the paper's tables.

Everything but the sampler is the π0.5 row: the same checkpoint, the same input, the same noise, the same five
executed actions. The sampler is `instruments/oracle_push_sampler.py`; what it does and why the sign is
what it is stands in `instruments/oracle_push.py`. The push is on at every Euler step of every query,
for the whole episode.

The row asks the evaluation for an oracle (`row.json`: oracle; `evaluation/oracle.py`): the position of
the task's Reach target at every query. The size of the push is `row.json`: push.size, in executed-action
units; the unit's config record states it, with the oracle, the guard and the policy's output scale.
"""

from __future__ import annotations

import os
import types
from typing import Any, Dict

import numpy as np

from evaluation import interface
from instruments import oracle_push
from evaluation.rows import _openpi


class Pi05OraclePush(_openpi.OpenpiRow):
    noise_seed_controls = (
        _openpi.OpenpiRow.noise_seed_controls + " The two velocity evaluations of a step start from the same sample."
    )

    def __init__(self, row: Dict[str, Any]) -> None:
        self.push = oracle_push.settings(row)  # a row.json without a size stops here, before any weights load
        super().__init__(row)
        from openpi.shared import nnx_utils
        from instruments import oracle_push_sampler

        if not getattr(self.train_config.model, "pi05", False):
            raise SystemExit("the oracle push is defined on π0.5; row.json names %r" % self.train_config.name)
        self.action_dim = int(self.train_config.model.action_dim)
        self._sample = nnx_utils.module_jit(
            types.MethodType(
                lambda model, observation, noise, push: oracle_push_sampler.sample_actions(
                    model, observation, noise, push
                ),
                self.policy._model,
            )
        )
        self.output_scale = self._output_scale()

    def _output_scale(self) -> np.ndarray:
        """How far one unit of the sample moves each executed action, [7], read from the policy's own output
        transform (un-normalisation with the checkpoint's statistics, then the release's output mapping) by
        probing it. The push rests on that map being the same for every state and every action of a chunk and
        on a dimension of the sample moving its own dimension of the action only; each is checked here."""

        def returned(sample: np.ndarray, state: np.ndarray) -> np.ndarray:
            outputs = self.policy._output_transform({"state": state.copy(), "actions": sample.copy()})
            return np.asarray(outputs["actions"], np.float64)

        zeros = np.zeros((self.horizon, self.action_dim), np.float32)
        state = np.zeros(self.action_dim, np.float32)
        base = returned(zeros, state)
        if not np.array_equal(base, returned(zeros, np.ones(self.action_dim, np.float32))):
            raise SystemExit("the policy's output depends on the state: the oracle push assumes it does not")
        scale = np.zeros(base.shape[1])
        for dimension in range(base.shape[1]):
            probe = zeros.copy()
            probe[:, dimension] = 1.0
            moved = returned(probe, state) - base  # [horizon, 7]
            others = np.delete(moved, dimension, axis=1)
            if np.abs(others).max() > 1e-9 or np.ptp(moved[:, dimension]) > 1e-9:
                raise SystemExit("the policy's output is not one scale per action dimension (dimension %d)" % dimension)
            scale[dimension] = moved[0, dimension]
        if not (scale[:2] > 0).all():
            raise SystemExit("the policy's output scale of a horizontal dimension is not positive: %s" % scale[:2])
        return scale

    # ------------------------------------------------------------------------------------------- one query

    def act(self, observation: interface.Observation) -> interface.Actions:
        import jax
        import jax.numpy as jnp
        from openpi.models import model as _model

        told = (observation.oracle or {}).get("reach_target")
        if not told:
            raise RuntimeError("this row needs the oracle %r in every observation; none came" % oracle_push.KIND)
        target = np.asarray(told["position"], np.float64)
        t = target[:2] - np.asarray(observation.state["eef_pos"], np.float64)[:2]
        shift = oracle_push.shift(t, self.push["size"], self.push["guard_radius_m"])
        v_o = oracle_push.velocity(shift, self.output_scale[:2], self.action_dim)

        noise = _openpi.query_noise(self.episode, observation.query, self.horizon)
        inputs = self.policy._input_transform(dict(_openpi.request(observation)))
        batch = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        actions, kept = self._sample(
            _model.Observation.from_dict(batch), jnp.asarray(noise)[np.newaxis], jnp.asarray(v_o)[np.newaxis]
        )
        outputs = self.policy._output_transform(
            {"state": np.asarray(inputs["state"]), "actions": np.asarray(actions[0])}
        )
        kept = np.asarray(kept)[:, 0]  # [Euler steps, 3, horizon, 7]
        recorded = {key: kept[:, index] for index, key in enumerate(oracle_push.RECORDED)}
        recorded["shift"] = np.asarray(shift, np.float32)  # [2], executed-action units, toward the object
        recorded["target_position"] = target.astype(np.float32)  # [3], what the oracle told at this query
        return interface.Actions(np.asarray(outputs["actions"]), _openpi.EXECUTE, recorded)

    # ------------------------------------------------------------------------------------------- the report

    def inference(self) -> Dict[str, Any]:
        from psg import sampler

        settings = super().inference()
        settings["sampler"] = {"num_steps": sampler.EULER_STEPS}
        settings["passes_per_query"] = (
            "1 encoding of the images and the instruction; 2 action-expert evaluations at each Euler step"
        )
        return settings

    def method(self) -> Dict[str, Any]:
        from psg import sampler

        return {
            "name": oracle_push.METHOD,
            "code": {
                "address": "this repository, instruments (oracle_push.py, oracle_push_sampler.py)",
                "revision": os.environ.get("PSG_REVISION", "unknown"),
            },
            "settings": {
                "status": "an instrument for the paper's Fig. 7 (b): the row is told where the requested object "
                "is; no policy and no row of the paper's tables",
                "equation": "v~ = v + v_o - b * v_d",
                "b": sampler.B,
                "v_d": "v - v[action tokens may not read the instruction tokens], as PSG",
                "v_o": "-shift / output_scale on action dimensions 0 and 1, zero elsewhere; the same at every "
                "Euler step and for every action of the chunk",
                "shift": "size * min(1, |t| / guard_radius_m) * t / |t|: what the push adds to every action of the "
                "chunk, executed-action units, toward the object",
                "t": "the oracle's position of the Reach target's body minus the gripper site's, x and y, world, m, "
                "at the query",
                "size": self.push["size"],
                "size_unit": "executed-action units (one unit of a translation dimension moves the controller's "
                "target by its action scale per step)",
                "guard_radius_m": self.push["guard_radius_m"],
                "output_scale": [float(value) for value in self.output_scale],
                "output_scale_is": "executed-action units per unit of the sample, per action dimension, probed "
                "from the policy's output transform when the row loaded",
                "oracle": list(self.row["oracle"]),
                "grounding_term": "not evaluated: PSG's a * v_g is replaced by v_o",
                "applies_to": "action dimensions 0-%d (motion); dimension %d (gripper) keeps v"
                % (sampler.MOTION - 1, sampler.MOTION),
                "when": "every Euler step of every query, for the whole episode",
                "recorded_per_query": "v, v_o, v_d at every Euler step, each [Euler steps, horizon, 7], as the "
                "model's velocity (the sample moves by -v / Euler steps per step); shift [2]; target_position [3]",
            },
        }

    def report(self) -> Dict[str, Any]:
        report = super().report()
        report["requires"]["oracle"] = list(self.row["oracle"])  # the evaluation passes it and records that it did
        return report


def load(row: dict) -> Pi05OraclePush:
    return Pi05OraclePush(row)
