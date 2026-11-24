"""π0.5 + RSS: the authors' released checkpoint and their guided π0.5 (`Pi0_CFG`) behind the interface.

RSS is its own fine-tune of `pi05_base` (with captions in training), run with guidance on the velocity at every
Euler step. Everything below is read from the release at f51a0a3f:

* the server builds the policy with `policy_config.create_trained_policy_cfg(get_config(
  "pi05_cfg_libero_caption_infer"), <checkpoint>)` (`scripts/serve_policy.py` L93-95; the command in
  `run_server_eval.sh` L8-10);
* the config: `Pi0CFGConfig(pi05=True, action_horizon=10, max_token_len=350, discrete_state_input=True)`,
  `CFGLeRobotLiberoDataConfig(extra_caption=False, use_cfg=True)`, norm statistics `physical-intelligence/libero`
  (`training/misc/libero_cfg_config.py` L219-229); `num_steps` 10 and `guidance_scale` 1.5 are that data
  config's defaults (L39-40), handed to the sampler as `sample_kwargs` (`policies/policy_config.py` L123-126);
* the input: openpi's LIBERO inputs, then `TokenizeCFGPrompt`: the prompt **and the robot state** as text,
  "Task: <prompt>, State: <256-bin digits>;\\nAction: ", and a null prompt "" with the same state
  (`training/config.py` L160-171, `transforms.py` L310-336, `models/tokenizer.py` L22-33);
* the sampler: two prefixes (conditioned, null), and at each of the 10 Euler steps
  v = v_u + γ (v_c − v_u) on the velocity, all dimensions (`models/pi0_cfg.py` L289-376);
* the client sends the request of openpi's LIBERO client, with the state, and executes 5 of the 10 actions
  per query (`examples/libero_shortcut/main_origin.py` L32-33, L119-152).

The one thing restated here is `CFGPolicy.infer` (`policies/policy.py` L132-150), with the query's noise added
to the sampler's arguments; the release's `infer` takes no noise and splits its key once per query in server
order.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from evaluation.interface import Actions, Adapter, Episode, Observation
from evaluation.rows.pi05_cag import openpi_libero as L

EXECUTE = 5  # main_origin.py L33
CONFIG = "pi05_cfg_libero_caption_infer"
SUBFOLDER = "pi05_cfg_libero_caption_exp1/29999"
LOADED = ["%s/params/*" % SUBFOLDER, "%s/assets/*" % SUBFOLDER]  # train_state/ (32 GB) is not needed for inference


class Rss(Adapter):
    def __init__(self, row: Dict[str, Any], policy=None, observation_cls=None):
        self.row = row
        self._episode = None
        if policy is None:
            from openpi.models import model as _model
            from openpi.policies import policy_config
            from openpi.training import config

            ck = row["checkpoint"]
            snapshot = L.hub_snapshot(ck["address"].split("huggingface.co/")[1], ck["revision"], allow=LOADED)
            self._checkpoint = snapshot / SUBFOLDER
            self._train_config = config.get_config(CONFIG)
            policy = policy_config.create_trained_policy_cfg(self._train_config, self._checkpoint)
            observation_cls = _model.CFGObservation
        self.policy = policy
        self._observation_cls = observation_cls

    def begin_episode(self, episode: Episode) -> None:
        self._episode = episode

    def infer(self, request: Dict[str, Any], noise: np.ndarray) -> Dict[str, Any]:
        """`CFGPolicy.infer` of the release (policies/policy.py L132-150) with `noise` passed to the sampler."""
        import jax
        import jax.numpy as jnp

        p = self.policy
        inputs = p._input_transform(jax.tree.map(lambda x: x, request))
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        p._rng, sample_rng = jax.random.split(p._rng)
        outputs = {
            "state": inputs["state"],
            "actions": p._sample_actions(sample_rng, self._observation_cls.from_dict(inputs),
                                         noise=jnp.asarray(noise)[np.newaxis, ...], **p._sample_kwargs),
        }
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        return p._output_transform(outputs)

    def act(self, observation: Observation) -> Actions:
        out = self.infer(L.libero_request(observation), L.query_noise(self._episode, observation.query))
        return Actions(actions=np.asarray(out["actions"], dtype=np.float64), execute=EXECUTE)

    def report(self) -> Dict[str, Any]:
        p = self.policy
        cfg = p._sample_kwargs
        return {
            "model": {
                "name": "π0.5 + RSS (RAS and MCSI), the authors' fine-tune",
                "checkpoint": dict(self.row["checkpoint"], subfolder=SUBFOLDER, loaded=LOADED,
                                   sha256=L.tree_sha256(self._checkpoint)),
                "code": dict(self.row["code"]),
            },
            "inference": {
                "openpi_config": CONFIG,
                "sampler": "Euler (Pi0_CFG.sample_actions), guided at every step",
                "sampling_steps": int(cfg.get("num_steps")),
                "action_horizon": int(self._train_config.model.action_horizon),
                "max_token_len": int(self._train_config.model.max_token_len),
                "discrete_state_input": bool(self._train_config.model.discrete_state_input),
                "actions_executed_per_query": EXECUTE,
                "images": "main and wrist, each rotated by 180 degrees, resized with padding to 224 x 224 (openpi-client), uint8",
                "state": "read: eef_pos (3), eef_quat as axis-angle (3), gripper_qpos (2), normalised by the checkpoint's statistics, discretised into 256 bins and written into the prompt (discrete_state_input = true)",
                "prompt": "'Task: <instruction>, State: <bins>;\\nAction: '; the null prompt is '' with the same state",
                "normalisation": "the checkpoint's own statistics (assets/physical-intelligence/libero)",
                "precision": "bfloat16 parameters (restore_params)",
                "ensembling": "none",
            },
            "method": {
                "name": "rss-guidance",
                "code": dict(self.row["code"], file="src/openpi/models/pi0_cfg.py"),
                "settings": {
                    "guidance_scale": float(cfg.get("guidance_scale")),
                    "where": "on the velocity, at every Euler step, all action dimensions: v = v_u + guidance_scale * (v_c - v_u)",
                    "null_condition": "the empty prompt with the robot state kept",
                    "passes_per_step": 2,
                },
            },
            "noise_seed_controls": "The initial noise of every query: interface.noise_generator(episode, query).standard_normal((10, 32)), float32 (the shared rule: noise seed, task index, start state, query index); both guided branches start from it.",
            "environment": {"name": self.row["environment"], "python": L.python_version(),
                            "packages": L.versions("openpi", "openpi-client", "jax", "flax", "numpy", "huggingface-hub")},
            "requires": {"camera_height": 256, "camera_width": 256},
        }


def load(row: Dict[str, Any]) -> Adapter:
    return Rss(row)
