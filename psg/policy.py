"""PSG on an openpi π0.5 policy: one call returns one chunk of actions, as openpi's own `Policy.infer` does.

    policy = PsgPolicy(policy_config.create_trained_policy(config, checkpoint), config)   # a = 5, b = 1
    chunk = policy.infer(request)["actions"]

`request` is what the openpi policy itself takes; for π0.5-LIBERO the two camera images, the 8-number state and
the instruction as "prompt". The policy's own input and output transforms are applied around PSG's sampler
(`sampler.py`), and the name is taken from the prompt by `name_rule.py`. The row `pi05_psg` of the evaluation
runs through this class.

It reads four private attributes of openpi (the policy's `_model`, `_input_transform` and `_output_transform`, and
the tokenizer's `_tokenizer`), at the commit the `openpi` environment pins (15a9616).
"""

from __future__ import annotations

import types
from typing import Any, Dict, Optional

import numpy as np

from psg import name_rule
from psg import sampler


class PsgPolicy:
    """An openpi π0.5 policy whose chunks are sampled with PSG."""

    def __init__(self, policy: Any, config: Any, *, a: float = sampler.A, b: float = sampler.B) -> None:
        """policy: from openpi's `create_trained_policy`; config: the openpi training config it was made from."""
        from openpi.models import tokenizer
        from openpi.shared import nnx_utils

        if not getattr(config.model, "pi05", False):
            raise ValueError("PSG is defined on π0.5; the config %r is not π0.5" % config.name)
        self.policy = policy
        self.a, self.b = float(a), float(b)
        a, b = self.a, self.b  # Python numbers: one compiled program per pair of weights
        self._sample = nnx_utils.module_jit(
            types.MethodType(
                lambda model, observation, noise, name: sampler.sample_actions(model, observation, noise, name, a=a, b=b),
                policy._model,
            )
        )
        self._tokenizer = tokenizer.PaligemmaTokenizer(config.model.max_token_len)
        self.horizon = int(config.model.action_horizon)
        self.action_dim = int(config.model.action_dim)
        self.names: Dict[str, Dict[str, Any]] = {}

    def name_of(self, instruction: str) -> Dict[str, Any]:
        """The name of one instruction: its phrases, the prompt tokens that spell them, and the mask of those tokens."""
        if instruction not in self.names:
            pieces_model = self._tokenizer._tokenizer
            ids, real = self._tokenizer.tokenize(instruction)
            active = [int(token) for token, is_real in zip(ids, real) if is_real]
            pieces = [
                "" if token == pieces_model.bos_id() else pieces_model.id_to_piece(token).replace("▁", " ")
                for token in active
            ]
            phrases = name_rule.phrases(instruction)
            indices = name_rule.token_indices(pieces, phrases)
            mask = np.zeros(len(ids), bool)
            mask[np.flatnonzero(real)[indices]] = True
            self.names[instruction] = {
                "phrases": phrases,
                "token_indices": indices,
                "token_pieces": [pieces[i] for i in indices],
                "prompt_tokens": np.asarray(ids),
                "mask": mask,
            }
        return self.names[instruction]

    def infer(self, request: Dict[str, Any], noise: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """One chunk under PSG.

        noise: float32 [horizon, action_dim], the sample at time 1; drawn from a fresh generator if not given.
        Returns "actions" [horizon, 7] after the policy's output transform, and "velocities": v, v_g and v_d at
        every Euler step, each [Euler steps, horizon, 7], as the model's velocity.
        """
        import jax
        import jax.numpy as jnp
        from openpi.models import model as _model

        name = self.name_of(request["prompt"])
        inputs = self.policy._input_transform(dict(request))
        if not np.array_equal(np.asarray(inputs["tokenized_prompt"]), name["prompt_tokens"]):
            raise RuntimeError("the policy tokenised the instruction differently from the name rule's tokeniser")
        if noise is None:
            noise = np.random.default_rng().standard_normal((self.horizon, self.action_dim)).astype(np.float32)
        batch = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        actions, kept = self._sample(
            _model.Observation.from_dict(batch), jnp.asarray(noise)[np.newaxis], jnp.asarray(name["mask"])[np.newaxis]
        )
        outputs = self.policy._output_transform({"state": np.asarray(inputs["state"]), "actions": np.asarray(actions[0])})
        kept = np.asarray(kept)[:, 0]  # [Euler steps, 3, horizon, 7]
        return {
            "actions": np.asarray(outputs["actions"]),
            "velocities": {key: kept[:, index] for index, key in enumerate(sampler.RECORDED)},
        }
