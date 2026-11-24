"""π0.5 + CAG with the vision-action model: the authors' `CounterfactualActionGuidedPolicy` of
`openpi-cf/scripts/serve_policy_VA.py` (CAG's fork, 8280a500) behind the interface.

What the release does at one query (`RELEASE.md` in `pi05_cag/` has the full table):

* the conditioned policy is the released π0.5-LIBERO (config `pi05_libero`); the unconditioned policy is the
  authors' vision-action model (config `pi05_libero_unconditioned`: `mask_language_tokens=True`, a field of the
  fork, `openpi-cf/src/openpi/training/config.py` L787-812) loaded from its own checkpoint (L228-238);
* each policy transforms the request with its own input transform (L175-176); the unconditioned input then has
  `tokenized_prompt` and its mask set to zeros (L141-147, L180), and the model masks the language tokens again
  itself (`models/pi0.py` of the fork, `embed_prefix(..., mask_language_tokens=self.mask_language_tokens)`);
* both are sampled from **the same noise tensor** (L188-199); the two final chunks are mixed once,
  a = a_u + ω (a_c − a_u), in normalised space (L203), and un-normalised by the conditioned policy's output
  transform (L205-209).

ω = 1.5, as the README's command for this variant (L308) and the paper. Five actions executed per query.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from evaluation.interface import Actions, Adapter, Episode, Observation
from evaluation.rows.pi05_cag import openpi_libero as L
from evaluation.rows.pi05_cag.adapter import EXECUTE, OMEGA, inference_of, release_module

SERVER_FILE = "serve_policy_VA.py"
SERVER_BLOB = "7f25d6f630cecdfbc7870c64285fa58ca0d27212"  # git blob of openpi-cf/scripts/serve_policy_VA.py at 8280a500


class CagVisionAction(Adapter):
    def __init__(self, row: Dict[str, Any], policy=None):
        self.row = row
        self._episode = None
        if policy is None:
            server, self._server_path = release_module(SERVER_FILE, SERVER_BLOB)
            from openpi.policies import policy_config
            from openpi.training import config

            cond, uncond = row["checkpoint"], row["unconditioned_checkpoint"]
            self._checkpoint = L.local_checkpoint(cond["address"])
            self._uncond_checkpoint = L.hub_snapshot(uncond["address"].split("huggingface.co/")[1], uncond["revision"])
            conditional = policy_config.create_trained_policy(config.get_config(cond["openpi_config"]), self._checkpoint)
            unconditional = policy_config.create_trained_policy(config.get_config(uncond["openpi_config"]), self._uncond_checkpoint)
            # The release's create_policy passes guidance_scale only; uncond_empty_prompt keeps its default, False (L236-238).
            policy = server.CounterfactualActionGuidedPolicy(conditional, unconditional, guidance_scale=OMEGA)
        self.policy = policy

    def begin_episode(self, episode: Episode) -> None:
        self._episode = episode

    def act(self, observation: Observation) -> Actions:
        out = self.policy.infer(L.libero_request(observation), noise=L.query_noise(self._episode, observation.query))
        return Actions(actions=np.asarray(out["actions"], dtype=np.float64), execute=EXECUTE)

    def report(self) -> Dict[str, Any]:
        p = self.policy
        cond, uncond = p._conditional_policy, p._unconditional_policy
        u = dict(self.row["unconditioned_checkpoint"], sha256=L.tree_sha256(self._uncond_checkpoint))
        return {
            "model": {
                "name": "π0.5-LIBERO with CAG's vision-action model",
                "checkpoint": dict(self.row["checkpoint"], sha256=L.tree_sha256(self._checkpoint), unconditioned=u),
                "code": dict(self.row["code"]),
            },
            "inference": dict(inference_of(cond, cond._model, executed=EXECUTE),
                              unconditioned_model_masks_language=bool(getattr(uncond._model, "mask_language_tokens", False)),
                              unconditioned_sampling_steps=int(uncond._sample_kwargs.get("num_steps", 10))),
            "method": {
                "name": "cag-vision-action",
                "code": dict(self.row["code"], file="openpi-cf/scripts/" + SERVER_FILE, git_blob=L.git_blob(self._server_path)),
                "settings": {
                    "guidance_scale": float(p._guidance_scale),
                    "uncond_empty_prompt": bool(p._uncond_empty_prompt),
                    "unconditioned_input": "the unconditioned policy's own transform of the request, tokenized_prompt and its mask set to zeros",
                    "unconditioned_model": self.row["unconditioned_checkpoint"]["address"],
                    "noise": "one tensor per query, the same for both samplings",
                    "mixed": "once per query, on the final sampled chunk, all 32 action dimensions, before the conditioned policy's un-normalisation",
                    "rule": "a = a_u + guidance_scale * (a_c - a_u)",
                    "samplings_per_query": 2,
                },
            },
            "noise_seed_controls": L.NOISE_SENTENCE,
            "environment": {"name": self.row["environment"], "python": L.python_version(),
                            "packages": L.versions("openpi", "openpi-client", "jax", "flax", "numpy", "huggingface-hub")},
            "requires": {"camera_height": 256, "camera_width": 256},
        }


def load(row: Dict[str, Any]) -> Adapter:
    return CagVisionAction(row)
