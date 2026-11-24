"""π0.5 + CAG, the training-free variant: the authors' `CounterfactualActionGuidedPolicy` behind the interface.

The method is the authors' code, imported from the row's environment (CAG's openpi fork at the pinned revision,
`openpi-cf/scripts/serve_policy_TF.py`); nothing of it is restated here. What the release does at one query
(file and lines at 8280a500; `RELEASE.md` beside this file has the full table):

* the conditioned input is openpi's `_input_transform` of the request (L147); the unconditioned input is the same
  with `tokenized_prompt` and `tokenized_prompt_mask` set to zeros (L113-119, L151; `uncond_empty_prompt=False`,
  the default, L57);
* both are sampled by the base policy's `_sample_actions` from **the same noise tensor** (L159-168);
* the two final chunks are mixed once, a = a_u + ω (a_c − a_u), on all 32 action dimensions in the model's
  normalised space (L172), and then un-normalised and cut to 7 by the output transform (L174-178).

ω = 1.5 (the paper, `main.tex` L609-611). The README's command for this variant passes
2.0 (README L274, L283); that is not used. Five of the ten actions are executed per query (`eval/main_cf.py` L38).

The adapter passes the query's noise through the release's own `infer(obs, noise=...)` (L146); the release's
websocket server never does and draws it from a key split once per query in server order (L157-159).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict

import numpy as np

from evaluation.interface import Actions, Adapter, Episode, Observation
from evaluation.rows.pi05_cag import openpi_libero as L

OMEGA = 1.5
EXECUTE = 5  # replan_steps of the release's client, eval/main_cf.py L38
SERVER_FILE = "serve_policy_TF.py"
SERVER_BLOB = "705c7ef5f9c6137d5b4504f4e9aed52b046ab3a6"  # git blob of openpi-cf/scripts/serve_policy_TF.py at 8280a500


def release_module(file_name: str, blob: str):
    """Import one of CAG's server scripts from the fork openpi is installed from, after checking it is the pinned file."""
    import openpi

    path = Path(openpi.__file__).resolve().parents[2] / "scripts" / file_name
    if not path.exists():
        raise RuntimeError("openpi is not CAG's fork: no %s next to it (%s)" % (file_name, path.parent))
    found = L.git_blob(path)
    if found != blob:
        raise RuntimeError("%s is git blob %s, the pinned revision has %s" % (file_name, found, blob))
    spec = importlib.util.spec_from_file_location("cag_" + file_name[:-3], str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


class CagTrainingFree(Adapter):
    def __init__(self, row: Dict[str, Any], policy=None, server=None):
        self.row = row
        self._episode = None
        if policy is None:  # the real model; tests pass a stand-in
            server, self._server_path = release_module(SERVER_FILE, SERVER_BLOB)
            from openpi.policies import policy_config
            from openpi.training import config

            self._checkpoint = L.local_checkpoint(row["checkpoint"]["address"])
            base = policy_config.create_trained_policy(config.get_config(row["checkpoint"]["openpi_config"]), self._checkpoint)
            policy = server.CounterfactualActionGuidedPolicy(base, guidance_scale=OMEGA, uncond_empty_prompt=False)
        self.policy = policy
        self.server = server

    # -- the interface ---------------------------------------------------------------------------------------
    def begin_episode(self, episode: Episode) -> None:
        self._episode = episode

    def act(self, observation: Observation) -> Actions:
        request = L.libero_request(observation)
        out = self.policy.infer(request, noise=L.query_noise(self._episode, observation.query))
        actions = np.asarray(out["actions"], dtype=np.float64)
        return Actions(actions=actions, execute=EXECUTE)

    def report(self) -> Dict[str, Any]:
        p = self.policy
        base = p._base_policy
        model = base._model
        return {
            "model": {
                "name": "π0.5-LIBERO",
                "checkpoint": dict(self.row["checkpoint"], sha256=L.tree_sha256(self._checkpoint)),
                "code": dict(self.row["code"]),
            },
            "inference": inference_of(base, model, executed=EXECUTE),
            "method": {
                "name": "cag-training-free",
                "code": dict(self.row["code"], file="openpi-cf/scripts/" + SERVER_FILE, git_blob=L.git_blob(self._server_path)),
                "settings": {
                    "guidance_scale": float(p._guidance_scale),
                    "uncond_empty_prompt": bool(p._uncond_empty_prompt),
                    "unconditioned_input": "the conditioned input with tokenized_prompt and tokenized_prompt_mask set to zeros",
                    "unconditioned_model": "the same policy",
                    "noise": "one tensor per query, the same for both samplings",
                    "mixed": "once per query, on the final sampled chunk, all 32 action dimensions, before un-normalisation",
                    "rule": "a = a_u + guidance_scale * (a_c - a_u)",
                    "samplings_per_query": 2,
                },
            },
            "noise_seed_controls": L.NOISE_SENTENCE,
            "environment": {"name": self.row["environment"], "python": L.python_version(),
                            "packages": L.versions("openpi", "openpi-client", "jax", "flax", "numpy")},
            "requires": {"camera_height": 256, "camera_width": 256},
        }


def inference_of(base, model, executed: int) -> Dict[str, Any]:
    """The base policy's inference settings, read from the loaded objects."""
    return {
        "sampler": "Euler (openpi Pi0.sample_actions)",
        "sampling_steps": int(base._sample_kwargs.get("num_steps", 10)),
        "sampling_steps_source": "policy sample_kwargs" if "num_steps" in base._sample_kwargs else "Pi0.sample_actions default",
        "action_horizon": int(model.action_horizon),
        "action_dim_padded": int(model.action_dim),
        "actions_executed_per_query": int(executed),
        "images": "main and wrist, each rotated by 180 degrees, resized with padding to 224 x 224 (openpi-client), uint8",
        "state": "eef_pos (3), eef_quat as axis-angle (3), gripper_qpos (2) sent; read by the model only if its tokens hold it (pi05_libero: discrete_state_input = false)",
        "prompt": "the instruction as given",
        "normalisation": "the checkpoint's own statistics",
        "precision": "bfloat16 parameters (openpi restore_params)",
        "ensembling": "none",
    }


def load(row: Dict[str, Any]) -> Adapter:
    return CagTrainingFree(row)
