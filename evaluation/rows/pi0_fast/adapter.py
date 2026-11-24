"""π0-FAST: the released π0-FAST-LIBERO checkpoint.

Two things differ from the other openpi rows:

- It draws no noise: actions are decoded token by token, greedily (temperature 0), so the noise seed controls
  nothing.
- **Normalisation is z-score.** openpi at 15a9616 would choose quantile normalisation for a FAST model; with it
  the released LIBERO checkpoint was seen not to complete the original tasks, with z-scores it did (our
  exploration; the row's first smoke repeats the comparison on one task before any table run). The record
  states the kind in use.
"""

from __future__ import annotations

import numpy as np

from evaluation import interface
from evaluation.rows import _openpi


class Pi0Fast(_openpi.OpenpiRow):
    use_quantile_norm = False
    noise_seed_controls = "Nothing: actions are decoded greedily (temperature 0); the policy is deterministic."

    def act(self, observation: interface.Observation) -> interface.Actions:
        actions = np.asarray(self.policy.infer(_openpi.request(observation))["actions"])
        return interface.Actions(actions, _openpi.EXECUTE)


def load(row: dict) -> Pi0Fast:
    return Pi0Fast(row)
