"""The sampler of the oracle push (`oracle_push.py` says what it is): two velocities per Euler step and a push.

    v~ = v + v_o - b * v_d          v_d = v - v[A -/-> l], as PSG; v_o is given, the same at every step

It reuses the method's layout of the prefix (`psg/sampler.py::prefix_parts`) and its constants; the
method imports nothing from here. The velocity function below is the method's, which is local to the method's
own function and so cannot be called from outside it: the same lines, without the grounding cut's encoding.

Runs in the `openpi` environment.
"""

from __future__ import annotations

import einops
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask
from instruments import oracle_push
from psg import sampler


def sample_actions(
    model,
    observation: _model.Observation,
    noise: jax.Array,
    push: jax.Array,
    *,
    b: float | jax.Array = sampler.B,
) -> tuple[jax.Array, jax.Array]:
    """One chunk of actions under the oracle push.

    model: openpi's `Pi0` with the released π0.5-LIBERO parameters.
    noise: [batch, horizon, action_dim], the sample at time 1.
    push:  float32 [batch, action_dim], v_o (`oracle_push.velocity`): added at every Euler step to every action
           of the chunk.

    Returns the actions [batch, horizon, action_dim] and, for the record, v, v_o and v_d at every Euler step
    (`oracle_push.RECORDED`): float32 [Euler steps, batch, 3, horizon, 7].
    """
    observation = _model.preprocess_observation(None, observation, train=False)
    tokens, real, segments = model.embed_prefix(observation)
    may_read = make_attn_mask(real, segments)  # [batch, p, p]: may the token of a row read the token of a column
    positions = jnp.cumsum(real, axis=1) - 1
    _, instruction = sampler.prefix_parts(observation, tokens.shape[1])
    _, encoding = model.PaliGemma.llm([tokens, None], mask=may_read, positions=positions)
    without_instruction = real & ~instruction[None, :]  # A -/-> l

    batch, horizon = noise.shape[0], model.action_horizon
    dt = -1.0 / sampler.EULER_STEPS  # openpi's convention: time 1 is noise, time 0 the actions
    v_o = jnp.broadcast_to(push[:, None, :], noise.shape).astype(jnp.float32)

    def velocity(x, time, readable):
        """The action expert's velocity at (x, time), its tokens reading the prefix tokens in `readable`."""
        action_tokens, action_real, action_segments, time_condition = model.embed_suffix(
            observation, x, jnp.broadcast_to(time, batch)
        )
        may_read = jnp.concatenate(
            [
                einops.repeat(readable, "b p -> b s p", s=action_tokens.shape[1]),
                make_attn_mask(action_real, action_segments),
            ],
            axis=-1,
        )
        positions = jnp.sum(real, axis=-1)[:, None] + jnp.cumsum(action_real, axis=-1) - 1
        (_, out), _ = model.PaliGemma.llm(
            [None, action_tokens],
            mask=may_read,
            positions=positions,
            kv_cache=encoding,
            adarms_cond=[None, time_condition],
        )
        return model.action_out_proj(out[:, -horizon:]).astype(jnp.float32)

    def euler_step(carry):
        x, time, step, kept = carry
        v, v_without_direct = jax.lax.optimization_barrier(
            (velocity(x, time, real), velocity(x, time, without_instruction))
        )
        v_d = v - v_without_direct
        v_tilde = v + v_o - b * v_d
        v_tilde = jnp.concatenate([v_tilde[..., : sampler.MOTION], v[..., sampler.MOTION :]], axis=-1)  # the gripper keeps v
        kept = kept.at[step].set(jnp.stack([v, v_o, v_d], axis=1)[..., : sampler.MOTION + 1])
        return x + dt * v_tilde.astype(x.dtype), time + dt, step + 1, kept

    kept = jnp.zeros((sampler.EULER_STEPS, batch, len(oracle_push.RECORDED), horizon, sampler.MOTION + 1), jnp.float32)
    actions, _, _, kept = jax.lax.while_loop(lambda carry: carry[1] >= -dt / 2, euler_step, (noise, 1.0, 0, kept))
    return actions, kept
