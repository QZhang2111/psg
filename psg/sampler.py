"""Pathway-Separated Guidance on π0.5: the sampler.

The released policy integrates a velocity v from noise to a chunk of actions (Eq. 1 of the paper). The
instruction enters v along two pathways:

    grounding   A -> I -> N    the action tokens read the image tokens, which have read the name
    direct      A -> l         the action tokens read the instruction tokens themselves

A cut blocks one read in every layer and head and gives a new velocity at the same sample:

    v[I_m -/-> N]   the main-camera image tokens may not read the name tokens   (a second encoding of the prefix)
    v[A -/-> l]     the action tokens may not read the instruction tokens       (the same encoding)

and PSG replaces v, at every Euler step, with

    v_g = v - v[I_m -/-> N]            what grounding contributes
    v_d = v - v[A -/-> l]              what the direct pathway contributes
    v~  = v + a * v_g - b * v_d        a = 5, b = 1

on the six motion dimensions of each action; the gripper dimension keeps v.

"The instruction tokens" l of the second cut are the whole valid prompt segment: every prompt token from the
opening <bos> to the closing newline, both included, and nothing else. The released π0.5-LIBERO prompt holds no
state tokens, and padding is never readable.

Nothing here changes the model: both cuts are attention masks passed to openpi's own transformer, on the
released checkpoint. With a = b = 0 the velocity field that is integrated is openpi's own, v. As a program in
finite precision it gives openpi's `sample_actions` where the model computes in float32 (the test on a small
model, within 1e-4). On the released weights the action expert's velocity is bfloat16: this sampler takes it
to float32 before the Euler update, openpi multiplies the bfloat16 velocity by the step first. Measured on
eight recorded observations with the released weights, the two chunks differ by up to 0.0031 at a = b = 0,
where the guidance at a = 5, b = 1 moves the chunk by up to 0.50.
"""

from __future__ import annotations

import einops
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask

A = 5.0  # weight of the grounding contribution
B = 1.0  # weight of the direct contribution; 1 removes it
MOTION = 6  # action dimensions 0-5: translation and rotation. Dimension 6 is the gripper.
MAIN_CAMERA = "base_0_rgb"  # openpi's name for the main camera's image
EULER_STEPS = 10  # as the released policy samples
RECORDED = ("v", "v_g", "v_d")  # what the sampler leaves for the record at every Euler step, in this order


def prefix_parts(observation: _model.Observation, prefix_length: int) -> tuple[jax.Array, jax.Array]:
    """Which prefix positions are the main camera's image tokens, and which are the instruction's: bool [p] each.

    openpi lays the prefix out as the image tokens of each camera in turn, then the instruction tokens.
    """
    cameras = list(observation.images)
    per_camera, remainder = divmod(prefix_length - observation.tokenized_prompt.shape[1], len(cameras))
    if remainder:
        raise ValueError("the prefix is not <cameras> x <image tokens> + <instruction tokens>")
    position = jnp.arange(prefix_length)
    first = cameras.index(MAIN_CAMERA) * per_camera
    main_image = (position >= first) & (position < first + per_camera)
    instruction = position >= len(cameras) * per_camera
    return main_image, instruction


def sample_actions(
    model,
    observation: _model.Observation,
    noise: jax.Array,
    name: jax.Array,
    *,
    a: float | jax.Array = A,
    b: float | jax.Array = B,
) -> tuple[jax.Array, jax.Array]:
    """One chunk of actions under PSG.

    model: openpi's `Pi0` with the released π0.5-LIBERO parameters.
    noise: [batch, horizon, action_dim], the sample at time 1.
    name:  bool [batch, instruction tokens], true at the instruction tokens that spell the name (`name_rule`).

    Returns the actions [batch, horizon, action_dim] and, for the record, v, v_g and v_d at every Euler step:
    float32 [Euler steps, batch, 3, horizon, 7] (`RECORDED`).
    """
    observation = _model.preprocess_observation(None, observation, train=False)
    tokens, real, segments = model.embed_prefix(observation)
    may_read = make_attn_mask(real, segments)  # [batch, p, p]: may the token of a row read the token of a column
    positions = jnp.cumsum(real, axis=1) - 1
    main_image, instruction = prefix_parts(observation, tokens.shape[1])

    def encode(may_read):
        """The backbone's pass over images and instruction; what it leaves for the action expert to read."""
        _, encoding = model.PaliGemma.llm([tokens, None], mask=may_read, positions=positions)
        return encoding

    encoding = encode(may_read)
    name = jnp.pad(name, ((0, 0), (tokens.shape[1] - name.shape[1], 0)))  # the instruction ends the prefix
    grounding_cut = main_image[None, :, None] & name[:, None, :]  # I_m -/-> N
    encoding_without_grounding = encode(may_read & ~grounding_cut)
    without_instruction = real & ~instruction[None, :]  # A -/-> l

    batch, horizon = noise.shape[0], model.action_horizon
    dt = -1.0 / EULER_STEPS  # openpi's convention: time 1 is noise, time 0 the actions

    def velocity(x, time, encoding, readable):
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
        velocities = (
            velocity(x, time, encoding, real),
            velocity(x, time, encoding_without_grounding, real),
            velocity(x, time, encoding, without_instruction),
        )
        velocities = jax.lax.optimization_barrier(velocities)
        v, v_without_grounding, v_without_direct = velocities
        v_g = v - v_without_grounding
        v_d = v - v_without_direct
        v_tilde = v + a * v_g - b * v_d
        recorded = [v, v_g, v_d]
        v_tilde = jnp.concatenate([v_tilde[..., :MOTION], v[..., MOTION:]], axis=-1)  # the gripper keeps v
        kept = kept.at[step].set(jnp.stack(recorded, axis=1)[..., : MOTION + 1])
        return x + dt * v_tilde.astype(x.dtype), time + dt, step + 1, kept

    kept = jnp.zeros((EULER_STEPS, batch, len(RECORDED), horizon, MOTION + 1), jnp.float32)
    actions, _, steps, kept = jax.lax.while_loop(lambda carry: carry[1] >= -dt / 2, euler_step, (noise, 1.0, 0, kept))
    return actions, kept
