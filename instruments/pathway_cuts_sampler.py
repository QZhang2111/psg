"""The sampler of the pathway cuts (`pathway_cuts.py` says what they are): openpi's sampler with two masks.

It follows openpi's `Pi0.sample_actions` line by line (`src/openpi/models/pi0.py`: the prefix pass, then the
Euler steps of the action expert) and differs from it in two places only, both attention masks passed to
openpi's own transformer:

    image_cut    the prompt positions the image tokens may not read, in the backbone's pass over the prefix
    action_cut   the prompt positions the action side may not read, at every Euler step

Each is bool [batch, prompt length] and an argument of the compiled program, not a constant of it: one program
serves every arm, and with both all false its masks are openpi's own. The arithmetic of the Euler step is
openpi's, unchanged (no cast, the same step size), so that with nothing cut the program returns what the
released policy's program returns.

The layout it relies on is openpi's, for π0.5 and for π0: the prefix is the image tokens of every camera in
turn, then the prompt. The action side is every token the action expert computes: the action tokens, and in
π0 the state token before them.

Runs in the `openpi` environment.
"""

from __future__ import annotations

import einops
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask

EULER_STEPS = 10  # as the released policies sample


def image_tokens(observation: _model.Observation, prefix_length: int) -> int:
    """How many prefix positions are image tokens: every position before the prompt, which ends the prefix."""
    cameras = len(observation.images)
    images = prefix_length - observation.tokenized_prompt.shape[1]
    if images <= 0 or images % cameras:
        raise ValueError("the prefix is not <cameras> x <image tokens> + <prompt tokens>")
    return images


def _prompt_columns(cut: jax.Array, prefix_length: int) -> jax.Array:
    """bool [batch, prompt length] -> bool [batch, prefix length]: the prompt ends the prefix."""
    return jnp.pad(cut, ((0, 0), (prefix_length - cut.shape[1], 0)))


def _encode(model, observation: _model.Observation, image_cut: jax.Array):
    """The backbone's pass over images and prompt, the image tokens not reading the prompt positions in
    `image_cut`. `observation` is preprocessed. Returns which prefix positions hold a token and the encoding
    (every layer's keys and values) the action side reads."""
    tokens, real, segments = model.embed_prefix(observation)
    may_read = make_attn_mask(real, segments)  # [batch, p, p]: may the token of a row read the token of a column
    positions = jnp.cumsum(real, axis=1) - 1
    length = tokens.shape[1]
    image = jnp.arange(length) < image_tokens(observation, length)  # bool [p]: the rows that are image tokens
    denied = image[None, :, None] & _prompt_columns(image_cut, length)[:, None, :]
    _, encoding = model.PaliGemma.llm([tokens, None], mask=may_read & ~denied, positions=positions)
    return real, encoding


def _denoise(model, observation: _model.Observation, real, encoding, noise, action_cut, num_steps: int):
    """openpi's Euler integration from `noise` (time 1) to the actions (time 0), the action side not reading
    the prompt positions in `action_cut`."""
    dt = -1.0 / num_steps
    batch = observation.state.shape[0]
    readable = real & ~_prompt_columns(action_cut, real.shape[1])  # [batch, p]

    def step(carry):
        x, time = carry
        suffix_tokens, suffix_real, suffix_segments, time_condition = model.embed_suffix(
            observation, x, jnp.broadcast_to(time, batch)
        )
        may_read = jnp.concatenate(
            [
                einops.repeat(readable, "b p -> b s p", s=suffix_tokens.shape[1]),
                make_attn_mask(suffix_real, suffix_segments),
            ],
            axis=-1,
        )
        positions = jnp.sum(real, axis=-1)[:, None] + jnp.cumsum(suffix_real, axis=-1) - 1
        (_, out), _ = model.PaliGemma.llm(
            [None, suffix_tokens],
            mask=may_read,
            positions=positions,
            kv_cache=encoding,
            adarms_cond=[None, time_condition],
        )
        v = model.action_out_proj(out[:, -model.action_horizon :])
        return x + dt * v, time + dt

    actions, _ = jax.lax.while_loop(lambda carry: carry[1] >= -dt / 2, step, (noise, 1.0))
    return actions


def sample_actions(
    model,
    observation: _model.Observation,
    noise: jax.Array,
    image_cut: jax.Array,
    action_cut: jax.Array,
    *,
    num_steps: int = EULER_STEPS,
) -> jax.Array:
    """One chunk of actions under an arm's cuts.

    model:      openpi's `Pi0` with a released checkpoint's parameters (π0.5 or π0).
    noise:      [batch, horizon, action_dim], the sample at time 1.
    image_cut:  bool [batch, prompt length], the prompt positions the image tokens may not read.
    action_cut: bool [batch, prompt length], the prompt positions the action side may not read.
    num_steps:  a Python number, as in openpi's sampler. With 1 the result is the noise minus the arm's
                velocity at the first Euler step.

    Returns the actions [batch, horizon, action_dim].
    """
    observation = _model.preprocess_observation(None, observation, train=False)
    real, encoding = _encode(model, observation, image_cut)
    return _denoise(model, observation, real, encoding, noise, action_cut, num_steps)


def sample_actions_replaced_encoding(
    model,
    observation: _model.Observation,
    noise: jax.Array,
    image_cut: jax.Array,
    action_cut: jax.Array,
    replaced: jax.Array,
) -> jax.Array:
    """`sample_actions`, with the encoding's keys and values at the prompt positions `replaced` (bool [batch,
    prompt length]) set to other numbers before the action side reads them: x -> -x - 1, in every layer.

    For the check of the direct cut: a token the action side does not read at all cannot change the chunk,
    whatever its keys and values are.
    """
    observation = _model.preprocess_observation(None, observation, train=False)
    real, (keys, values) = _encode(model, observation, image_cut)  # each [layers, batch, p, heads, width]
    where = _prompt_columns(replaced, real.shape[1])[None, :, :, None, None]
    encoding = (jnp.where(where, -keys - 1, keys), jnp.where(where, -values - 1, values))
    return _denoise(model, observation, real, encoding, noise, action_cut, EULER_STEPS)


def image_encoding(model, observation: _model.Observation, image_cut: jax.Array) -> tuple[jax.Array, jax.Array]:
    """What the backbone leaves at the image tokens under `image_cut`: their keys and values in every layer,
    float32, each [layers, batch, image tokens, heads, width].

    For the check of the grounding cut: tokens that do not read the instruction cannot change with its words.
    A camera slot without an image (the policies' third slot in LIBERO) holds padding, which no token reads;
    its positions are returned as zeros. (A padding position reads nothing at all, and openpi's attention
    then spreads evenly over every position, the instruction included: what it holds does change with the
    words, under any cut, and reaches nothing.)
    """
    observation = _model.preprocess_observation(None, observation, train=False)
    real, (keys, values) = _encode(model, observation, image_cut)
    images = image_tokens(observation, real.shape[1])
    holds_a_token = real[None, :, :images, None, None]
    return (
        jnp.where(holds_a_token, keys[:, :, :images], 0).astype(jnp.float32),
        jnp.where(holds_a_token, values[:, :, :images], 0).astype(jnp.float32),
    )
