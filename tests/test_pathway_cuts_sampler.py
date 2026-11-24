"""The pathway cuts' sampler on randomly initialised small models, no weights: a π0.5 and a π0.

A host test: it builds models and compiles JAX (per model: openpi's sampler and four programs of the
instrument, each once; for π0.5 also PSG's). `tests/conftest.py` leaves it out unless PSG_HOST_TESTS=1.

The models are those of `test_psg.py` (`dummy` Gemma configs, float32, the zero-initialised parameters filled
so that the action expert's gates are open). The observation is LIBERO's layout: two cameras hold an image,
the third slot is padding.
"""

from __future__ import annotations

import dataclasses
import types

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as _model
import openpi.models.pi0_config as pi0_config
from openpi.shared import nnx_utils
from instruments import pathway_cuts
from instruments import pathway_cuts_sampler as cuts

ROWS = 2  # the batch: two observations, to see that each is cut by its own columns


@pytest.fixture(scope="module", params=["pi05", "pi0"])
def built(request):
    pi05 = request.param == "pi05"
    config = pi0_config.Pi0Config(
        paligemma_variant="dummy", action_expert_variant="dummy", pi05=pi05, action_horizon=10,
        discrete_state_input=False, dtype="float32",
    )  # fmt: skip
    model = config.create(jax.random.key(0))
    graphdef, state = nnx.split(model)
    leaves, tree = jax.tree.flatten(state.to_pure_dict())
    leaf_keys = jax.random.split(jax.random.key(4), len(leaves))
    leaves = [
        0.2 * jax.random.normal(key, leaf.shape, leaf.dtype)
        if jnp.issubdtype(leaf.dtype, jnp.floating) and not bool(jnp.any(leaf))
        else leaf
        for key, leaf in zip(leaf_keys, leaves)
    ]
    state.replace_by_pure_dict(jax.tree.unflatten(tree, leaves))
    model = nnx.merge(graphdef, state)

    ones = config.fake_obs(ROWS)
    keys = jax.random.split(jax.random.key(3), len(ones.images) + 2)
    cameras = list(ones.images)
    length = config.max_token_len
    words = 12  # the prompt: <bos>, ten words, the closing newline; the rest is padding
    real = np.zeros((ROWS, length), bool)
    real[:, :words] = True
    tokens = np.asarray(jax.random.randint(keys[0], (ROWS, length), 2, 1000, ones.tokenized_prompt.dtype))
    observation = dataclasses.replace(
        ones,
        images={
            name: jax.random.uniform(key, image.shape, image.dtype, -1.0, 1.0)
            for key, (name, image) in zip(keys[2:], ones.images.items())
        },
        image_masks={name: jnp.full((ROWS,), name != cameras[-1]) for name in cameras},  # the third slot is padding
        state=jax.random.uniform(keys[1], ones.state.shape, ones.state.dtype, -1.0, 1.0),
        tokenized_prompt=jnp.asarray(tokens),
        tokenized_prompt_mask=jnp.asarray(real),
    )
    noise = jax.random.normal(jax.random.key(1), (ROWS, config.action_horizon, config.action_dim))

    def program(function):
        return nnx_utils.module_jit(types.MethodType(function, model))

    sample = program(
        lambda model, image_cut, action_cut: cuts.sample_actions(model, observation, noise, image_cut, action_cut)
    )
    first_step = program(
        lambda model, image_cut, action_cut: cuts.sample_actions(
            model, observation, noise, image_cut, action_cut, num_steps=1
        )
    )
    replaced = program(
        lambda model, image_cut, action_cut, where: cuts.sample_actions_replaced_encoding(
            model, observation, noise, image_cut, action_cut, where
        )
    )
    encoding = program(
        lambda model, prompt, image_cut: cuts.image_encoding(
            model, dataclasses.replace(observation, tokenized_prompt=prompt), image_cut
        )
    )
    released = np.asarray(nnx_utils.module_jit(model.sample_actions)(jax.random.key(2), observation, noise=noise))

    def call(function, *arguments):
        out = function(*(jnp.asarray(argument) for argument in arguments))
        return tuple(np.asarray(part) for part in out) if isinstance(out, tuple) else np.asarray(out)

    return types.SimpleNamespace(
        pi05=pi05, config=config, model=model, observation=observation, noise=np.asarray(noise), tokens=tokens,
        real=real, nothing=np.zeros_like(real), released=released,
        sample=lambda *a: call(sample, *a), first_step=lambda *a: call(first_step, *a),
        replaced=lambda *a: call(replaced, *a), encoding=lambda *a: call(encoding, *a),
    )  # fmt: skip


def test_the_layout_is_the_cameras_then_the_prompt(built):
    observation = _model.preprocess_observation(None, built.observation, train=False)
    prefix = 3 * 256 + built.config.max_token_len
    assert cuts.image_tokens(observation, prefix) == 3 * 256
    with pytest.raises(ValueError):
        cuts.image_tokens(observation, prefix + 1)
    suffix, _, _, _ = built.model.embed_suffix(observation, jnp.asarray(built.noise), jnp.ones((ROWS,)))
    assert suffix.shape[1] == built.config.action_horizon + (0 if built.pi05 else 1)  # π0: the state token first


def test_with_nothing_cut_it_is_openpis_sampler_bit_for_bit(built):
    chunk = built.sample(built.nothing, built.nothing)
    assert chunk.dtype == built.released.dtype
    np.testing.assert_array_equal(chunk, built.released)


def test_each_cut_changes_the_chunk_and_the_two_differ(built):
    grounding = built.sample(built.real, built.nothing)
    direct = built.sample(built.nothing, built.real)
    assert np.abs(grounding - built.released).max() > 0, "the grounding cut changed nothing: is its mask applied?"
    assert np.abs(direct - built.released).max() > 0, "the direct cut changed nothing: is its mask applied?"
    assert np.abs(grounding - direct).max() > 0


def test_an_observation_is_cut_by_its_own_columns(built):
    only_first = built.nothing.copy()
    only_first[0] = built.real[0]
    for chunk in (built.sample(only_first, built.nothing), built.sample(built.nothing, only_first)):
        np.testing.assert_array_equal(chunk[1], built.released[1])
        assert np.abs(chunk[0] - built.released[0]).max() > 0


def test_a_cut_of_padding_positions_cuts_nothing(built):
    padding = ~built.real
    np.testing.assert_array_equal(built.sample(padding, padding), built.released)


def test_under_the_direct_cut_the_instructions_keys_and_values_do_not_reach_the_chunk(built):
    """The action side's attention on the instruction is exactly zero: what stands there cannot matter."""
    direct = built.sample(built.nothing, built.real)
    np.testing.assert_array_equal(built.replaced(built.nothing, built.real, built.real), direct)
    # the control: read, the same replacement changes the chunk
    assert np.abs(built.replaced(built.nothing, built.nothing, built.real) - built.released).max() > 0
    # and position by position: a position that is not cut is still read
    half = built.real.copy()
    half[:, 6:] = False
    rest = built.real & ~half
    assert np.abs(built.replaced(built.nothing, half, rest) - built.sample(built.nothing, half)).max() > 0
    np.testing.assert_array_equal(built.replaced(built.nothing, half, half), built.sample(built.nothing, half))


def test_under_the_grounding_cut_the_image_tokens_do_not_change_with_the_words(built):
    other = np.stack([pathway_cuts.other_words(ids, real) for ids, real in zip(built.tokens, built.real)])
    other = other.astype(built.tokens.dtype)
    assert (other != built.tokens).any(axis=1).all()
    for own, swapped in zip(built.encoding(built.tokens, built.real), built.encoding(other, built.real)):
        assert own.shape[2] == 3 * 256 and own.dtype == np.float32
        np.testing.assert_array_equal(own, swapped)
        assert np.abs(own[:, :, : 2 * 256]).max() > 0  # the two cameras' tokens
        assert not own[:, :, 2 * 256 :].any()  # the padding slot is returned as zeros
    # the control: with nothing cut the image tokens do read the words
    for own, swapped in zip(built.encoding(built.tokens, built.nothing), built.encoding(other, built.nothing)):
        assert np.abs(own - swapped).max() > 0


def test_one_step_of_size_one_gives_the_velocity_at_the_first_step(built):
    """noise - (one step) is the cut model's velocity at (noise, time 1); for π0.5 PSG's sampler records the
    same velocity for the same cut, from its own program."""
    if not built.pi05:
        pytest.skip("PSG's sampler, which this is laid beside, is defined on π0.5")
    from psg import sampler as psg

    name = jnp.zeros(built.real.shape, bool)
    recorded = nnx_utils.module_jit(
        types.MethodType(
            lambda model: psg.sample_actions(
                model, built.observation, jnp.asarray(built.noise), name, a=0.0, b=1.0
            ),
            built.model,
        )
    )()[1]
    v, _, v_d = (np.asarray(recorded)[0, :, index] for index in range(3))  # the first Euler step: [rows, horizon, 7]
    uncut = (built.noise - built.first_step(built.nothing, built.nothing))[..., :7]
    direct = (built.noise - built.first_step(built.nothing, built.real))[..., :7]
    np.testing.assert_allclose(uncut, v, rtol=0, atol=1e-5)
    np.testing.assert_allclose(direct, v - v_d, rtol=0, atol=1e-5)
    assert np.abs(v_d).max() > 1e-3  # the cut is not a no-op here
