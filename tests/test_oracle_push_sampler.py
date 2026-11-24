"""The oracle push's sampler on a randomly initialised small π0.5, no weights (the model of `test_psg.py`).

A host test: it builds a model and compiles JAX (openpi's sampler and the instrument's, once each).
`tests/conftest.py` leaves it out unless PSG_HOST_TESTS=1.
"""

from __future__ import annotations

import dataclasses
import types

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import openpi.models.pi0_config as pi0_config
from openpi.shared import nnx_utils
from instruments import oracle_push
from instruments import oracle_push_sampler
from psg import sampler


@pytest.fixture(scope="module")
def pi05():
    config = pi0_config.Pi0Config(
        paligemma_variant="dummy", action_expert_variant="dummy", pi05=True, action_horizon=10,
        discrete_state_input=False, dtype="float32",
    )  # fmt: skip
    model = config.create(jax.random.key(0))
    # Open the gates of the action expert, as `test_psg.py` does and explains.
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
    ones = config.fake_obs(2)
    keys = jax.random.split(jax.random.key(3), len(ones.images) + 1)
    observation = dataclasses.replace(
        ones,
        images={
            name: jax.random.uniform(key, image.shape, image.dtype, -1.0, 1.0)
            for key, (name, image) in zip(keys[1:], ones.images.items())
        },
        tokenized_prompt=jax.random.randint(keys[0], ones.tokenized_prompt.shape, 2, 1000, ones.tokenized_prompt.dtype),
    )
    noise = jax.random.normal(jax.random.key(1), (2, config.action_horizon, config.action_dim))
    program = nnx_utils.module_jit(
        types.MethodType(
            lambda model, push, b: oracle_push_sampler.sample_actions(model, observation, noise, push, b=b), model
        )
    )
    released = np.asarray(nnx_utils.module_jit(model.sample_actions)(jax.random.key(2), observation, noise=noise))

    def run(push, b):
        actions, kept = program(jnp.asarray(push, jnp.float32), jnp.float32(b))
        kept = np.asarray(kept)
        assert kept.shape[2] == len(oracle_push.RECORDED)
        return (np.asarray(actions),) + tuple(kept[:, :, index] for index in range(kept.shape[2]))

    return config, np.asarray(noise), released, run


def a_push(config):
    """Two rows, two different pushes, as the adapter builds them."""
    scale = np.asarray([0.8, 0.6])
    return np.stack(
        [oracle_push.velocity([0.3, 0.0], scale, config.action_dim), oracle_push.velocity([-0.1, 0.2], scale, config.action_dim)]
    )


def test_without_a_push_and_with_b_zero_it_is_openpis_sampler(pi05):
    config, _, released, run = pi05
    actions, _, v_o, _ = run(np.zeros((2, config.action_dim)), 0.0)
    np.testing.assert_allclose(actions, released, rtol=1e-4, atol=1e-4)
    assert not v_o.any()


def test_the_actions_are_the_equation_integrated_and_the_push_is_recorded(pi05):
    config, noise, released, run = pi05
    push = a_push(config)
    actions, v, v_o, v_d = run(push, 1.0)
    assert np.abs(v_d).max() > 0  # the cut is applied
    for step in range(sampler.EULER_STEPS):  # the same push at every step and for every action of the chunk
        np.testing.assert_array_equal(v_o[step], np.broadcast_to(push[:, None, :7], v_o[step].shape))
    v_tilde = v + v_o - 1.0 * v_d
    v_tilde[..., sampler.MOTION] = v[..., sampler.MOTION]  # the gripper keeps v
    expected = noise[..., :7] - v_tilde.sum(axis=0) / sampler.EULER_STEPS
    np.testing.assert_allclose(actions[..., :7], expected, rtol=1e-4, atol=1e-4)
    assert np.abs(actions[..., :2] - released[..., :2]).max() > 0


def test_the_push_moves_the_sample_by_minus_itself_beyond_what_the_model_answers(pi05):
    """The term's own contribution over the ten steps is -v_o: the sign the instrument's definition rests on."""
    config, _, _, run = pi05
    push = a_push(config)
    _, _, v_o, _ = run(push, 1.0)
    own = -v_o.sum(axis=0) / sampler.EULER_STEPS  # dt * sum over the steps, [rows, horizon, 7]
    np.testing.assert_allclose(own[..., :2] * np.asarray([0.8, 0.6]), np.asarray([[0.3, 0.0], [-0.1, 0.2]])[:, None, :].repeat(10, axis=1), atol=1e-6)
    assert not own[..., 2:].any()
