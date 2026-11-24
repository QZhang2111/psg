"""PSG's sampler on a randomly initialised small π0.5 (`dummy` Gemma configs; the image encoder keeps its full
size), no weights.

A host test: it builds a model and compiles JAX. `tests/conftest.py` leaves it out unless PSG_HOST_TESTS=1.
An earlier form of this file, which compiled the sampler once per call, reached 18 GB and was killed. This form
compiles openpi's sampler and PSG's once each and reuses them; it peaked at 3.4 GB on a host.

**Why the model's zero-initialised parameters are filled with random values.** In π0.5 every block of the action
expert adds its attention and its MLP output through a gate computed from the time embedding by a layer whose
weights start at zero (`gemma.RMSNorm` with a condition, `_gated_residual`). On a freshly initialised model the
gates are closed: the velocity does not depend on anything the action tokens read, and both cuts give exactly
zero, whatever the precision. The first two host runs showed just that (v_g and v_d exactly zero, in bfloat16
and in float32). With the gates open, a cut of one token has to show; if it does not, the mask is not applied.

The model computes in float32 and the observation is random, so that small effects are not rounded away.
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
from psg import sampler

MOTION = slice(0, sampler.MOTION)


@pytest.fixture(scope="module")
def pi05():
    config = pi0_config.Pi0Config(
        paligemma_variant="dummy", action_expert_variant="dummy", pi05=True, action_horizon=10,
        discrete_state_input=False, dtype="float32",
    )  # fmt: skip
    model = config.create(jax.random.key(0))
    # Open the gates: every parameter that starts at zero gets random values (see the module's docstring).
    # The parameters are replaced the way openpi's own `BaseModelConfig.load` replaces them.
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
    # openpi's `module_jit` passes the parameters as arguments; closing over the model would compile them in
    # as constants.
    psg = nnx_utils.module_jit(
        types.MethodType(
            lambda model, name, a, b: sampler.sample_actions(model, observation, noise, name, a=a, b=b), model
        )
    )
    released = np.asarray(nnx_utils.module_jit(model.sample_actions)(jax.random.key(2), observation, noise=noise))

    def run(tokens_per_row, a, b):
        """PSG with the name at the given instruction-token indices of each of the two rows. Returns the actions
        and v, v_g, v_d, each [Euler steps, rows, horizon, 7]."""
        name = np.zeros((2, config.max_token_len), bool)
        for row, tokens in enumerate(tokens_per_row):
            name[row, tokens] = True
        actions, kept = psg(jnp.asarray(name), jnp.float32(a), jnp.float32(b))
        kept = np.asarray(kept)
        assert kept.shape[2] == len(sampler.RECORDED)
        return (np.asarray(actions),) + tuple(kept[:, :, index] for index in range(kept.shape[2]))

    return config, observation, np.asarray(noise), released, run


def test_prefix_parts(pi05):
    config, observation, _, _, _ = pi05
    prefix = 3 * 256 + config.max_token_len
    main_image, instruction = sampler.prefix_parts(_model.preprocess_observation(None, observation, train=False), prefix)
    assert np.flatnonzero(main_image).tolist() == list(range(256))
    assert np.flatnonzero(instruction).tolist() == list(range(768, prefix))


def test_without_guidance_it_is_openpis_sampler(pi05):
    """With a = b = 0 on this small model, which computes in float32: there the two programs agree within the
    tolerance. The released checkpoint's velocity is bfloat16 and the two then differ in rounding; that is
    measured on the released weights (`psg/sampler.py` gives the number), not here."""
    _, _, _, released, run = pi05
    actions, _, _, _ = run([[2, 3], [2]], 0.0, 0.0)
    np.testing.assert_allclose(actions, released, rtol=1e-4, atol=1e-4)


def test_the_actions_are_the_equation_integrated(pi05):
    _, _, noise, released, run = pi05
    a, b = 5.0, 1.0
    actions, v, v_g, v_d = run([[2, 3], [4]], a, b)
    v_tilde = v + a * v_g - b * v_d
    v_tilde[..., sampler.MOTION] = v[..., sampler.MOTION]  # the gripper keeps v
    expected = noise[..., :7] - v_tilde.sum(axis=0) / sampler.EULER_STEPS
    np.testing.assert_allclose(actions[..., :7], expected, rtol=1e-4, atol=1e-4)
    assert np.abs(actions[..., MOTION] - released[..., MOTION]).max() > 0


def test_each_cut_changes_the_velocity(pi05):
    config, _, _, _, run = pi05
    every_token = list(range(config.max_token_len))
    _, _, v_g, v_d = run([[2], every_token], 5.0, 1.0)
    assert np.abs(v_d[0][..., MOTION]).max() > 0, "the direct cut left the velocity unchanged: is its mask applied?"
    assert np.abs(v_g[0, 1][..., MOTION]).max() > 0, "the grounding cut of the whole instruction changed nothing"
    assert np.abs(v_g[0, 0][..., MOTION]).max() > 0, "the grounding cut of one name token changed nothing"


def test_the_name_enters_the_grounding_contribution_only(pi05):
    _, _, _, _, run = pi05
    _, v_one, g_one, d_one = run([[2, 3], [4]], 5.0, 1.0)
    _, v_other, g_other, d_other = run([[9], [4]], 5.0, 1.0)
    # At the first Euler step both runs are at the same sample: v and v_d do not know the name, v_g does.
    np.testing.assert_allclose(v_one[0], v_other[0], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(d_one[0], d_other[0], rtol=1e-5, atol=1e-5)
    assert np.abs(g_one[0, 0] - g_other[0, 0]).max() > 0


def test_no_name_no_grounding_contribution(pi05):
    _, _, _, _, run = pi05
    _, _, v_g, _ = run([[], [2]], 5.0, 1.0)
    scale = np.abs(v_g[:, 1]).max()
    assert scale > 0  # row 1 has one name token
    assert np.abs(v_g[:, 0]).max() <= 1e-5 * max(scale, 1.0)  # row 0 has none: v_g is zero at every step


def test_rows_of_a_batch_do_not_mix(pi05):
    _, _, _, _, run = pi05
    both, _, _, _ = run([[2, 3], [4]], 5.0, 1.0)
    swapped, _, _, _ = run([[2, 3], [7]], 5.0, 1.0)
    np.testing.assert_allclose(both[0], swapped[0], rtol=1e-5, atol=1e-5)
    assert np.abs(both[1] - swapped[1]).max() > 0

