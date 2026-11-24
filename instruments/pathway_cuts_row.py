"""The adapter of the pathway-cut rows: a released openpi checkpoint (π0.5 or π0) sampled under one arm's cut.

Everything but the sampler is the released row (`evaluation/rows/_openpi.py`): the same checkpoint, the same
input, the same noise, the same five executed actions. The sampler is `pathway_cuts_sampler.py`; what the arms
are, and what each row measures about itself at the first query of every episode, stands in `pathway_cuts.py`.
The arm is `row.json`: cut. These rows are instruments and belong to no table of the paper.

Runs in the `openpi` environment.
"""

from __future__ import annotations

import os
import types
from typing import Any, Dict, List

import numpy as np

from evaluation import interface
from instruments import pathway_cuts
from evaluation.rows import _openpi


class PathwayCut(_openpi.OpenpiRow):
    def __init__(self, row: Dict[str, Any]) -> None:
        self.arm = pathway_cuts.arm(row)  # a row.json without an arm stops here, before any weights load
        super().__init__(row)
        from openpi.models import tokenizer
        from openpi.shared import nnx_utils
        from instruments import pathway_cuts_sampler as sampler

        model = self.policy._model
        self.pi05 = bool(getattr(self.train_config.model, "pi05", False))

        def program(function):
            return nnx_utils.module_jit(types.MethodType(function, model))

        #: The arm's program: the cuts are its arguments, so the same compiled program gives every arm's chunk.
        self._sample = program(
            lambda model, observation, noise, image_cut, action_cut: sampler.sample_actions(
                model, observation, noise, image_cut, action_cut
            )
        )
        #: One Euler step of size 1 through the same function: the noise minus the velocity at the first step.
        self._first_step = program(
            lambda model, observation, noise, image_cut, action_cut: sampler.sample_actions(
                model, observation, noise, image_cut, action_cut, num_steps=1
            )
        )
        if self.arm == "direct":
            self._sample_replaced = program(
                lambda model, observation, noise, image_cut, action_cut, replaced: (
                    sampler.sample_actions_replaced_encoding(
                        model, observation, noise, image_cut, action_cut, replaced
                    )
                )
            )
        if self.arm == "grounding":
            self._image_encoding = program(
                lambda model, observation, image_cut: sampler.image_encoding(model, observation, image_cut)
            )
        self._euler_steps = sampler.EULER_STEPS
        self._tokenizer = tokenizer.PaligemmaTokenizer(self.train_config.model.max_token_len)
        self._instructions: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------------------------------- the instruction

    def instruction_of(self, instruction: str) -> Dict[str, Any]:
        """The prompt of one instruction as openpi's tokenizer gives it, and the positions that are cut."""
        if instruction not in self._instructions:
            pieces_model = self._tokenizer._tokenizer
            ids, real = self._tokenizer.tokenize(instruction)
            ids, real = np.asarray(ids), np.asarray(real).astype(bool)
            pieces = [pieces_model.id_to_piece(int(token)) if is_real else "" for token, is_real in zip(ids, real)]
            closing = pieces_model.decode([int(ids[np.flatnonzero(real)[-1]])]) if real.any() else ""
            told = pathway_cuts.instruction_tokens(ids, real, pieces_model.bos_id(), pieces, closing)
            told["prompt_tokens"], told["real"] = ids, real
            self._instructions[instruction] = told
        return self._instructions[instruction]

    def prepare(self, instructions: List[str]) -> None:
        for instruction in instructions:
            self.instruction_of(instruction)

    # ------------------------------------------------------------------------------------------- one query

    def act(self, observation: interface.Observation) -> interface.Actions:
        import jax
        import jax.numpy as jnp
        from openpi.models import model as _model

        told = self.instruction_of(observation.instruction)
        noise = _openpi.query_noise(self.episode, observation.query, self.horizon)
        inputs = self.policy._input_transform(dict(_openpi.request(observation)))
        # The positions that are cut are those of the prompt this very request gives the model.
        if not np.array_equal(np.asarray(inputs["tokenized_prompt"]), told["prompt_tokens"]) or not np.array_equal(
            np.asarray(inputs["tokenized_prompt_mask"]).astype(bool), told["real"]
        ):
            raise RuntimeError("the policy tokenised the instruction differently from the record's tokeniser")

        def model_observation(prompt_tokens: Any) -> Any:
            given = dict(inputs, tokenized_prompt=prompt_tokens)
            batch = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], given)
            return _model.Observation.from_dict(batch)

        image_cut, action_cut = (
            jnp.asarray(pathway_cuts.columns(cut, told["real"]))[np.newaxis] for cut in pathway_cuts.ARMS[self.arm]
        )
        model_input, model_noise = model_observation(inputs["tokenized_prompt"]), jnp.asarray(noise)[np.newaxis]
        chunk = np.asarray(self._sample(model_input, model_noise, image_cut, action_cut))
        actions = self._executed(model_input, chunk)

        if observation.query == 0:
            record = self._first_query(
                observation, noise, told, model_observation, model_input, model_noise, image_cut, action_cut, chunk
            )
        else:
            record = self._not_measured()
        record = {name: np.asarray(value, np.float32) for name, value in record.items()}
        return interface.Actions(actions, _openpi.EXECUTE, record)

    def _executed(self, model_input: Any, chunk: np.ndarray) -> np.ndarray:
        """A sampled chunk as the actions to execute, exactly as `Policy.infer` makes them: the state it passes
        to the output transform is the model input's own (float32, as the model was given it), not the transform's
        NumPy state. π0's output transform adds the state back to the actions, so the two differ in the last
        digits there, and an episode then goes its own way."""
        outputs = {"state": np.asarray(model_input.state[0, ...]), "actions": np.asarray(chunk[0, ...])}
        return np.asarray(self.policy._output_transform(outputs)["actions"])

    def _not_measured(self) -> Dict[str, np.ndarray]:
        record = {name: np.full((), np.nan, np.float32) for name in pathway_cuts.RECORDED[self.arm]}
        record["first_velocity"] = np.full((self.horizon, interface.ACTION_DIM), np.nan, np.float32)
        return record

    def _first_query(
        self, observation, noise, told, model_observation, model_input, model_noise, image_cut, action_cut, chunk
    ):
        """What the arm measures about itself on the episode's first observation (`pathway_cuts.py`). Raises
        `pathway_cuts.CheckFailed` when a cut does not do what its definition says."""
        import jax
        import jax.numpy as jnp

        record = self._not_measured()
        nothing = jnp.zeros_like(image_cut)
        instruction = jnp.asarray(told["real"])[np.newaxis]

        # With nothing cut, this instrument's program gives the released policy's own program's chunk.
        released = np.asarray(self.policy._sample_actions(jax.random.key(0), model_input, noise=model_noise))
        uncut = chunk if self.arm == "none" else np.asarray(self._sample(model_input, model_noise, nothing, nothing))
        record["check_released"] = np.float32(
            pathway_cuts.require_equal(
                "with nothing cut the instrument's chunk is not the released policy's", uncut, released
            )
        )
        # And the actions the row would execute from that chunk are those `Policy.infer` returns for the same request.
        by_infer = np.asarray(self.policy.infer(dict(_openpi.request(observation)), noise=noise)["actions"])
        record["check_released_actions"] = np.float32(
            pathway_cuts.require_equal(
                "with nothing cut the row's actions are not those Policy.infer returns",
                self._executed(model_input, uncut),
                by_infer,
            )
        )

        first = np.asarray(self._first_step(model_input, model_noise, image_cut, action_cut))
        record["first_velocity"] = (np.asarray(model_noise) - first)[0, :, : interface.ACTION_DIM].astype(np.float32)

        if self.arm == "direct":
            replaced = np.asarray(
                self._sample_replaced(model_input, model_noise, image_cut, action_cut, instruction)
            )
            record["check_replaced_encoding"] = np.float32(
                pathway_cuts.require_equal(
                    "under the direct cut the chunk changes with the instruction's keys and values", replaced, chunk
                )
            )
            replaced = np.asarray(self._sample_replaced(model_input, model_noise, nothing, nothing, instruction))
            record["control_replaced_encoding"] = np.float32(
                pathway_cuts.require_different(
                    "with nothing cut the chunk does not change with the instruction's keys and values",
                    replaced,
                    uncut,
                )
            )

        if self.arm == "grounding":
            other = model_observation(np.asarray(pathway_cuts.other_words(told["prompt_tokens"], told["real"])))

            def encodings(cut):
                return [
                    np.concatenate([np.asarray(part) for part in self._image_encoding(observation, cut)])
                    for observation in (model_input, other)
                ]

            record["check_swapped_words"] = np.float32(
                pathway_cuts.require_equal(
                    "under the grounding cut the image tokens' encoding changes with the instruction's words",
                    *encodings(image_cut),
                )
            )
            record["control_swapped_words"] = np.float32(
                pathway_cuts.require_different(
                    "with nothing cut the image tokens' encoding does not change with the instruction's words",
                    *encodings(nothing),
                )
            )
        return record

    # ------------------------------------------------------------------------------------------- the report

    def inference(self) -> Dict[str, Any]:
        settings = super().inference()
        settings["sampler"] = {"num_steps": self._euler_steps}
        settings["passes_per_query"] = (
            "1 encoding of the images and the instruction; 1 action-expert evaluation at each Euler step; at the "
            "first query of an episode the row's checks add further passes, which do not enter the actions"
        )
        return settings

    def method(self) -> Dict[str, Any]:
        image, action = pathway_cuts.ARMS[self.arm]
        action_side = "the action tokens" if self.pi05 else "the state token and the action tokens"
        return {
            "name": pathway_cuts.METHOD,
            "code": {
                "address": "this repository, instruments (pathway_cuts.py, pathway_cuts_sampler.py, "
                "pathway_cuts_row.py)",
                "revision": os.environ.get("PSG_REVISION", "unknown"),
            },
            "settings": {
                "status": "an instrument for the paper's pathway cuts (Sec. 3.3): one of the instruction's two "
                "ways to the action is cut; no row of the paper's tables",
                "arm": self.arm,
                "image_tokens_may_not_read_the_instruction": image,
                "action_side_may_not_read_the_instruction": action,
                "image_tokens": "the image tokens of every camera: every prefix position before the prompt",
                "action_side": action_side + ": every token the action expert computes",
                "instruction_is": "every prompt position that holds a token, from <bos> to the closing newline, "
                "as the policy's own input transform tokenised this request",
                "how": "attention masks passed to openpi's transformer; every layer and head, every Euler step "
                "of every query, for the whole episode; the model's velocity under the cut is integrated in "
                "every action dimension",
                "instruction": {
                    instruction: {key: told[key] for key in ("positions", "ids", "pieces")}
                    for instruction, told in sorted(self._instructions.items())
                },
                "checks": {
                    "when": "the first query of every episode, on its own observation and noise, in this "
                    "process; a failed check stops the episode with an error",
                    "tolerance": "zero: every comparison is inside one process and has to hold bit for bit",
                    "measured": {name: pathway_cuts.RECORDED_IS[name] for name in pathway_cuts.RECORDED[self.arm]},
                },
                "recorded_per_query": "%s; NaN at every query but the first of an episode"
                % ", ".join(pathway_cuts.RECORDED[self.arm]),
            },
        }


def load(row: dict) -> PathwayCut:
    return PathwayCut(row)
