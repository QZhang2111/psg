"""An instrument, not the method: one of the instruction's two ways to the action is cut, and the policy is run.

The instruction reaches the action of π0.5 and of π0 along two pathways (the paper's Sec. 3.3):

    grounding   the image tokens read the instruction in the backbone, and the action side reads the image tokens
    direct      the action side reads the instruction tokens themselves

An arm cuts one of the two reads, in every layer and head, at every Euler step of every query, for the whole
episode, and leaves everything else as the released policy has it:

    none        nothing is cut: the released policy through this instrument's own program
    grounding   the image tokens of both cameras may not read the instruction
    direct      the action side may not read the instruction: the action tokens, and in π0 the state token,
                which π0 computes with them

"The instruction" is the whole sentence as the model was given it: every prompt position from `<bos>` to the
closing newline. These are not PSG's cuts: PSG's grounding cut (`psg/sampler.py`) denies the main
camera's tokens the name tokens only, and PSG changes the six motion dimensions of the velocity, where an arm
here integrates the cut model's own velocity in every dimension. The method imports nothing from here.

A cut is an attention mask passed to openpi's own transformer (`pathway_cuts_sampler.py`); the model and its
parameters are not touched. The rows are `evaluation/rows/<model>_cut_<arm>/`, their adapter is
`pathway_cuts_row.py`. They belong to no table of the paper.

**What an arm shows about itself before its episodes are believed.** At the first query of every episode, on
that episode's own first observation and noise, the row measures the following in its own process and stops
the episode with an error when one does not hold (a unit with a failed episode is not collected, so an arm
whose check fails leaves no table). Each comparison is inside one process, and each is exact: the tolerance
is zero.

    released    with nothing cut, this instrument's program returns the chunk of the released policy's own
                program (`Policy`'s jitted `sample_actions`), bit for bit, and the actions the row would execute
                from it are the actions `Policy.infer` returns for the same request, bit for bit (the way from a
                chunk to actions is the row's own code, and in π0 it adds the state back). Every arm measures
                both: the cuts are arguments of one compiled program, which the arm's own chunk comes from.
    direct      the action side's attention on the instruction is exactly zero. Shown by what follows from
                it: the arm's chunk does not change by one bit when the instruction's keys and values in the
                backbone's encoding are replaced by other numbers before the action side reads them. Beside
                it, the control: with nothing cut the same replacement changes the chunk.
    grounding   what the backbone leaves at the image tokens (keys and values, every layer) does not change
                by one bit when the instruction's words are replaced by others of equal length. Beside it,
                the control: with nothing cut the same replacement changes them.

The measured numbers go into the episode's trace (`RECORDED`), with the arm's velocity at the first Euler
step of the first query.

This module is the definition and needs NumPy only; the sampler is `pathway_cuts_sampler.py` (JAX).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

METHOD = "pathway-cut"

#: arm -> (the image tokens may not read the instruction, the action side may not read the instruction)
ARMS = {"none": (False, False), "grounding": (True, False), "direct": (False, True)}

#: What a row leaves in the trace at every query, by arm (NaN at every query but an episode's first).
RECORDED = {
    "none": ("first_velocity", "check_released", "check_released_actions"),
    "grounding": (
        "first_velocity", "check_released", "check_released_actions", "check_swapped_words", "control_swapped_words",
    ),
    "direct": (
        "first_velocity", "check_released", "check_released_actions", "check_replaced_encoding",
        "control_replaced_encoding",
    ),
}

RECORDED_IS = {
    "first_velocity": "the arm's velocity at the first Euler step, [horizon, 7], as the model's velocity: the "
    "noise minus the sample after one Euler step of size 1 through the arm's program",
    "check_released": "largest absolute difference between this instrument's chunk with nothing cut and the "
    "released policy's own program's chunk, same observation and noise; the episode stops unless they are "
    "equal bit for bit",
    "check_released_actions": "largest absolute difference between the actions this row would execute from its "
    "chunk with nothing cut and the actions `Policy.infer` returns for the same request and noise; the episode "
    "stops unless they are equal bit for bit",
    "check_replaced_encoding": "largest absolute difference between the arm's chunk and the arm's chunk with "
    "the instruction's keys and values replaced in the encoding; the episode stops unless they are equal bit "
    "for bit",
    "control_replaced_encoding": "the same difference with nothing cut; the episode stops unless it is above "
    "zero",
    "check_swapped_words": "largest absolute difference of the image tokens' keys and values, over every "
    "layer, between the instruction and the instruction with its words in another order, under the arm's "
    "cut; the episode stops unless they are equal bit for bit",
    "control_swapped_words": "the same difference with nothing cut; the episode stops unless it is above zero",
}


def arm(row: Dict[str, Any]) -> str:
    """The row's arm from its definition (`row.json`: cut), checked."""
    name = row.get("cut")
    if name not in ARMS:
        raise SystemExit(
            "row %r: cut is %r in row.json; it must be one of %s" % (row.get("row"), name, ", ".join(ARMS))
        )
    return name


def instruction_tokens(
    ids: Sequence[int], real: Sequence[bool], bos: int, pieces: Sequence[str], closing: str
) -> Dict[str, Any]:
    """The prompt positions that are the instruction, from one tokenized prompt as the model is given it.

    ids, real: the prompt's token ids and which of its positions hold a token (the rest is padding), both of
               the prompt's full length.
    bos:       the tokenizer's id of `<bos>`.
    pieces:    the tokenizer's piece of every id in `ids` (padding included; only the real ones are read).
    closing:   the tokenizer's decoding of the last real token alone.

    The instruction is every real position. It has to be one block from position 0, begin with `<bos>` and
    end with the newline that closes the prompt; anything else is refused, because the cut would then not be
    the sentence the definition names.
    """
    ids = [int(token) for token in ids]
    positions = [index for index, is_real in enumerate(real) if bool(is_real)]
    if len(ids) != len(list(real)) or len(ids) != len(list(pieces)):
        raise ValueError("ids, real and pieces must have the prompt's length")
    if not positions or positions != list(range(len(positions))):
        raise ValueError("the prompt's tokens are not one block from position 0: %s" % positions)
    if ids[0] != int(bos):
        raise ValueError("the prompt does not begin with <bos>: first id %d" % ids[0])
    if closing != "\n":
        raise ValueError("the prompt does not end with the closing newline: its last token decodes to %r" % closing)
    if len(positions) < 3:
        raise ValueError("the prompt holds no word between <bos> and the closing newline")
    return {
        "positions": positions,
        "ids": [ids[index] for index in positions],
        "pieces": ["<bos>"] + [str(pieces[index]) for index in positions[1:-1]] + ["\\n"],
    }


def other_words(ids: Sequence[int], real: Sequence[bool]) -> List[int]:
    """The prompt with other words of equal length: the same token ids with the words in reverse order.

    `<bos>`, the closing newline and the padding stay where they are, so the prompt has the same length and
    the same positions hold a token. Where the reverse order is the same sentence, the words are moved on by
    one position instead; a sentence of one repeated token is refused.
    """
    ids = [int(token) for token in ids]
    count = int(np.sum(np.asarray(real, bool)))
    words = ids[1 : count - 1]
    for candidate in (words[::-1], words[1:] + words[:1]):
        if candidate != words:
            return ids[:1] + candidate + ids[count - 1 :]
    raise ValueError("the instruction's words give no other order: %s" % words)


def columns(cut: bool, real: Sequence[bool]) -> np.ndarray:
    """The prompt positions one side may not read, bool [prompt length]: the instruction, or none."""
    real = np.asarray(real, bool)
    return real.copy() if cut else np.zeros_like(real)


def max_abs_difference(a: Any, b: Any) -> float:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("shapes differ: %s and %s" % (a.shape, b.shape))
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0


class CheckFailed(RuntimeError):
    """A cut did not do what its definition says, on this episode's first observation."""


def require_equal(what: str, a: Any, b: Any) -> float:
    """The largest absolute difference of two arrays that have to be equal bit for bit; raises when they are not."""
    a, b = np.asarray(a), np.asarray(b)
    difference = max_abs_difference(a, b)
    if a.dtype != b.dtype or not np.array_equal(a, b):
        raise CheckFailed(
            "pathway cut, check failed: %s; largest absolute difference %.6g (dtypes %s, %s)"
            % (what, difference, a.dtype, b.dtype)
        )
    return difference


def require_different(what: str, a: Any, b: Any) -> float:
    """The largest absolute difference of two arrays that must not be equal (a control); raises when they are."""
    if np.array_equal(np.asarray(a), np.asarray(b)):
        raise CheckFailed("pathway cut, control failed: %s; the two are equal" % what)
    difference = max_abs_difference(a, b)
    return difference if np.isfinite(difference) else float("inf")  # NaN in a record means "not measured here"
