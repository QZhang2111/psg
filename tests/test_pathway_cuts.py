"""The pathway cuts: what the definition's own functions do, what the rows declare. Standard library and NumPy
only.

The masks themselves, on a model, are `test_pathway_cuts_sampler.py` (host). What a row does at a query is
checked by the row itself, at the first query of every episode (`instruments/pathway_cuts.py`).
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from evaluation import layout
from instruments import pathway_cuts

ROWS = {
    "pi05_cut_none": ("pi05", "none"), "pi05_cut_grounding": ("pi05", "grounding"),
    "pi05_cut_direct": ("pi05", "direct"),
    "pi0_cut_none": ("pi0", "none"), "pi0_cut_grounding": ("pi0", "grounding"), "pi0_cut_direct": ("pi0", "direct"),
}  # fmt: skip

BOS = 2
#           <bos> pick  up   the  soup  \n   padding
IDS = [BOS, 40, 41, 42, 43, 108, 0, 0]
REAL = [True] * 6 + [False] * 2
PIECES = ["<bos>", "▁pick", "▁up", "▁the", "▁soup", "\n", "", ""]


# ------------------------------------------------------------------------------------------- the rows


def test_the_rows_are_the_released_rows_but_for_the_cut():
    for row, (model, arm) in ROWS.items():
        definition, released = layout.load_row(row), layout.load_row(model)
        assert pathway_cuts.arm(definition) == arm
        assert definition["method"] == pathway_cuts.METHOD
        for key in ("environment", "checkpoint"):  # the same model in the same environment
            assert definition[key] == released[key], (row, key)


def test_only_these_rows_state_a_cut_and_the_method_does_not_know_them():
    assert sorted(row for row in layout.rows() if "cut" in layout.load_row(row)) == sorted(ROWS)
    for path in sorted((layout.REPO / "psg").glob("*.py")) + [layout.ROWS / "pi05_psg" / "adapter.py"]:
        text = path.read_text()
        assert "pathway_cuts" not in text, path
        assert not re.search(r"^\s*(from|import)\s+instruments\b", text, re.MULTILINE), path


def test_a_row_without_a_known_arm_does_not_load():
    for cut in (None, "both", "name", True):
        with pytest.raises(SystemExit):
            pathway_cuts.arm({"row": "x", "cut": cut})


def test_the_arms_cut_one_read_each_and_say_what_they_record():
    assert pathway_cuts.ARMS == {"none": (False, False), "grounding": (True, False), "direct": (False, True)}
    for arm, names in pathway_cuts.RECORDED.items():
        assert arm in pathway_cuts.ARMS
        assert set(names) <= set(pathway_cuts.RECORDED_IS)
        assert {"first_velocity", "check_released", "check_released_actions"} <= set(names)


# ------------------------------------------------------------------------------------------- the instruction


def test_the_instruction_is_every_token_from_bos_to_the_closing_newline():
    told = pathway_cuts.instruction_tokens(IDS, REAL, BOS, PIECES, "\n")
    assert told["positions"] == [0, 1, 2, 3, 4, 5]
    assert told["ids"] == IDS[:6]
    assert told["pieces"] == ["<bos>", "▁pick", "▁up", "▁the", "▁soup", "\\n"]
    assert pathway_cuts.columns(True, REAL).tolist() == REAL
    assert pathway_cuts.columns(False, REAL).tolist() == [False] * 8


@pytest.mark.parametrize(
    "ids, real, closing",
    [
        ([7] + IDS[1:], REAL, "\n"),  # no <bos> first
        (IDS, REAL, "soup"),  # no closing newline
        (IDS, [True, True, False, True, True, True, False, False], "\n"),  # a hole
        (IDS, [False] + REAL[1:], "\n"),  # not from position 0
        (IDS, [False] * 8, "\n"),  # nothing
        ([BOS, 108] + [0] * 6, [True, True] + [False] * 6, "\n"),  # no word
    ],
)
def test_a_prompt_of_another_form_is_refused(ids, real, closing):
    with pytest.raises(ValueError):
        pathway_cuts.instruction_tokens(ids, real, BOS, PIECES, closing)


def test_other_words_keep_the_length_the_ends_and_the_padding():
    other = pathway_cuts.other_words(IDS, REAL)
    assert other == [BOS, 43, 42, 41, 40, 108, 0, 0]
    assert len(other) == len(IDS) and sorted(other) == sorted(IDS)
    # a sentence that reads the same backwards is moved on by one position instead
    assert pathway_cuts.other_words([BOS, 40, 41, 40, 108, 0], [True] * 5 + [False]) == [BOS, 41, 40, 40, 108, 0]
    with pytest.raises(ValueError):
        pathway_cuts.other_words([BOS, 40, 40, 40, 108], [True] * 5)


# ------------------------------------------------------------------------------------------- the checks' own rule


def test_equal_means_bit_for_bit():
    a = np.asarray([1.0, 2.0], np.float32)
    assert pathway_cuts.require_equal("x", a, a.copy()) == 0.0
    last_bit = a.copy()
    last_bit[1] = np.nextafter(np.float32(2.0), np.float32(3.0))
    with pytest.raises(pathway_cuts.CheckFailed, match="largest absolute difference"):
        pathway_cuts.require_equal("x", a, last_bit)
    with pytest.raises(pathway_cuts.CheckFailed):
        pathway_cuts.require_equal("x", a, a.astype(np.float64))
    with pytest.raises(ValueError):
        pathway_cuts.require_equal("x", a, a[:1])


def test_a_control_has_to_differ():
    a = np.asarray([1.0, 2.0], np.float32)
    assert pathway_cuts.require_different("x", a, a + 1) == 1.0
    with pytest.raises(pathway_cuts.CheckFailed):
        pathway_cuts.require_different("x", a, a.copy())
    assert pathway_cuts.require_different("x", a, np.asarray([1.0, np.nan], np.float32)) == float("inf")

