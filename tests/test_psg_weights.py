"""The rows that are `pi05_psg` but for PSG's two weights: what they declare. NumPy and the standard library.

That such a row runs the method's own program at the method's weights was shown on a card, on traces: the row
set to a = 5, b = 1 gave the same traces as `pi05_psg`'s episodes. The sampler's arithmetic for any a and b is
`test_psg.py` (host).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation import layout
from evaluation.rows.pi05_psg import adapter

ARMS = {"pi05_psg_a5_b0": (5.0, 0.0), "pi05_psg_a0_b1": (0.0, 1.0)}


def test_the_two_arms_are_the_psg_row_but_for_the_weights():
    psg = layout.load_row("pi05_psg")
    assert adapter.weights(psg) is None  # the method's row states none: its weights are the sampler's own
    for row, (a, b) in ARMS.items():
        definition = layout.load_row(row)
        assert adapter.weights(definition) == (a, b)
        assert definition["method"] == adapter.WEIGHTS_METHOD
        for key in ("environment", "checkpoint"):  # the same model in the same environment
            assert definition[key] == psg[key], (row, key)


def test_only_these_rows_state_weights_and_the_method_does_not_know_them():
    assert sorted(row for row in layout.rows() if "weights" in layout.load_row(row)) == sorted(ARMS)
    for path in sorted((layout.REPO / "psg").glob("*.py")):
        text = Path(path).read_text()
        assert all(row not in text for row in ARMS), path


def test_weights_must_be_numbers():
    assert adapter.weights({"row": "x", "weights": {"a": 5, "b": 1.0}}) == (5.0, 1.0)
    for bad in ({}, {"a": 5.0}, {"a": "5", "b": 1.0}, {"a": True, "b": 1.0}, {"a": None, "b": 0.0}):
        with pytest.raises(SystemExit):
            adapter.weights({"row": "x", "weights": bad})
