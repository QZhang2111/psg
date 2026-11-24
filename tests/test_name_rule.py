"""The name rule: which words of an instruction are "the name". Standard library only."""

from __future__ import annotations

import pytest

from psg import name_rule

OBJECT_FOODS = [
    "alphabet soup", "cream cheese", "salad dressing", "bbq sauce", "ketchup", "tomato sauce", "butter", "milk",
    "chocolate pudding", "orange juice",
]  # fmt: skip


def test_name_rule_on_the_object_suite():
    for food in OBJECT_FOODS:
        assert name_rule.phrases(f"pick up the {food} and place it in the basket") == [food, "basket"]


@pytest.mark.parametrize(
    ("instruction", "expected"),
    [
        ("put both moka pots on the stove", ["moka pots", "stove"]),
        ("put the yellow and white mug in the microwave and close it", ["yellow and white mug", "microwave"]),
        ("pick up the black bowl from table center and place it on the plate", ["black bowl", "table center", "plate"]),
        ("put the bowl on top of the cabinet", ["bowl", "cabinet"]),
        ("open the middle drawer of the cabinet", ["middle drawer", "cabinet"]),
        ("push the plate to the front of the stove", ["plate", "stove"]),
        ("turn on the stove", ["stove"]),
    ],
)
def test_name_rule_phrases(instruction, expected):
    assert name_rule.phrases(instruction) == expected


def test_name_rule_refuses_an_instruction_without_a_noun_phrase():
    with pytest.raises(ValueError):
        name_rule.phrases("go")


def test_token_indices():
    pieces = ["", "pick", " up", " the", " alphabet", " soup", " and", " place", " it", " in", " the", " basket", "\n"]
    assert name_rule.token_indices(pieces, ["alphabet soup", "basket"]) == [4, 5, 11]
    pieces = ["", "put", " the", " bowl", " on", " the", " bowl", " rack"]
    assert name_rule.token_indices(pieces, ["bowl", "bowl rack"]) == [3, 6, 7]
    with pytest.raises(ValueError):
        name_rule.token_indices(pieces, ["plate"])
