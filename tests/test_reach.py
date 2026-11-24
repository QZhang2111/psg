"""The Reach labeller on hand-made traces. Standard library and NumPy; no simulator."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation import reach
from evaluation import trace as _trace

import synthetic as s


def make(steps, settle=()):
    """A trace from `steps`: (hand position, gripper command, contacts) per policy step; `settle` likewise."""
    writer = _trace.TraceWriter({"episode": "test"}, s.TABLES)
    writer.frame(_trace.START, None, s.frame([0, 0, 0.3]), False)
    for hand, gripper, contacts in settle:
        writer.frame(_trace.SETTLE, s.action(gripper), s.frame(hand, contacts), False)
    for hand, gripper, contacts in steps:
        writer.frame(_trace.POLICY, s.action(gripper), s.frame(hand, contacts), False)
    return writer.arrays()


NEAR_SOUP = [0.49, 0.0, 0.02]
NEAR_TOMATO = [0.5, 0.29, 0.02]
ABOVE = [0.0, 0.0, 0.3]


def test_first_close_is_on_the_nearest_entity():
    trace = make([(ABOVE, -1, []), (NEAR_SOUP, -1, []), (NEAR_SOUP, 1, []), (NEAR_TOMATO, 1, [])])
    row = reach.label(trace, {"rule": "first_close", "target": ["alphabet_soup_1"]})
    assert (row["reach"], row["first_close_step"], row["first_close_entity"]) == (1, 2, "alphabet_soup_1")
    assert row["first_close_body"] == "alphabet_soup_1_main" and row["first_close_distance_m"] == pytest.approx(0.0224, abs=1e-3)
    assert row["reach_rule"] == "first_close" and row["reach_target"] == "alphabet_soup_1" and row["labeller"] == reach.REVISION
    assert reach.label(trace, {"rule": "first_close", "target": ["tomato_sauce_1"]})["reach"] == 0
    assert reach.label(trace, {"rule": "first_close", "target": ["tomato_sauce_1", "alphabet_soup_1"]})["reach"] == 1


def test_only_the_first_close_counts():
    trace = make([(NEAR_TOMATO, 1, []), (ABOVE, -1, []), (NEAR_SOUP, 1, [])])
    row = reach.label(trace, {"rule": "first_close", "target": ["alphabet_soup_1"]})
    assert (row["reach"], row["first_close_step"], row["first_close_entity"]) == (0, 0, "tomato_sauce_1")


def test_no_close_is_no_reach():
    row = reach.label(make([(NEAR_SOUP, -1, []), (NEAR_SOUP, -1, [])]), {"rule": "first_close", "target": ["alphabet_soup_1"]})
    assert row["reach"] == 0 and row["first_close_step"] == "" and row["first_close_entity"] == ""


def test_settling_steps_are_not_policy_steps():
    trace = make([(NEAR_SOUP, 1, [])], settle=[(NEAR_TOMATO, 1, [(1, s.TOMATO, -0.001)])])
    row = reach.label(trace, {"rule": "first_close", "target": ["alphabet_soup_1"]})
    assert (row["reach"], row["first_close_step"], row["first_contact_step"]) == (1, 0, "")


def test_a_part_of_a_fixture():
    at_drawer = [0.0, 0.6, 0.31]
    trace = make([(at_drawer, 1, [])])
    assert reach.label(trace, {"rule": "first_close", "target": ["wooden_cabinet_1:middle"]})["reach"] == 1
    assert reach.label(trace, {"rule": "first_close", "target": ["wooden_cabinet_1:top"]})["reach"] == 0
    row = reach.label(trace, {"rule": "first_close", "target": ["wooden_cabinet_1"]})
    assert row["reach"] == 1 and row["first_close_body"] == "wooden_cabinet_1_cabinet_middle"


def test_a_geom_is_placed_by_its_bodys_pose():
    # The cabinet base's geom sits 10 cm along the body's x axis; turned a quarter about z it lies along y.
    quat = s.IDENTITY.copy()
    quat[s.CABINET] = [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]
    writer = _trace.TraceWriter({"episode": "test"}, s.TABLES)
    writer.frame(_trace.START, None, s.frame(ABOVE), False)
    writer.frame(_trace.POLICY, s.action(1), s.frame([0.0, 0.7, 0.0], body_quat=quat), False)
    model = reach.Model(writer.arrays())
    centres = model.geom_centres(s.PLACES.astype(float), quat.astype(float))
    np.testing.assert_allclose(centres[s.CABINET], [0.0, 0.7, 0.0], atol=1e-6)
    row = reach.label(writer.arrays(), None)
    assert row["first_close_body"] == "wooden_cabinet_1_base" and row["first_close_distance_m"] == pytest.approx(0.0, abs=1e-4)


def test_first_contact_ignores_the_table_and_takes_the_first_entity():
    trace = make([
        (ABOVE, -1, [(1, s.TABLE, -0.002)]),
        (NEAR_TOMATO, -1, [(1, s.TABLE, -0.002), (1, s.TOMATO, -0.001)]),
        (NEAR_SOUP, -1, [(1, s.SOUP, -0.001)]),
    ])  # fmt: skip
    row = reach.label(trace, {"rule": "first_contact", "target": ["tomato_sauce_1"]})
    assert (row["reach"], row["first_contact_step"], row["first_contact_entity"]) == (1, 1, "tomato_sauce_1")
    assert reach.label(trace, {"rule": "first_contact", "target": ["alphabet_soup_1"]})["reach"] == 0


def test_a_contact_must_touch_and_must_be_the_grippers():
    trace = make([(NEAR_SOUP, -1, [(1, s.SOUP, 0.004), (0, s.TOMATO, -0.001)]), (NEAR_SOUP, -1, [(1, s.SOUP, 0.0)])])
    assert reach.label(trace, None)["first_contact_step"] == 1


def test_a_body_belongs_to_the_entity_it_is_named_after():
    model = reach.Model(make([(ABOVE, -1, [])]))
    assert model.body_entity[s.PLATE_10] is None  # "plate_10_main" is not a body of "plate_1"
    assert model.body_entity[s.DRAWER] == "wooden_cabinet_1" and model.body_entity[0] is None


def test_without_a_reach_definition_the_events_are_still_written():
    row = reach.label(make([(NEAR_SOUP, 1, [(1, s.SOUP, -0.001)])]), None)
    assert (row["reach"], row["reach_rule"], row["reach_target"]) == ("", "", "")
    assert row["first_close_entity"] == "alphabet_soup_1" and row["first_contact_entity"] == "alphabet_soup_1"


def test_an_unknown_rule_is_refused():
    with pytest.raises(ValueError):
        reach.label(make([(NEAR_SOUP, 1, [])]), {"rule": "nearest", "target": ["alphabet_soup_1"]})


def test_a_trace_survives_the_file(tmp_path):
    writer = _trace.TraceWriter({"episode": "PP-O.fake.t0.s0.n7"}, s.TABLES)
    writer.frame(_trace.START, None, s.frame(ABOVE), False)
    writer.query(5, np.zeros((10, 7)), {"v": np.ones((10, 10, 7))})
    writer.frame(_trace.POLICY, s.action(1), s.frame(NEAR_SOUP, [(1, s.SOUP, -0.001)]), True)
    digest = writer.save(tmp_path / "raw" / "traces" / "e.npz")
    back = _trace.load(tmp_path / "raw" / "traces" / "e.npz")
    assert digest == _trace.file_sha256(tmp_path / "raw" / "traces" / "e.npz") and len(digest) == 64
    assert _trace.meta(back)["episode"] == "PP-O.fake.t0.s0.n7"
    assert back["phase"].tolist() == [0, 2] and back["predicate"].tolist() == [False, True]
    assert back["query_frame"].tolist() == [0] and back["query/v"].shape == (1, 10, 10, 7)
    assert back["contact_other_geom"].tolist() == [s.SOUP] and np.isnan(back["action"][0]).all()
    assert reach.label(back, {"rule": "first_close", "target": ["alphabet_soup_1"]})["reach"] == 1


# ---- first contact on a fixture: the entity, the raw body, and the first part of the named kind ----

AT_CABINET = [0.0, 0.6, 0.1]
DRAWER_TARGET = {"rule": "first_contact", "target": ["wooden_cabinet_1:middle"]}


def test_the_frame_first_and_then_the_named_drawer_is_a_hit():
    trace = make([
        (AT_CABINET, -1, [(1, s.CABINET, -0.001)]),
        (AT_CABINET, -1, [(1, s.CABINET, -0.001), (1, s.DRAWER, -0.001)]),
        (AT_CABINET, -1, [(1, s.TOP_DRAWER, -0.001)]),
    ])  # fmt: skip
    row = reach.label(trace, DRAWER_TARGET)
    assert (row["first_contact_step"], row["first_contact_entity"], row["first_contact_body"]) == (0, "wooden_cabinet_1", "wooden_cabinet_1_base")
    assert (row["first_part_step"], row["first_part_body"]) == (1, "wooden_cabinet_1_cabinet_middle")
    assert row["reach"] == 1


def test_the_right_cabinet_and_the_wrong_drawer_is_a_miss_that_still_says_cabinet():
    trace = make([
        (AT_CABINET, -1, [(1, s.CABINET, -0.001)]),
        (AT_CABINET, -1, [(1, s.TOP_DRAWER, -0.001)]),
        (AT_CABINET, -1, [(1, s.DRAWER, -0.001)]),
    ])  # fmt: skip
    row = reach.label(trace, DRAWER_TARGET)
    assert (row["reach"], row["first_contact_entity"], row["first_part_body"]) == (0, "wooden_cabinet_1", "wooden_cabinet_1_cabinet_top")


def test_another_entity_first_is_a_miss_whatever_drawer_follows():
    trace = make([(NEAR_SOUP, -1, [(1, s.SOUP, -0.001)]), (AT_CABINET, -1, [(1, s.DRAWER, -0.001)])])
    row = reach.label(trace, DRAWER_TARGET)
    assert (row["reach"], row["first_contact_entity"], row["first_part_body"]) == (0, "alphabet_soup_1", "wooden_cabinet_1_cabinet_middle")


def test_the_named_drawer_touched_directly_is_a_hit_and_the_frame_alone_is_not():
    assert reach.label(make([(AT_CABINET, -1, [(1, s.DRAWER, -0.001)])]), DRAWER_TARGET)["reach"] == 1
    row = reach.label(make([(AT_CABINET, -1, [(1, s.CABINET, -0.001)])]), DRAWER_TARGET)
    assert (row["reach"], row["first_contact_entity"], row["first_part_step"], row["first_part_body"]) == (0, "wooden_cabinet_1", "", "")


def test_a_target_without_a_part_is_read_at_the_entity_level():
    trace = make([(AT_CABINET, -1, [(1, s.TOP_DRAWER, -0.001)])])
    row = reach.label(trace, {"rule": "first_contact", "target": ["wooden_cabinet_1"]})
    assert (row["reach"], row["first_part_body"]) == (1, "")


def test_the_kind_of_a_part_is_read_from_the_bodies_names():
    model = reach.Model(make([(ABOVE, -1, [])]))
    assert model.parts_like("wooden_cabinet_1:middle") == {"wooden_cabinet_1_cabinet_middle", "wooden_cabinet_1_cabinet_top"}
    assert model.parts_like("wooden_cabinet_1:base") == {"wooden_cabinet_1_base"}  # no kind in the name: the part alone
    assert model.parts_like("wooden_cabinet_1") == set() and model.parts_like("alphabet_soup_1:lid") == set()
