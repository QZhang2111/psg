"""The block definitions against their counts and their own invariants. Standard library and pytest; they read
`evaluation/blocks/<block>/block.json` only (no benchmark files, no simulator).
"""

from __future__ import annotations

import json

import pytest

from evaluation import layout

SUITE = {"S": "libero_spatial", "O": "libero_object", "G": "libero_goal", "L": "libero_10"}
PL_PER_SUITE = {"libero_spatial": 385, "libero_object": 403, "libero_goal": 425, "libero_10": 312}
BLOCKS = {
    "L-S", "L-O", "L-G", "L-L", "PP-S", "PP-O", "PP-G", "PT-S", "PT-O", "PT-G", "PL", "OS-O", "OS-G",
    "CF-S", "CF-O", "CF-OOD",
}  # fmt: skip
CF = {"CF-S": (15, 220, "libero_spatial"), "CF-O": (10, 280, "libero_object"), "CF-OOD": (15, 220, "libero_spatial")}


def load(name):
    return json.loads((layout.REPO / "evaluation" / "blocks" / name / "block.json").read_text())


def states(spec):
    out = []
    for part in str(spec).split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def episodes(block):
    return sum(len(states(t.get("start_states", block["start_states"]))) for t in block["tasks"]) * len(block["seeds"])


def test_these_are_the_blocks():
    assert {path.parent.name for path in (layout.REPO / "evaluation" / "blocks").glob("*/block.json")} == BLOCKS


@pytest.mark.parametrize("name", sorted(BLOCKS - set(CF)))
def test_block(name):
    block = load(name)
    assert block["block"] == name
    assert [t["index"] for t in block["tasks"]] == list(range(len(block["tasks"])))
    for t in block["tasks"]:
        assert t.get("instruction") and t["instruction"] == t["instruction"].strip(), t["index"]
        if name.startswith("L-"):
            assert t["reach"] is None  # LIBERO has no Reach
        else:
            assert t["reach"]["rule"] in ("first_close", "first_contact") and t["reach"]["target"], t["index"]
    n = episodes(block)
    if name[:2] in ("L-", "PP", "PT"):
        assert (len(block["tasks"]), n, block["seeds"], block.get("base_suite")) == (10, 500, [7], SUITE[name[-1]])
    if name.startswith("PT"):
        assert sum(t["instruction"] != t["original_instruction"] for t in block["tasks"]) == 10
    if name == "PL":
        per = {}
        for t in block["tasks"]:
            per[t["base_suite"]] = per.get(t["base_suite"], 0) + 1
            assert t["family"] in ("add", "level", "moved_level") and isinstance(t["moves_target"], bool), t["index"]
            assert not (t["family"] == "add" and t["moves_target"]), t["index"]
        assert per == PL_PER_SUITE and n == 1525
    if name == "OS-O":
        assert (len(block["tasks"]), n) == (70, 689)  # 700 scenes less the 11 not run, seed 7
    if name == "OS-G":
        assert (len(block["tasks"]), n) == (21, 210)


@pytest.mark.parametrize("name", sorted(CF))
def test_cf_block(name):
    block = load(name)
    n, max_steps, base = CF[name]
    assert block["block"] == name and block["base_suite"] == base and len(block["tasks"]) == n
    assert block["rollout"] == {"environment_seed": 7, "settle_steps": 10, "max_steps": max_steps, "stop": "predicate"}
    assert block["start_states"] == "0-49" and block["seeds"] == [7] and block["loader"] == "conditions"
    assert episodes(block) == {"CF-S": 750, "CF-O": 500, "CF-OOD": 750}[name]
    for i, t in enumerate(block["tasks"]):
        assert t["index"] == i
        assert t["instruction"] == " ".join(t["task"].split("_")).split("-")[-1]
        p = t["predicate"]
        assert p["success"] in p["stop_any"] and t["training_condition"] in p["stop_any"]
        assert p["success"] != t["training_condition"]
        assert t["reach"]["rule"] == "first_close" and len(t["reach"]["target"]) == 1
        assert t["reach"]["target"][0] == p["success"].split()[1]  # Reach target = the instructed condition's subject
        assert len(t["bddl_sha256"]) == 64 and len(t["init_git_blob"]) == 40
    if name == "CF-S":
        assert all("reach_source" in t for t in block["tasks"])
