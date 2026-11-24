# Evaluation blocks: the test sets, one copy for all rows

One directory per block, `evaluation/blocks/<block>/block.json`. A block is a benchmark × suite × perturbation with all its
tasks and start states. It says what is run and how an episode is scored; it says nothing about any model.
`python -m evaluation.run --block <block> --row <row>` runs every task × start state × seed of it.

Blocks: `L-S` `L-O` `L-G` `L-L` (LIBERO) · `PP-O` `PP-G` `PP-S` (LIBERO-PRO Position) · `PT-O` `PT-G` `PT-S`
(LIBERO-PRO Task) · `PL` (LIBERO-Plus Layout) · `OS-O` `OS-G` (our own scenes) · `CF-S` `CF-O` `CF-OOD`
(LIBERO-CF).

## `block.json`

```json
{
  "block": "PP-O",
  "title": "LIBERO-PRO Position, Object",
  "benchmark": {
    "name": "LIBERO-PRO",
    "code": {"address": "github.com/Zxy-MLlab/LIBERO-PRO", "revision": "<full commit>", "licence": "MIT"},
    "files": {"address": "huggingface.co/datasets/zhouxueyang/LIBERO-Pro", "revision": "<full commit>", "licence": "CC-BY-4.0"},
    "suite": "libero_object_swap",
    "perturbation": "position"
  },
  "environment": "sim-libero-pro",
  "base_suite": "libero_object",
  "loader": "libero",
  "rollout": {"environment_seed": 7, "settle_steps": 10, "max_steps": 280, "stop": "predicate"},
  "seeds": [7],
  "start_states": "0-49",
  "tasks": [
    {
      "index": 0,
      "task": "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
      "bddl": "libero_object_swap/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl",
      "init": "libero_object_swap/pick_up_the_alphabet_soup_and_place_it_in_the_basket.pruned_init",
      "instruction": "pick up the alphabet soup and place it in the basket",
      "reach": {"rule": "first_close", "target": ["alphabet_soup_1"]}
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `benchmark.code`, `.files` | Address and full revision of the benchmark's code and of its task files, when the files are a separate download. Read from the page, not from memory |
| `benchmark.suite` | The suite's name as the benchmark's code registers it |
| `base_suite` | The LIBERO suite the block's scenes come from: `libero_spatial`, `libero_object`, `libero_goal` or `libero_10`. Every model is told it per episode (a release with per-suite statistics chooses by it). A task may override it (`tasks[].base_suite`: LIBERO-Plus Layout mixes suites) |
| `environment` | Name of the simulator environment that holds this benchmark's code (`evaluation/environments/<name>.json`). The three forks are all a Python package called `libero`, so each has its own environment |
| `loader` | How the evaluation builds an episode: `libero` (a BDDL file and a start state from an init file), `layout` (LIBERO-Plus: one start state per variant), `relocate` (our scenes), `conditions` (LIBERO-CF's termination). The loaders are the evaluation's code; a block only names one |
| `rollout` | `environment_seed` 7; `settle_steps` 10; `max_steps` after settling: 220 Spatial, 280 Object, 300 Goal, 520 Long, LIBERO-CF 220 / 280 / 220, LIBERO-Plus and our scenes as their suite; `stop`: `predicate` (the episode ends when the success predicate holds). A task may override `max_steps` (LIBERO-Plus mixes suites) |
| `seeds` | Noise seeds; `[7]` in every block |
| `start_states` | Indices into the task's init file: `"0-49"`, `"6-15"`; a task may override (`"0"` for a Layout variant) |
| `tasks[].index` | Position in this list, from 0; it is the `t<index>` of the episode's name and never changes once a unit has run |
| `tasks[].task` | The task's name in its benchmark |
| `tasks[].bddl`, `.init` | Paths relative to the benchmark's BDDL and init-state roots. Add `bddl_sha256` and `init_sha256` when you can compute them from the files; without them the first run records what it loaded and the digests are pinned from that record |
| `tasks[].instruction` | **The text the model is given, written out.** Under LIBERO-PRO Task it is the BDDL's changed `:language`, not the file name; under LIBERO-CF the counterfactual instruction; under LIBERO-Plus the base instruction without the file name's suffixes |
| `tasks[].reach` | `{"rule", "target"}`, or `null` where the block has no Reach (LIBERO). See below |
| `tasks[].predicate` | Only where success is not the BDDL goal of the file: LIBERO-CF, `{"stop_any": ["<condition>", ...], "success": "<condition>"}` in the benchmark's own condition syntax: the episode ends when any listed condition holds and succeeds when the instructed one does |
| `tasks[].relocate` | Only our scenes: `{"body": "cream_cheese_1", "place": "x6y8", "xy": [x, y]}`: the base task's start state with this one body set to the new place, everything else as it was. If the grid's scene builder did more than this, say what, and the loader follows it |
| `tasks[].family`, `.moves_target` | Only LIBERO-Plus Layout: `add` / `level` / `moved_level`; whether the variant moves the target (and `tasks[].base_suite`, the suite the variant comes from) |

Extra per-task columns are allowed (they are copied into the config record); the names above are fixed.

## The Reach rule and target of a task

```json
"reach": {"rule": "first_close", "target": ["alphabet_soup_1"]}
"reach": {"rule": "first_contact", "target": ["wooden_cabinet_1:middle"]}
```

- `rule`: `first_close` when the task begins with a grasp: the first time the gripper closes, it closes on the
  target. `first_contact` when the task needs no grasp (open a drawer, push a plate, turn on the stove): the first
  thing the gripper touches is the target. The rule belongs to the task, never to what a policy did.
- `target`: the entity the hand has to go for **first**, by the scene's own name for it (the object or fixture
  names of the BDDL, `alphabet_soup_1`, `flat_stove_1`). A part of a fixture is `"<entity>:<part>"`, where
  `<part>` is a piece of the part's body name (`wooden_cabinet_1:middle` for the middle drawer). Several
  entries mean any of them.
- For a two-stage task the target is the first stage's.
- Where the BDDL does not name the target (LIBERO-PRO Goal after a swap, CF-Spatial, a Layout variant), it is
  mapped by hand from the instruction; add `"reach_source": "<how>"` to the task.
- What cannot be decided from the files (which body of a fixture is the handle; a fixture without collision
  geometry) is checked in the simulator before the block is run, not guessed.

## Where a definition comes from

Every `block.json` was generated from its benchmark's own files by a development script that is not in this
repository; the file is the definition, and its tasks are what every config record copies. Its `built_by` or
`generated_by` says so. `tests/test_blocks.py` checks each definition's counts and invariants.
