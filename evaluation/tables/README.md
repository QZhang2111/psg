# The paper's tables from `results/`

```
python -m evaluation.tables.render
```

prints Tables 1-3 as the paper prints them (rows in the paper's order, columns, numbers, and the marks † and
`*`; compared with `printed/`), from the episode tables under `results/` and the seven cited numbers of `cited_libero.csv` (any
Python >= 3.8, standard library, from the checkout; no card, no model, no simulator). `--table 2` prints one,
`--out <directory>` also writes `table1.csv`, `table2.csv`, `table3.csv`, and `--check` compares every number
with the tables as printed (`printed/`) and names each one that differs: 130 + 60 + 32 numbers, all equal.
The paper's bold is not printed: there it marks the best of a column among the rows that need no training,
and the rows that require training (†) are gray and not ranked.

Nothing a table is made of is in code. It is in four small files here:

| File | What it holds |
|---|---|
| [`tables.json`](tables.json) | per table: which row of this repository is which line, which block is which column, the rule of every average, Table 2's three groups |
| [`os_g_scenes.csv`](os_g_scenes.csv) | Table 2's 21 scenes: object, place, distance from the training place, group |
| [`cited_libero.csv`](cited_libero.csv) | the seven LIBERO averages the paper cites (marked `*`), each with the four suites' numbers, its public source, and whether it is the checkpoint run here (blank where the source does not say) |
| [`printed/`](printed) | the three tables as the paper prints them, for `--check` |

## The rules

A number is a percentage of a unit's episodes: 100 × (episodes with `success` = 1, or with `reach` = 1) /
(episodes of the unit), from `results/<block>/<row>/episodes.csv`. It is computed as an exact fraction and
rounded to one decimal, a half rounded up (83.75 prints as 83.8). A unit is read only when it is whole: every
task × start state × seed its config record names is there once and none failed; otherwise the command stops
and names the unit.

| Table | Column | Rule |
|---|---|---|
| 1 | LIBERO, Avg | the mean of the four suites' success percentages (`L-S`, `L-O`, `L-G`, `L-L`); for the seven rows marked `cited` in `tables.json`, the number their authors report (`cited_libero.csv`), printed with `*`; SmolVLA is one of them, although its LIBERO blocks were run here too (`results/L-*/smolvla_exec10/`, 83.8) |
| 1 | LIBERO-PRO Position: Object, Goal, Spatial | the unit's percentage (`PP-O`, `PP-G`, `PP-S`) |
| 1 | LIBERO-PRO Position, Avg | the mean of those three percentages |
| 1 | LIBERO-PRO Task, Avg | the mean of the three suites' percentages (`PT-O`, `PT-G`, `PT-S`); the suites' own numbers are not printed |
| 1 | LIBERO-Plus, Layout | the unit's percentage (`PL`, 1,525 episodes) |
| 2 | < 20, 20-30, > 30 cm | all episodes of the group's scenes of `OS-G` counted together (4, 4 and 10 scenes × 10 start states) |
| 3 | Spatial, Object, OOD | the unit's percentage (`CF-S`, `CF-O`, `CF-OOD`) |
| 3 | Avg | the mean of the three subsets' percentages, each subset weighing the same; not the count over all 2,000 episodes (the subsets have 750, 500 and 750) |

Where the blocks of an average have the same number of episodes (Table 1), the mean of their percentages is
also the percentage over all their episodes; in Table 3 it is not.

## Table 2's groups

A scene of `OS-G` is one object (the bowl, the cream cheese, the wine bottle) at one place: its training place
or one of six new ones. **A scene's distance** is how far the object stands, at the first query of an episode,
from where it stands in the training-place scene of the same task and start state, measured horizontally and
averaged over the scene's ten start states. It is read from the recorded episodes of a unit of `OS-G`:

```
python -m evaluation.tables.scenes distances <output root>/OS-G/pi05/raw/traces --out evaluation/tables/os_g_scenes.csv
```

(NumPy; the traces are not in git.) The start state does not depend on the policy: the traces of eight rows
gave the same file, byte for byte. A scene belongs to the group its mean distance falls in, with the cuts of
`tables.json` at 20 and 30 cm; the three training-place scenes are in no column of the table. One scene has
start states on both sides of a cut: the cream cheese at `x6y8`, 16.7 to 20.8 cm over its ten start states,
mean 18.8, in the group below 20.

## Where the gripper first closed (the Object scenes)

```
python -m evaluation.tables.first_close results/PP-O/pi05/episodes.csv --tasks 0,3,6,8,9
```

counts, per task of an episode table of `PP-O` or `L-O`, where the gripper first closed: on the requested
food; on the body that stands at the food's training place; on anything else; or never. `success` stands
beside the four. Standard library; several episode tables may be given, and `--out` writes the counts as CSV.
It takes the episode table of any row: a policy, a method, an arm of an instrument.

Which body stands where is [`object_places.csv`](object_places.csv). LIBERO-PRO Position exchanges two
objects of a scene, one of them the requested food (`evaluation/blocks/PP-O/block.json`, each task's `swapped`); the
table names the other of the pair as the body at the food's training place, and

```
python -m evaluation.tables.scenes places <output root>/L-O/pi05/raw/traces <output root>/PP-O/pi05/raw/traces \
    --out evaluation/tables/object_places.csv
```

checks it against the recorded start states (that body stands within 2 cm of the food's training place in
every one of the 50; the largest offset found is 1.2 cm) and refuses to write when they disagree.

**The strict column.** LIBERO-Object trains the requested food of every task on one of two places of the
table: (−0.12, −0.24) m for tasks 0, 4, 6, 7, 8 and (0.05, −0.10) m for tasks 1, 2, 3, 5, 9. In five tasks of
`PP-O` (1, 3, 5, 6, 7) the exchange puts the requested food on the one of the two its own task does not use
(`requested_on_other_training_place` = 1), so a hand that goes to that place closes on the food without
having looked for it. `strict` counts a close on the food only in the other tasks; the line of such a task
says `-` and adds nothing to the sum.

What the command gives on this repository's episode tables, for the five tasks of the paper's behaviour
figure (0 the alphabet soup, 3 the BBQ sauce, 6 the butter, 8 the chocolate pudding, 9 the orange juice; 50
start states each, one noise seed):

| Block | Row | Episodes | On the food | On the body at its place | Anything else | No close | Success | Strict |
|---|---|---|---|---|---|---|---|---|
| `PP-O` | `openvla_oft` | 250 | 0 | 250 | 0 | 0 | 0 | 0 |
| `PP-O` | `smolvla_exec10` | 250 | 0 | 249 | 1 | 0 | 0 | 0 |
| `PP-O` | `turbovla` | 250 | 0 | 242 | 8 | 0 | 0 | 0 |
| `PP-O` | `pi0_fast` | 250 | 0 | 248 | 2 | 0 | 0 | 0 |
| `PP-O` | `pi0` | 250 | 0 | 236 | 14 | 0 | 1 | 0 |
| `PP-O` | `pi05` | 250 | 60 | 154 | 36 | 0 | 36 | 35 |
| `L-O` | `pi05` | 250 | 250 | (the food itself) | 0 | 0 | 246 | 250 |

π0.5's 60 closes on the moved food are 21, 0, 25, 0 and 14 in the five tasks; the 25 are the butter's, which
stands on the other training place, so the strict count is 21 + 0 + 14 = 35.

## Tests

`tests/test_tables.py` (standard library and NumPy; no card): the rounding; the mean of percentages against
the pooled count; the scene groups; the cited mark; a unit that is not whole is refused; this repository's
`results/` give the printed tables; the four places of a first close and the strict count; the two scene
tables from hand-made traces, and the refusal when traces contradict the block.
