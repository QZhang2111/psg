# How a = 5 and b = 1 were chosen: the method's two weights on eleven development tasks

**Recorded from an earlier repository of this project; not re-run here.** These episodes were
run by that repository's own runner, not by `run.py`.

The paper says the method's two weights were selected on 11 development tasks under the Original, Position
and Task conditions, on start states 6-15, and then kept for every evaluation. This record holds the counts
behind that choice.

## What was measured

- **Model.** π0.5, the released LIBERO checkpoint, and the method on it:
  ṽ = v + a·v_g − b·v_d on the six motion dimensions, the name being every noun phrase of the instruction.
- **Tasks.** Eleven tasks of LIBERO: three of Object, three of Goal, three of Spatial, two of Long
  ([`development_tasks.csv`](development_tasks.csv)), each under three conditions: Original, LIBERO-PRO
  Position, LIBERO-PRO Task. Start states 6-15; noise seeds 7 and 17. 110 episodes per condition and seed.
- **Success** is the benchmark's task completion. **Reach**: the gripper's first close is on the requested
  object; it is counted on ten of the tasks (100 episodes per condition and seed; the drawer task is left
  out).
- **Settings compared.** The released policy; a = 2, b = 1; a = 5, b = 1 (the method), all with both terms
  applied for the whole episode. For a = 5, b = 1 two more ways of applying the terms were run at one seed:
  until the gripper first closes, and while the gripper command is open.
- **Gained and lost.** Every setting is paired with the released policy on the same task, start state and
  noise; the tables count the episodes a setting gains and loses against it.

## The tables

| File | What it holds |
|---|---|
| [`selection.csv`](selection.csv) | per suite and for all 11 tasks, per condition, seed or both seeds, and setting: episodes, successes, Reach, and episodes gained and lost against the released policy |
| [`original_all_40_tasks.csv`](original_all_40_tasks.csv) | the Original condition on all 40 LIBERO tasks at seed 7, per suite and setting: successes, gained, lost |
| [`development_tasks.csv`](development_tasks.csv) | the eleven tasks |

**The two weights**, all 11 tasks, both seeds, both terms for the whole episode:

| Condition | Measure | Released policy | a = 2, b = 1 | a = 5, b = 1 (the method) |
|---|---|---|---|---|
| Original | Success | 212 of 220 | 211 of 220 | 213 of 220 |
| Position | Reach | 95 of 200 | 119 of 200 | 139 of 200 |
| Position | Success | 27 of 220 | 76 of 220 | 82 of 220 |
| Task | Reach | 100 of 200 | 118 of 200 | 140 of 200 |
| Task | Success | 46 of 220 | 59 of 220 | 63 of 220 |

**When the terms are applied**, a = 5 and b = 1, all 11 tasks, seed 7:

| Condition | Measure | Whole episode (the method) | Until the first close | While the gripper is open |
|---|---|---|---|---|
| Original | Success | 108 of 110 | 109 of 110 | 105 of 110 |
| Position | Reach | 69 of 100 | 70 of 100 | 69 of 100 |
| Position | Success | 42 of 110 | 32 of 110 | 39 of 110 |
| Task | Reach | 71 of 100 | 69 of 100 | 69 of 100 |
| Task | Success | 36 of 110 | 33 of 110 | 33 of 110 |

**What the method costs on the trained tasks**, Original, all 40 tasks of LIBERO, seed 7, start states 6-15:

| Setting | Original success, all 40 tasks, seed 7 |
|---|---|
| released policy | 390 of 400 |
| a = 2, b = 1 | 385 of 400 |
| a = 5, b = 1 (the method) | 387 of 400 |
| a = 5, b = 1, until the first close | 390 of 400 |
| a = 5, b = 1, while the gripper is open | 385 of 400 |

## How it is read

- a = 5 has more Reach than a = 2 under Position (139 against 119 of 200) and under Task (140 against 118),
  more success under both (82 against 76, 63 against 59 of 220), and the same success on the trained tasks
  (213 against 211 of 220).
- Applying the terms for the whole episode has more Position success than stopping at the first close (42
  against 32 of 110, one seed); Reach is within one episode (69 and 70 of 100).
- On all 40 trained tasks the method has 387 of 400 against the released policy's 390 at this seed.

## What this record is not

- **Not a held-out selection.** The eleven tasks and start states 6-15 are also evaluated in this
  repository's tables, which run every task of a suite on start states 0-49 (the two Long tasks under
  Original only).
- Small counts: 10 episodes per task, condition and seed. The two seeds of one setting differ by up to 9
  episodes in a cell of 110 (the method's Task success: 36 at seed 7, 27 at seed 17; `selection.csv`).
- Other values (a = 8; b = 0.5 and b = 0) and another rule for the name were tried in the earlier repository
  on the same tasks; they are not recorded here.
- The settings were run by another program than this repository's rows. The method's row here
  (`pi05_psg`) is the same equation with a = 5 and b = 1; its numbers on the tables' episodes are in
  `results/`.
