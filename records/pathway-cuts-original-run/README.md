# The pathway cuts' original run: the counts the paper prints

Recorded, not re-run here. This folder holds the earlier run of the pathway cuts, 200 episodes per arm, whose
counts the paper prints in its pathway-cut figure and in Sec. 3.3 (29, 0 and 64 of 200 for π0.5 under
Position). No code in this repository produced these files, and nothing under `results/` holds them.

The same cuts were made again in this repository on 2026-11-16, on 250 episodes per arm: the eight units
`results/PP-O/pi05_cut_*/`, `results/PP-O/pi0_cut_*/`, `results/L-O/pi05_cut_none/` and
`results/L-O/pi05_cut_direct/` (`instruments/README.md`). Those are other episodes, counted by this
repository's labeller, and give other numbers; the figure is drawn from the files here.

| File | What it holds |
|---|---|
| `episodes.tsv` | One line per episode, 5,600 in all: two models (`pi0`, `pi05`) × two panels (`P`, `X`) × seven arms × five tasks of LIBERO Object (0, 3, 6, 8, 9) × start states 0-19 × noise seeds 7 and 17. Columns: `model`, `seed`, `arm`, `panel`, `task`, `state`, `first_close` (the category of the first gripper close), `nearest` (the body nearest the hand at that close), `first_descent`, `joint`, `success` |
| `counts.json` | `first_close`: per model, panel and arm, the number of episodes in each first-close category (`n` is 200 in each of the 28). Also a list of predictions with whether each was met, the names of the readout files the counts were made from, and transition counts |
| `look-replan0.json` | Attention at the first query, in tables for the main and the wrist camera: per model, panel and arm, the median and quartiles over the episodes of the attention on each of seven regions, and ratios of an arm to the arm with nothing cut. It is on five arms: `N0`, `IS`, `AS`, `BS`, and `BL`, which is not among the seven arms of the episode table; the note that came with these files calls it "no instruction". `IN`, `AN` and `BN` are not in this file |
| `position_layouts.json` | **An extract.** Three fields (`training_slot`, `scene_slot`, `layout`) of each of the ten Position units, copied from the run's panel definition, a JSON file of 30 units. The figure reads nothing else of it |

What the figure was drawn from: panel `P` (Position) and three of the seven arms,
`N0` (nothing cut), `IS` (the grounding cut) and `AS` (the direct cut), 200 episodes per bar. The categories
are named as in the data: `green` is a first close on the requested object, `train=scene` one at its training
place. Counted again from `episodes.tsv`, every bar equals `counts.json`; with the closes at the other training place
split off, π0.5 has 29, 0 and 64 on the requested object. Panel (c) is the table "main camera, replan 0" of `look-replan0.json`, region `requested`.

The two JSON files hold more than the figure reads: `counts.json` also the predictions, the readout names
and the transitions; `look-replan0.json` also its own checks and predictions, a list of replans, tables of direction, and tables
per task.

## The panels, arms and categories of the run

As the run's design states them. The figure draws panel `P` and the arms `N0`, `IS` and `AS`; the other panel
and the other four arms are in the files and are not drawn.

- **Panels.** `P`, Position: the requested food has exchanged places with another object, and the request is
  the same. `X`, Task-paired: the scene is unchanged and the instruction names another food.
- **Arms.** Every cut denies attention before the softmax, in every layer and head, at every query.

  | Arm | The image tokens of both cameras may not read | The action side may not read |
  |---|---|---|
  | `N0` | nothing | nothing |
  | `IS` | the whole instruction | nothing |
  | `AS` | nothing | the whole instruction |
  | `BS` | the whole instruction | the whole instruction |
  | `IN` | the name's tokens | nothing |
  | `AN` | nothing | the name's tokens |
  | `BN` | the name's tokens | the name's tokens |

  The action side is the action tokens and, in π0, the state token. The whole instruction is every prompt
  token from the first to the closing newline. The name is the requested food's name in the instruction
  given.
- **Categories of a first close.** `green`: where the requested food stands now. `blue`: the requested food's
  training place. `orange`: the scene task's place. `train=scene`: a place that is both, which is the
  training place under Position.
