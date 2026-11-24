# Where six released policies first close when another food of the scene is requested

**Recorded from an earlier repository of this project; not re-run here.** This repository has
no Task-paired block.

This is the measurement behind an earlier draft's figure of six policies' first closes: the Task-paired half,
which nothing here makes again, and the Position half as the earlier repository ran it. The Position half has since been counted again from this
repository's own episode tables, on other episodes (see the end of this page).

## What was measured

- **Scenes.** LIBERO-PRO on five tasks of LIBERO-Object (indices 0, 3, 6, 8, 9), start states 0-19, noise
  seeds 7 and 17. **Position:** the requested food exchanged with another food, the instruction unchanged.
  **Task-paired:** the scene unchanged, another food of the scene requested with the instruction's trained
  wording. The five tasks are the ones in which, under Task-paired, the requested food's training place is
  not the place of the scene's own task and the requested food does not stand at its own training place, so
  that the three places a hand can go to are three different places.
- **Policies.** Six released checkpoints: [`models.csv`](models.csv). OpenVLA-OFT and TurboVLA draw no noise
  and repeat an episode exactly, so they have one seed: 100 episodes per kind of scene; the other four have
  200.
- **First close.** The body nearest the gripper, in the horizontal plane, at the step the gripper first
  closes. Every episode falls in one place: on the requested food where it stands now; on the body at the
  requested food's training place; on the body at the place of the scene's own task; on another object; or
  no close. Under Position the training place and the place of the scene's own task are one place.
- **Rollout.** The start state from the benchmark's file, 10 settling steps, then at most 220 policy steps;
  environment seed 0.

## The tables

| File | What it holds |
|---|---|
| [`first_close.csv`](first_close.csv) | per policy and kind of scene: episodes, the count in each place, successes |
| [`episodes.csv`](episodes.csv) | the 2,000 episodes: policy, kind of scene, task, start state, seed, what was requested, the first close's step, body, distance and place, success |
| [`scenes.csv`](scenes.csv) | the ten scenes: the instruction, the requested food, the food of the scene's own task, which body stands at each of the two places |
| [`models.csv`](models.csv) | the six checkpoints, with revision or digest, camera size and seeds |

**Task-paired** (the scene unchanged, another food requested):

| Policy | Episodes | Requested food | Its training place | Scene task's place | Another object | No close | Success |
|---|---|---|---|---|---|---|---|
| OpenVLA-OFT | 100 | 0 | 0 | 100 | 0 | 0 | 0 |
| SmolVLA | 200 | 0 | 30 | 167 | 3 | 0 | 0 |
| TurboVLA | 100 | 0 | 0 | 100 | 0 | 0 | 0 |
| pi0-FAST | 200 | 0 | 191 | 3 | 2 | 4 | 0 |
| pi0 | 200 | 0 | 197 | 0 | 3 | 0 | 0 |
| pi0.5 | 200 | 2 | 157 | 36 | 5 | 0 | 1 |

**Position** (the requested food moved; its training place is also the place of the scene's own task):

| Policy | Episodes | Requested food | Its training place | Another object | No close | Success |
|---|---|---|---|---|---|---|
| OpenVLA-OFT | 100 | 0 | 100 | 0 | 0 | 0 |
| SmolVLA | 200 | 0 | 200 | 0 | 0 | 0 |
| TurboVLA | 100 | 0 | 100 | 0 | 0 | 0 |
| pi0-FAST | 200 | 0 | 199 | 1 | 0 | 0 |
| pi0 | 200 | 0 | 197 | 3 | 0 | 0 |
| pi0.5 | 200 | 42 | 121 | 37 | 0 | 16 |

The ten scenes:

| Task | Scenes | Instruction | At the requested food's training place | At the scene task's place |
|---|---|---|---|---|
| 0 | Position | pick up the alphabet soup and place it in the basket | butter_1 | butter_1 |
| 3 | Position | pick up the bbq sauce and place it in the basket | chocolate_pudding_1 | chocolate_pudding_1 |
| 6 | Position | pick up the butter and place it in the basket | tomato_sauce_1 | tomato_sauce_1 |
| 8 | Position | pick up the chocolate pudding and place it in the basket | ketchup_1 | ketchup_1 |
| 9 | Position | pick up the orange juice and place it in the basket | chocolate_pudding_1 | chocolate_pudding_1 |
| 0 | Task-paired | pick up the cream cheese and place it in the basket | salad_dressing_1 | alphabet_soup_1 |
| 3 | Task-paired | pick up the ketchup and place it in the basket | chocolate_pudding_1 | bbq_sauce_1 |
| 6 | Task-paired | pick up the orange juice and place it in the basket | tomato_sauce_1 | butter_1 |
| 8 | Task-paired | pick up the salad dressing and place it in the basket | orange_juice_1 | chocolate_pudding_1 |
| 9 | Task-paired | pick up the chocolate pudding and place it in the basket | butter_1 | orange_juice_1 |

## How it is read

Task-paired separates two places that Position leaves as one. A hand that goes to the place of the scene's
own task follows the scene; a hand that goes to the requested food's training place follows the instruction
to a remembered place; a hand that closes on the requested food follows the food.

- OpenVLA-OFT and TurboVLA first close at the scene task's place in 100 of 100 episodes, SmolVLA in 167 of
  200: the instruction changes little.
- π0 first closes at the requested food's training place in 197 of 200 and π0-FAST in 191 of 200: the
  instruction changes where the hand goes, to a place and not to the food (0 closes on the food in both).
- π0.5 does the same in 157 of 200, goes to the scene task's place in 36, and closes on the requested food
  in 2.
- Under Position only π0.5 first closes on the moved food: 42 of 200.

## Beside this repository's own runs

- **Position was counted again here**, from the tables' episode tables: the same five tasks, start states
  0-49, one noise seed, 250 episodes per policy ([`evaluation/tables/README.md`](../../evaluation/tables/README.md)). There π0.5 first closes on the
  moved food in 60 of 250 and the other five policies in 0 of 250. Other episodes and other settings, so the
  two counts are not expected to be equal: 42 of 200 is what the earlier draft's figure showed.
- **What differs from the tables' rows.** OpenVLA-OFT and TurboVLA are here the releases' checkpoints for the
  Object suite alone; the tables run the checkpoints trained on all four suites. SmolVLA is another
  revision of the same release and, as that release runs it, executes a whole chunk before it is asked again;
  the tables' row executes 10 actions per query. The step limit is 220 here and 280 in the tables, the
  environment seed 0 and 7. The first close is labelled by horizontal distance to the body here and by the
  nearest geometry centre in the tables. How many actions the other policies executed per query in these
  runs is not recorded here.
- **Task-paired is not LIBERO-PRO's Task perturbation as the tables run it** (`PT-O`): that block gives
  LIBERO-PRO's own changed sentence; this one requests another food with the trained wording.
