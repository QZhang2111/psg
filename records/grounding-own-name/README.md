# Does a moved food get the name's attention under its own name more than under another's?

**Recorded from an earlier repository of this project; not re-run here.** No code in this
repository reads attention.

This is panel (a) of the paper's grounding figure. Panel (b), the Reach of the same three foods under the
method's settings, is made here from `results/` ([`PAPER.md`](../../PAPER.md) at the root, the grounding
figure's row: the two single-term rows).

## What was measured

- **Model.** π0.5, the released LIBERO checkpoint.
- **Frames.** LIBERO-PRO Position on LIBERO-Object, three tasks: the alphabet soup (task 0), the BBQ sauce
  (task 3), the butter (task 6); start states 6-15, one noise seed; the observation of the first query. 10
  frames per food, 30 in all. In each the requested food stands away from its training place.
- **Passes.** Six foods are in view in every frame. Each frame is passed six times, the instruction naming
  one of them each time ("pick up the <food> and place it in the basket"): 180 frame × name pairs. One pass
  is the model's prefix pass with both camera images; no action is generated.
- **Name attention on a food.** The attention of the name's tokens on the main camera's 256 patches,
  averaged over the name's tokens and all 18 layers × 8 heads; on a food, weighted by the food's pixels per
  patch and divided by the camera's mean per patch. 1 is the image average.
- **The ratio.** Per frame, the requested food's value when the instruction names it, divided by the highest
  value the same food gets in that frame under the five other names. Above 1: the food is attended most under
  its own name. Below 1: some other name puts more of its attention on this food than the food's own name
  does.

## The tables

| File | What it holds |
|---|---|
| [`own_name_ratio.csv`](own_name_ratio.csv) | the 30 frames: the requested food's value under its own name, the highest under another name, which name that was, the ratio |
| [`name_swap.csv`](name_swap.csv) | the 180 frame × name pairs: the requested food's value under each name, and which food had the most name attention in that pass |
| [`name_swap_three_models.csv`](name_swap_three_models.csv) | the same 180 pairs in each of three models (`paligemma`, `pi05_base`, `pi05_libero`), 540 lines, with every value the earlier repository read out: the requested food's rank, margin, ratio and density, the leader and the ranking, the camera's share, and the density on each food. The grounding figure was drawn from this file |

| Requested food | Frames | Ratio, median | Smallest | Largest | Frames above 1 |
|---|---|---|---|---|---|
| alphabet_soup_1 | 10 | 3.32 | 1.96 | 4.01 | 10 of 10 |
| butter_1 | 10 | 2.84 | 1.69 | 4.27 | 10 of 10 |
| bbq_sauce_1 | 10 | 0.32 | 0.14 | 0.55 | 0 of 10 |

## How it is read

- The soup and the butter get more name attention under their own names than under any other name in every
  one of their 10 frames (median ratios 3.3 and 2.8).
- The BBQ sauce gets less under its own name than under another in every one of its 10 frames, and the other
  name is the same each time: "ketchup" (`own_name_ratio.csv`, column `that_other_name`). In that scene
  the two bottles are crossed: under "ketchup" the BBQ sauce has the most name attention of all foods in 10
  of 10 frames, and under "bbq sauce" the ketchup has it in 7 of 10 (`name_swap.csv`).
- Beside panel (b), counted in this repository on 50 episodes per food: the two foods whose own name finds
  them are the two whose Reach rises from the released policy to the method (the soup 21 to 42, the butter
  25 to 50); the BBQ sauce stays at 0.

## What this record is not

- 10 frames per food, one start frame each, one kind of scene. Whether a food is attended under its own name
  is not a property of the food alone: in the earlier repository's own summary the BBQ sauce ranks first
  under its name in the butter's scene and not in its own.
- The same passes were made for PaliGemma and for π0.5's base checkpoint; they are not in the paper's panel
  and not recorded here.
- The value here weights patches by the food's pixels; the record `name-attention-lineage` counts every patch
  the object touches equally, on other scenes. The two are not on one scale.
