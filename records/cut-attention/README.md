# The action tokens' attention on the moved food, with each instruction pathway cut

**Recorded from an earlier repository of this project; not re-run here.** No code in this
repository reads attention.

This is the attention panel of the paper's pathway-cut figure: when one of the two ways the instruction reaches
the action is cut, does the action expert still attend to the food that was moved?

## What was measured

- **Models.** The released LIBERO checkpoints of π0.5 and π0.
- **Scenes.** LIBERO-PRO on five tasks of LIBERO-Object (the alphabet soup, the BBQ sauce, the butter, the
  chocolate pudding, the orange juice), start states 0-19, noise seeds 7 and 17: 200 episodes per kind of
  scene. **Position:** the requested food exchanged with another food, the instruction unchanged.
  **Task-paired:** the scene unchanged, another food of the scene requested.
- **When.** The first query of an episode: the start state after the settling steps, before the policy has
  acted. Each episode's frame was taken from the run with nothing cut, and one forward pass was made per cut
  on that same frame with the same noise. So the cuts are compared on identical input.
- **The cuts.** "The instruction" is every prompt position, from `<bos>` to the closing newline.

  | Cut | What is denied, in every layer and head |
  |---|---|
  | nothing cut | nothing: the released policy |
  | grounding cut | the image tokens of both cameras may not read the instruction |
  | direct cut | the action side (the action tokens, and π0's state token) may not read the instruction |
  | both cuts | both of the above |
  | no instruction | nothing is cut; the policy is given the empty instruction |

- **Attention on a region.** The action tokens' attention weights on each of the main camera's 256 image
  patches (read after the softmax, in 32-bit floats), averaged over the 8 heads and the action tokens (10 in
  π0.5, 50 in π0), then over the 18 layers and the 10 Euler steps, all with equal weight. A region's value is
  its mean per patch divided by the main camera's mean per patch. **1 is the image average**; 2 means a patch
  of the region gets twice the attention of an average patch.
- **Regions**, from the simulator's segmentation in the policy's view at the patch grid: a patch with any pixel
  of the requested food is the requested food; otherwise the scene object with the most pixels in it (the body
  at the requested food's training place; under Task-paired the food of the scene's own task; other foods; the
  basket); otherwise the robot; otherwise the rest of the image.
- **Ratio to nothing cut.** The region's value under a cut divided by its value with nothing cut, on the same
  frame and noise; the table holds the median and quartiles over the episodes.

Checks the run kept for each model (2,000 passes): with nothing cut the pass returns the released sampler's
actions bit for bit in 400 of 400; the action tokens' attention on the instruction is exactly zero in all 800
passes under the direct cut and both cuts, and above zero in all 1,200 others.

## The table

[`attention.csv`](attention.csv): one line per model, kind of scene, cut and region (130 lines), with the
number of episodes, the median and quartiles of the attention, and of the ratio to nothing cut.

The paper's panel draws the requested food under Position for three cuts:

| Model | Cut | Episodes | Attention, median | Quartiles | Ratio to nothing cut, median |
|---|---|---|---|---|---|
| pi0 | nothing cut | 200 | 0.577 | 0.481 to 0.607 | – |
| pi0 | grounding cut | 200 | 0.584 | 0.482 to 0.615 | 1.011 |
| pi0 | direct cut | 200 | 0.565 | 0.483 to 0.588 | 0.972 |
| pi05 | nothing cut | 200 | 1.925 | 1.425 to 2.893 | – |
| pi05 | grounding cut | 200 | 1.552 | 0.717 to 1.871 | 0.540 |
| pi05 | direct cut | 200 | 1.867 | 1.551 to 3.139 | 1.041 |

## How it is read

- With nothing cut, π0.5's action tokens attend to the moved food at about twice the image average (median
  1.9); π0's at about half of it (0.58).
- Under the grounding cut, π0.5's attention on the moved food falls in half of the episodes to 0.54 of its
  value or less (the median of the per-episode ratio). It does not fall everywhere: the quartiles of that
  ratio are 0.43 and 1.22, so in at least a quarter of the episodes it rose. Under the direct cut it stays
  where it was (median ratio 1.04, quartiles 0.99 and 1.14).
- π0's attention on the moved food does not move under either cut (median ratios 1.01 and 0.97).
- The two medians of a row answer different questions. "Attention, median" is the median over episodes of
  the value itself; "ratio, median" is the median of each episode's own change. For π0.5 under the grounding
  cut the first goes from 1.93 to 1.55 and the second is 0.54.

## What this record is not

- Not where the gripper closes under the cuts: first closes are counted from episode tables
  ([`evaluation/tables/README.md`](../../evaluation/tables/README.md)).
- Not a statement about later queries: the attention was read at the first query only.
- The wrist camera's values and the per-task tables exist in the earlier repository and are not recorded here.
