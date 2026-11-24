# Attention on the requested food by the gripper's distance to it, in each camera

**Recorded from an earlier repository of this project; not re-run here.** No code in this
repository reads attention. The table was counted again, for this record, from the earlier repository's
recorded observations; no episode and no model pass was run.

This is panel (b) of the paper's attention-by-distance figure: how much the name's tokens and the action
tokens attend to the requested food, in the main camera and in the wrist camera, as a function of how far the
gripper is from the food. Panel (a) of that figure is a staged top view, drawn for illustration; it is no
measurement and is not recorded.

## What was measured

- **Model.** π0.5, the released LIBERO checkpoint, nothing changed.
- **Episodes.** Ten foods of LIBERO-Object, each at its training place and at six other places, start states
  6-15, two noise seeds: 1,378 episodes. (This repository's block `OS-O` has the same ten foods, seven places
  and start states: 689 pairs of scene and start state, one seed.)
- **Observations.** Six per episode, taken along its first 30 commanded actions (at 0, 5, 10, 15, 20 and
  30): 8,268 observations. The model's pass on each was read.
- **Attention on the food.** The attention of the name's tokens, and of the action tokens, on the patches of
  one camera, averaged over all 18 layers × 8 heads; on the food relative to that camera's mean per patch.
  **1 is the camera's average.** An observation counts for a camera only when the food has pixels in it.
- **Distance.** The horizontal distance between the gripper and the food at that observation, in seven
  bins. Each value below is the median over the bin's observations, the ten foods and all places together.

## The table

[`attention_by_distance.csv`](attention_by_distance.csv): 28 lines (name or action tokens × main or wrist
camera × seven bins) with the number of observations, the median, and the value as the paper's figure draws
it.

| Tokens | Camera | 0-5 cm | 5-10 | 10-15 | 15-20 | 20-25 | 25-30 | 30-40 |
|---|---|---|---|---|---|---|---|---|
| name | main | 15.0 | 12.4 | 11.5 | 12.4 | 13.5 | 9.9 | 6.6 |
| action | main | 7.2 | 5.2 | 4.2 | 3.8 | 3.4 | 2.3 | 1.8 |
| name | wrist | 6.3 | 8.4 | 7.8 | 5.8 | 3.1 | 0.55 | 0.24 |
| action | wrist | 3.3 | 3.0 | 2.3 | 1.8 | 0.83 | 0.43 | 0.31 |
| observations | main | 202 | 316 | 403 | 565 | 1418 | 1764 | 3464 |
| observations | wrist | 202 | 316 | 403 | 543 | 1352 | 1612 | 1243 |

Two of the 28 values differ from the figure's in the last digit: the figure has 12.5 and 8.5 at 5-10 cm for
the name (main and wrist); the medians are 12.445 and 8.449. That is what rounding twice gives
(12.445 to 12.45 to 12.5; 8.449 to 8.45 to 8.5). The other 26 are equal at the figure's own rounding.

## How it is read

- **Main camera, name.** The name attends to the food far above the camera's average at every distance:
  11.5 to 15.0 within 25 cm, 9.9 and 6.6 beyond.
- **Main camera, action tokens.** Their attention on the food falls with distance at every step, from 7.2
  within 5 cm to 1.8 at 30-40 cm.
- **Wrist camera.** Both fall below the camera's average once the gripper is far: the name beyond 25 cm (0.55,
  0.24), the action tokens beyond 20 cm (0.83, 0.43, 0.31), in observations where the food still has pixels
  in the wrist image.
- So of the four readings only the name's in the main camera stays tied to the food while the gripper is
  still far from it.

## What this record is not

- Not a per-food result: the bins pool ten foods and seven places, and near bins hold fewer observations
  (202 within 5 cm) than far ones.
- Distance and time are not separated: the gripper is far early in an episode and near late.
- Nothing here says where the gripper went; first closes are counted from episode tables
  ([`evaluation/tables/README.md`](../../evaluation/tables/README.md)).
