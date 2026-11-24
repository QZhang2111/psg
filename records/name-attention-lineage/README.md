# Does the object's name attend to the object when it is moved, along each model's lineage?

**Recorded from an earlier repository of this project; not re-run here.** No code in this
repository reads attention, and none loads PaliGemma, the base checkpoints, Prismatic or OpenVLA.

This is the paper's name-attention figure: (a) the wine bottle, where the words "wine bottle" attend before
and after the bottle is moved; (b) the name's attention on its object from the vision-language model through
the robot-pretrained base to the LIBERO policy, for the three π lineages and OpenVLA.

## What was measured

- **Scenes.** Nine objects of LIBERO-Goal and LIBERO-Object (Goal: the wine bottle, the bowl, the plate, the
  cream cheese; Object: the ketchup, the milk, the alphabet soup, the butter, the cream cheese), each at its
  training place and at six other places, in two start states: 126 scenes, 12 moved and 2 at the training
  place per object.
- **One pass per scene, no action.** The model is given the main-camera image (224 px, 16 × 16 patches) and
  the task's sentence; where a model takes a wrist image, it is blank. The attention of the name's tokens
  (for example "wine bottle") on the 256 image patches is read and averaged over all layers and heads.
- **Name attention on the object** (panel b): the name's attention per patch on the object's own patches
  divided by its attention per average patch of the image. **1 is the image average.** A patch belongs to the
  object when it holds at least one of its pixels. Per object the median over its scenes; then the geometric
  mean over the nine objects.
- **Share** (panel a): the name's attention on a set of patches as a percentage of its attention on the whole
  image. Two sets, read before and after the move: the patches the bottle covers at its training place, and
  the patches it covers at its new place (before the move the bottle does not touch those patches).
  Medians over the 12 moved scenes; "before" is read in the training-place scenes, for the new place in the
  one of the same start state.
- **Control.** Each scene is also read with another object's sentence, on the same patches: the column
  "another object's name".

| Lineage | Stage | Checkpoint | Layers × heads |
|---|---|---|---|
| π0, π0-FAST, π0.5 | vision-language model | PaliGemma 3B, 224 px, the weights openpi starts from | 18 × 8 |
| π0, π0-FAST, π0.5 | robot-pretrained base | openpi's `pi0_base`, `pi0_fast_base`, `pi05_base` | 18 × 8 |
| π0, π0-FAST, π0.5 | LIBERO fine-tuned | openpi's `pi0_libero`, `pi0_fast_libero`, `pi05_libero` | 18 × 8 |
| OpenVLA | vision-language model | Prismatic, DINOv2 + SigLIP at 224 px, Llama-2 7B | 32 × 32 |
| OpenVLA | robot-pretrained base | OpenVLA 7B | 32 × 32 |
| OpenVLA | LIBERO fine-tuned | OpenVLA-OFT, the LIBERO-Object checkpoint | 32 × 32 |

Three things that are particular to a checkpoint, as the earlier repository states them: OpenVLA-OFT's
language model attends in both directions as it is served, while Prismatic and OpenVLA are causal (their image
tokens cannot read the words); OpenVLA-OFT was given the same uncropped image as the other two stages, not
its served centre crop; π0-FAST's prefix carries the robot state, read here with quantile normalisation.

## The tables

| File | What it holds |
|---|---|
| [`lineage.csv`](lineage.csv) | panel (b): per checkpoint, name attention on the object when moved and at its training place, and with another object's name; scenes, and the smallest and largest object |
| [`lineage_per_object.csv`](lineage_per_object.csv) | the same per object: ten checkpoints × nine objects × moved or at the training place |
| [`wine_bottle.csv`](wine_bottle.csv) | panel (a): four models, the share on the training place and on the new place, before and after the move |
| [`wine_bottle_map.csv`](wine_bottle_map.csv) | panel (a)'s maps: per model and patch, the change of the share in percentage points (row 0 is the top and column 0 the left of the image as the policy sees it; the paper's figure is drawn mirrored left to right) |

**Panel (b): name attention on the object, relative to the image average** (geometric mean over nine objects):

| Lineage | Stage | Checkpoint | Object moved | Object at its training place | Another object's name, object moved |
|---|---|---|---|---|---|
| pi (shared root) | VLM | PaliGemma 3B, 224 px | 2.699 | 2.460 | 1.833 |
| pi0 | robot-pretrained base | pi0 base | 0.755 | 0.963 | 0.702 |
| pi0 | LIBERO fine-tuned | pi0 LIBERO | 0.791 | 0.958 | 0.845 |
| pi0-FAST | robot-pretrained base | pi0-FAST base | 0.741 | 1.446 | 0.775 |
| pi0-FAST | LIBERO fine-tuned | pi0-FAST LIBERO | 1.229 | 3.871 | 0.631 |
| pi0.5 | robot-pretrained base | pi0.5 base | 2.898 | 3.060 | 1.715 |
| pi0.5 | LIBERO fine-tuned | pi0.5 LIBERO | 6.376 | 7.039 | 1.303 |
| OpenVLA | VLM | Prismatic DINOv2 + SigLIP 224 px, 7B | 2.954 | 2.909 | 1.863 |
| OpenVLA | robot-pretrained base | OpenVLA 7B | 1.375 | 1.718 | 1.193 |
| OpenVLA | LIBERO fine-tuned | OpenVLA-OFT, LIBERO-Object | 1.499 | 1.446 | 1.427 |

**Panel (a): the wine bottle, share of the name's attention on the image, in %** (medians):

| Model | On the training place, before | after | On the new place, before | after |
|---|---|---|---|---|
| PaliGemma | 23.0 | 3.3 | 2.0 | 19.3 |
| pi0 LIBERO | 11.3 | 8.9 | 0.9 | 3.7 |
| pi0-FAST LIBERO | 15.4 | 6.1 | 2.1 | 0.4 |
| pi0.5 LIBERO | 47.7 | 2.0 | 0.1 | 29.8 |

## How it is read

- **The wine bottle.** In PaliGemma and in π0.5's LIBERO policy the name's attention leaves the training
  place and arrives on the bottle's new place (π0.5: 47.7 → 2.0 on the training place, 0.1 → 29.8 on the new
  place: the paper's "0 % to 30 %"). In π0's and π0-FAST's LIBERO policies it does not arrive there (3.7 and
  0.4).
- **Along the lineages.** PaliGemma's name attends to the moved object at 2.7 times the image average. π0.5
  keeps this through its base (2.9) and its LIBERO policy has more (6.4: the paper's number). π0 and π0-FAST
  have about the image average or less from the base on (0.76 and 0.79; 0.74 and 1.23).
- **OpenVLA.** Prismatic has 3.0, OpenVLA 1.4 and OpenVLA-OFT 1.5. For OpenVLA-OFT the value with another
  object's name on the same patches is nearly the same (1.43 against 1.50), so what is left there does not
  depend on which object is named; for π0.5's LIBERO policy the two are 1.30 and 6.38.
- The spread over objects is large and is in `lineage_per_object.csv`: for π0.5's LIBERO policy the nine
  medians run from 1.5 (the milk) to 12.6 (the wine bottle).

## What this record is not

- Not a measurement during an episode: one pass on a start frame, no action generated, the wrist image blank.
- The training-place values rest on 2 scenes per object, the moved ones on 12.
- The SmolVLA lineage was read in the earlier repository as well and is not in the paper's figure or here.
