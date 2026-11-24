# Instruments: the pathway cuts and the oracle push

No table reads anything here. Every command below is run from the repository's root; how a unit is run and
collected is in [`evaluation/README.md`](../evaluation/README.md).

`instruments/` is a part apart from the method: `psg/`
imports nothing from it and no row of a table does.

## The oracle push (the direction-and-oracle figure, panel b)

The row `pi05_oracle_push` is π0.5 with PSG's grounding term replaced by a push of a fixed size along the
true direction to the object (`instruments/oracle_push.py`). **This row reads object poses**: the
simulator tells it where the task's Reach target is, through the oracle channel
(`evaluation/oracle.py`), which is off for every other row and written into the unit's config record
(`observation.oracle`). It is no policy and no row of a table. Its size is `push.size` in
`evaluation/rows/pi05_oracle_push/row.json`, in executed-action units: 1.164 since 2026-11-14, four times the median
size of PSG's own push before the first close on the same 100 episodes (0.291,
`median_a_g_norm_before_first_close` in `results/_figures/fig7a/direction.csv.summary.json`), chosen from trials
of 0.291, 0.582 and 1.164 on the ten scenes at start state 6; the controller clips an action at 1, which the
next doubling would exceed by the push alone. A row whose definition names no size does not load (a trial of
another size: edit the file and pass `--smoke`; a run that is kept: commit the number first).

On the ten Goal scenes of Table 2 whose object stands more than 30 cm from its training place (block `OS-G`,
tasks 1, 3, 6, 8, 10, 15, 17, 18, 19, 20; 100 episodes):

```
T=1,3,6,8,10,15,17,18,19,20
python -m evaluation.run --block OS-G --row pi05_oracle_push --tasks $T
python -m evaluation.collect outputs --row pi05_oracle_push --part
```

Only Reach is read from its episode table. `collect --part` takes a unit that was run on named tasks of its
block and keeps the tasks in the collected record's `selection`; without the flag such a unit is refused, as
every table's unit must be whole. Panel (a), the direction of PSG's grounding term, is a readout of
`pi05_psg`'s traces made by a tool this repository no longer holds; its tables are kept in
`results/_figures/fig7a-os-o/` (the figure's input) and `results/_figures/fig7a/`.

## The pathway cuts (the paper's Sec. 3.3)

Six rows that belong to no table: `pi05_cut_none`,
`pi05_cut_grounding`, `pi05_cut_direct`, and `pi0_cut_none`, `pi0_cut_grounding`, `pi0_cut_direct`. Each is the
released checkpoint with one of the instruction's two ways to the action cut, in every layer and head, at every
Euler step of every query: under `grounding` the image tokens of both cameras may not read the instruction,
under `direct` the action side may not (the action tokens, and in π0 the state token); `none` cuts nothing and
is the released policy through the same program. The instruction is the whole sentence the model is given,
from `<bos>` to the closing newline. These are not PSG's cuts, and the cut model's own velocity is integrated
in every action dimension (`instruments/pathway_cuts.py`). The unit's config record lists the cut
positions with their tokens (`method.settings.instruction`).

A row checks its own cut at the first query of every episode, in its own process and bit for bit: with nothing
cut its program returns the released policy's chunk, and the actions it would execute from that chunk are
those `Policy.infer` returns; under the direct cut the chunk does not change when the
instruction's keys and values are replaced; under the grounding cut what the backbone leaves at the image
tokens does not change with the instruction's words. An episode whose check does not hold stops with an error,
and a unit with a failed episode is not collected.

By hand, on five tasks of LIBERO Object (0 the alphabet soup, 3 the BBQ sauce, 6 the butter, 8 the chocolate
pudding, 9 the orange juice), with the objects' places exchanged (`PP-O`) and in the original scenes (`L-O`);
250 episodes per arm:

```
T=0,3,6,8,9
for ROW in pi05_cut_none pi05_cut_direct; do
  python -m evaluation.run --block PP-O,L-O --row $ROW --tasks $T
done
for ROW in pi05_cut_grounding pi0_cut_none pi0_cut_grounding pi0_cut_direct; do
  python -m evaluation.run --block PP-O --row $ROW --tasks $T
done
python -m evaluation.collect outputs --part      # with --block and --row: one unit
```

Run so on 2026-11-16: π0.5's five arms on one machine, π0's three on another, 2,000 episodes, none in error, and
every check zero in every episode. The arm that cuts nothing
is the released policy's episodes bit for bit on one card (four of four trial episodes for each model); it is
not the tables' unit bit for bit, because that ran in another environment (π0.5 on `PP-O`: success 37 against
36 of 250). A cut arm is therefore set beside the nothing-cut arm of the same run, not beside a table's number.

Where an arm first closed the gripper is counted from its collected episode table, like any row's:
`python -m evaluation.tables.first_close results/PP-O/pi05_cut_direct/episodes.csv --tasks 0,3,6,8,9`
([`evaluation/tables/README.md`](../evaluation/tables/README.md)).

