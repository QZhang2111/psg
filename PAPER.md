# The paper, by table and figure

The tables and figures of the paper's main text, and what this repository holds of each.

"Made here" means this repository's code produces it and the result lies under `results/`. "Recorded, not
re-run" means the paper uses a measurement made earlier and no code here produces it; it is kept under
`records/` as small tables with one page each. The tables are made here and printed by one command (below).
The figures are not drawn by this repository: for each, the measurements it shows lie under `records/` or
`results/`, without a script.

| In the paper | Where | |
|---|---|---|
| Table 1: ten policies and methods on LIBERO, LIBERO-PRO Position and Task, LIBERO-Plus Layout | `results/L-*/`, `results/PP-*/`, `results/PT-*/`, `results/PL/`; as printed: [`evaluation/tables/printed/table1.csv`](evaluation/tables/printed/table1.csv) | made here (82 units); the LIBERO column of seven rows is cited from their authors ([`cited_libero.csv`](evaluation/tables/cited_libero.csv)); SmolVLA is one of them, although its four LIBERO units were run here too |
| Table 2: the same ten on our Goal scenes, by how far the object was moved | `results/OS-G/`; [`table2.csv`](evaluation/tables/printed/table2.csv) | made here (10 units) |
| Table 3: π0.5, CAG, CAG-VA and PSG on LIBERO-CF | `results/CF-S/`, `results/CF-O/`, `results/CF-OOD/`; [`table3.csv`](evaluation/tables/printed/table3.csv) | made here (12 units) |
| Name-attention figure | [`records/name-attention-lineage/`](records/name-attention-lineage/README.md) | recorded, not re-run |
| Pathway-cut figure and Sec. 3.3: π0.5 and π0 with one of the instruction's two ways to the action cut | the original run: [`records/pathway-cuts-original-run/`](records/pathway-cuts-original-run/README.md); the arms made again: `results/PP-O/pi05_cut_*/`, `results/PP-O/pi0_cut_*/`, `results/L-O/pi05_cut_none/`, `results/L-O/pi05_cut_direct/`; the attention panel as a table: [`records/cut-attention/`](records/cut-attention/README.md) | the counts the paper prints (29, 0 and 64 of 200) are the original run's, 200 episodes per arm: recorded, not re-run. The arms were made again here on 250 episodes per arm (8 units, [`instruments/README.md`](instruments/README.md)); those are other episodes and give other numbers |
| Method figure: one episode as the released policy and as PSG ran it | `results/_figures/fig4-butter/` | made by the replay instrument of this code's earlier form (`pi05lab/instruments`, as its config records say), which this repository no longer holds; the episode's own units, `results/PP-O/pi05/` and `results/PP-O/pi05_psg/`, are made here |
| Grounding figure: (a) the name's attention per object, (b) Reach with each of PSG's two terms alone | (a) [`records/grounding-own-name/`](records/grounding-own-name/README.md); (b) `results/PP-O/` of the rows `pi05`, `pi05_psg_a5_b0`, `pi05_psg_a0_b1`, `pi05_psg` | (a) recorded, not re-run; (b) made here |
| Attention-by-distance figure | [`records/attention-by-distance/`](records/attention-by-distance/README.md) | recorded, not re-run |
| Direction-and-oracle figure: (a) the direction of PSG's grounding term, (b) Reach with a push in the true direction | (a) `results/_figures/fig7a-os-o/` (`results/_figures/fig7a/` is the same readout on the ten far `OS-G` scenes, which gave the oracle push its size); (b) `results/OS-G/` of the rows `pi05`, `apt`, `pi05_cag`, `pi05_rss`, `pi05_psg`, `pi05_oracle_push` | (a) read from `pi05_psg`'s traces by a readout this repository no longer holds; (b) made here |
| How a = 5 and b = 1 were chosen | [`records/psg-weights-selection/`](records/psg-weights-selection/README.md) | recorded, not re-run |

The main text opens with a figure of three policies on task-paired scenes; this table does not list it. An earlier draft had a figure of six policies' first closes instead; its
measurement is the record [`records/behaviour-task-paired/`](records/behaviour-task-paired/README.md).

The figures are named by what they show, because their numbers in the paper have moved. The folders under
`results/_figures/` and some config records keep the figure numbers of an earlier draft: "Fig. 4" is the method
figure, "Fig. 5 (b)" panel (b) of the grounding figure, "Fig. 7" the direction-and-oracle figure.

`evaluation/tables/printed/` holds the three tables as the paper prints them. 215 of their 222 numbers come from
`results/`; the other seven are LIBERO averages cited from their authors (`cited_libero.csv`, marked `*`). One
of the seven is SmolVLA's 87.3; the LIBERO blocks run here with SmolVLA gave 83.8
(`results/L-*/smolvla_exec10/`). The tables of `printed/`, from these two sources (any Python >= 3.8, no card):

```
python -m evaluation.tables.render            # prints Tables 1-3
python -m evaluation.tables.render --check    # compares all 222 numbers with the tables as printed
```
