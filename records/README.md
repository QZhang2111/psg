# Records: what the paper uses and this repository does not make

A record is a measurement the paper's figures or text use that was made in an earlier repository of this
project and **is not made again here**: no code in this repository produces it, and nothing under `results/`
holds it. Each folder holds the small tables the paper reads and one page that says what was measured, on
what, and how it is read. Every record was made in that earlier repository and was not re-run here; its
page says so in its first line.

What this repository does make is in `results/`: the three tables, the pathway cuts made again, the oracle
push, and the first-close counts; `results/_figures/` keeps the data of the method figure and of the direction panel ([`evaluation/tables/README.md`](../evaluation/tables/README.md)).

| Record | What it shows |
|---|---|
| [`cut-attention`](cut-attention/README.md) | the action tokens' attention on the moved food at the first query, with each instruction pathway cut |
| [`pathway-cuts-original-run`](pathway-cuts-original-run/README.md) | the earlier run of the pathway cuts, 200 episodes per arm: the counts the paper prints (29, 0 and 64 of 200) |
| [`behaviour-task-paired`](behaviour-task-paired/README.md) | where six released policies first close when another food of the unchanged scene is requested (and under Position, as first run) |
| [`name-attention-lineage`](name-attention-lineage/README.md) | whether the object's name attends to the object when it is moved: the wine bottle in four models, and nine objects along the π and OpenVLA lineages |
| [`grounding-own-name`](grounding-own-name/README.md) | whether three moved foods get more of the name's attention under their own name than under another food's, in π0.5 |
| [`attention-by-distance`](attention-by-distance/README.md) | the name's and the action tokens' attention on the requested food by the gripper's distance to it, in the main and the wrist camera |
| [`psg-weights-selection`](psg-weights-selection/README.md) | how a = 5 and b = 1 were chosen: the method's two weights on eleven development tasks under Original, Position and Task |
