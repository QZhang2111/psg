"""The evaluation: one command for every row of the paper's tables (`python -m evaluation.run`).

The modules of this folder are shared by every row: the interface, the rollout, the record, the Reach labeller,
the collection tool. `rows/<row>/` is one row's adapter, `blocks/<block>/` one test set, `environments/` how
each environment is built, `tables/` the paper's tables from `results/`. See README.md in this folder.
"""
