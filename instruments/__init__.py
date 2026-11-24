"""The instruments: two interventions on the sampler, for the paper's Sec. 3.3 and its direction-and-oracle figure.

They are a part apart from the method (`psg/`), which imports nothing from here, and no row of the
paper's tables imports this package (`tests/test_oracle_push.py` holds both).

`oracle_push.py`        an intervention: the grounding term replaced by a push of a fixed size along the true
                        direction to the object, which the simulator tells the row through the oracle channel
                        (`evaluation/oracle.py`); the row `pi05_oracle_push` (Fig. 7 b).
`pathway_cuts.py`       an intervention: one of the instruction's two ways to the action is cut (the image
                        tokens may not read it, or the action side may not), on π0.5 and on π0, the paper's
                        Sec. 3.3; the rows `pi05_cut_*` and `pi0_cut_*`. `pathway_cuts_sampler.py` is the
                        sampler, `pathway_cuts_row.py` the rows' adapter.
"""
