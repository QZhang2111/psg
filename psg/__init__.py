"""Pathway-Separated Guidance (PSG): the method of the paper.

`sampler.py` is the method: three velocities per Euler step and one line that combines them.
`name_rule.py` says which instruction tokens are "the name".
`policy.py` puts the two on an openpi π0.5 policy: one call returns one chunk of actions.

This package imports openpi's model and nothing of the evaluation or of the instruments.
"""
