"""Which tests run where, and with which variables.

The tests that build a model or compile JAX need the `openpi` environment and a few GB of memory (the PSG
sampler test peaks at about 3.4 GB). They are left out, not even imported, unless PSG_HOST_TESTS=1 is set. Everything
else here uses the standard library and NumPy.

With PSG_HOST_TESTS=1 this file also sets, before any test is imported, the variables a unit runs with:
those `evaluation/environments/openpi.json` lists under "variables" (deterministic operations on a card), the same
source `evaluation/run.py` reads for a unit. Nothing here chooses the device: JAX computes these tests on a card when it
sees one and on the CPU when it does not. Without the variables a card gave different numbers from process to
process (11 or 12 different results in 20 fresh processes); with them, and on the CPU, one result in 20.
The session stops at its start if JAX is already imported when the variables would
have to be set: a variable set after that may not be read.
"""

import json
import os
import sys
from pathlib import Path

HOST_TESTS = ["test_psg.py", "test_oracle_push_sampler.py", "test_pathway_cuts_sampler.py"]

HOST = os.environ.get("PSG_HOST_TESTS") == "1"
collect_ignore = [] if HOST else list(HOST_TESTS)

DECLARATION = Path(__file__).resolve().parents[1] / "evaluation" / "environments" / "openpi.json"
# What a reader of a test log needs to know where and how JAX computed; printed in the session's header.
REPORTED = ("XLA_FLAGS", "TF_CUDNN_DETERMINISTIC", "CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS")


def session_variables(declared, given):
    """The declared variables as a test session gets them, {name: value}.

    declared: the "variables" of `evaluation/environments/openpi.json`. given: the caller's environment. A declared
    variable wins over the caller's value, as it does for a unit. `XLA_FLAGS` is a list of flags separated by
    spaces: the caller's other flags are kept and the declared ones are added after them, once.
    """
    out = {}
    for name, value in declared.items():
        value = str(value)
        if name == "XLA_FLAGS":
            mine = (given.get(name) or "").split()
            value = " ".join(mine + [flag for flag in value.split() if flag not in mine])
        out[str(name)] = value
    return out


def set_session_variables(environ, jax_is_imported):
    """Sets the declared variables in `environ` and returns {name: value} of what it changed.

    Raises RuntimeError when a variable would have to change while JAX is already imported: whether XLA still
    reads it then is not known here, and a session that only looks as if it had the settings is worse than one
    that stops. A caller who sets the variables itself, before starting pytest, is not affected."""
    wanted = session_variables(json.loads(DECLARATION.read_text())["variables"], environ)
    changed = {name: value for name, value in wanted.items() if environ.get(name) != value}
    if changed and jax_is_imported:
        raise RuntimeError(
            "tests/conftest.py: JAX was imported before the tests' variables could be set (%s). Something imports "
            "it before this file is read (a pytest plugin, a sitecustomize). Set them in the caller's environment "
            "instead, as evaluation/environments/openpi.json states them, and report this." % ", ".join(sorted(changed))
        )
    environ.update(changed)
    return changed


# At import, which pytest does before it imports any test of this directory.
SET_HERE = set_session_variables(os.environ, "jax" in sys.modules) if HOST else {}


def pytest_report_header(config):
    """One line in the session's header (not shown with -q): the variables that decide where and how JAX computes."""
    if not HOST:
        return None
    return "host tests: %s (set by tests/conftest.py: %s); JAX takes a card when it sees one" % (
        ", ".join("%s=%r" % (name, os.environ.get(name)) for name in REPORTED),
        ", ".join(sorted(SET_HERE)) or "none, the caller had them",
    )
