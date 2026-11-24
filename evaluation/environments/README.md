# Environments

The models and the simulators do not share a Python environment. Each one has a name here and is built once per
machine; `run.py` finds it by name.

- `evaluation/environments/<name>.json` (in git): what the environment is. `python`, `sources` (each repository installed in
  it, with address and full revision), `build` (the commands), `check` (one command that proves it imports).
- `environments.local.json` (not in git, one per machine): where each one is, `{"<name>": "<its python>"}`.
  `$PSG_ENVIRONMENTS` may point to another file. An entry may also be
  `{"python": "<its python>", "env": {"NAME": "value"}}` for the variables only a machine knows, and a
  simulator's entry may carry `"egl_device_of_card": {"<card>": <EGL device>}` where the machine's EGL numbers its
  cards in another order than CUDA does: a launch's simulator renders on the launch's own card
  (`evaluation/layout.py`, `simulator_environment`).

A simulator environment holds one benchmark's code (`sim-libero-pro`, `sim-libero-plus`, `sim-libero-cf`: the
three forks are all a package named `libero`), NumPy, and a video encoder (`imageio` with `imageio-ffmpeg`). A
model environment holds one release's code and NumPy. Neither needs anything of this repository installed:
`run.py` puts the checkout on `PYTHONPATH`, and the shared code under `evaluation/` runs on Python 3.8
and later with the standard library and NumPy.

Nothing is built by this repository's code, and no file here names a host.
