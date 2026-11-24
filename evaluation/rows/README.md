# Rows: one directory per row, of the paper's tables and of the instruments

`evaluation/rows/<row>/` holds what belongs to one row and to nothing else: which checkpoint, which code, in which
environment, and the adapter that puts the release behind the shared interface.

Rows: `openvla_oft` `smolvla_exec10` `turbovla` `pi0_fast` `pi0` `pi05` · `pi05_rss` `apt` · `pi05_cag` `pi05_cag_va`
`pi05_psg`.

`smolvla_exec10` runs the SmolVLA checkpoint with 10 actions executed per query (`"executed_per_query"` in row.json),
where the release's default is 50.

Beside them, belonging to no table: `pi05_psg_a5_b0` and `pi05_psg_a0_b1`, the `pi05_psg` adapter with other weights of PSG's two terms,
stated in `row.json` (`"weights"`; read by `evaluation/rows/pi05_psg/adapter.py`): the single-term arms of the paper's Fig. 5 (b).
And `pi05_oracle_push`, an instrument for the paper's Fig. 7 (`instruments/oracle_push.py`):
the one row that reads object poses, through the oracle channel its `row.json` asks for (`"oracle"`); `"push"` there
holds its size (1.164 executed-action units since 2026-11-14).
And `pi05_cut_none`, `pi05_cut_grounding`, `pi05_cut_direct` and the same three of `pi0`, instruments for the
paper's pathway cuts (`instruments/pathway_cuts.py`): the released checkpoint with one of the instruction's
two ways to the action cut, the arm stated in `row.json` (`"cut"`).

## `row.json`: what the entry needs before the model is loaded

```json
{
  "row": "openvla_oft",
  "name": "OpenVLA-OFT",
  "kind": "a released LIBERO checkpoint, run as released",
  "environment": "openvla-oft",
  "checkpoint": {"address": "huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10", "revision": "<full commit>", "licence": "MIT"},
  "code": {"address": "github.com/moojink/openvla-oft", "revision": "<full commit>", "licence": "MIT"},
  "method": "none"
}
```

`environment` names the Python environment the model runs in (`evaluation/environments/<name>.json` says how it is built:
Python version, the repositories and revisions installed, the commands). The entry finds that environment's
interpreter by name on each machine; a row never names a host or a path. Rows may share an environment.

## `adapter.py`: the release behind the interface

```python
from evaluation.interface import Actions, Adapter, Episode, Observation

class OpenVlaOft(Adapter):
    def report(self) -> dict: ...                       # the row's side of the config record
    def begin_episode(self, episode: Episode) -> None: ...
    def act(self, observation: Observation) -> Actions: ...

def load(row: dict) -> Adapter:                         # `row` is row.json
    ...
```

The interface is `evaluation/interface.py`; read its docstrings, they are the contract. In short:

- **In, at every query:** the two camera images exactly as the simulator returns them (uint8, the row's render
  size, not flipped), the raw robot state (end-effector position and quaternion, gripper and arm joints), each
  camera's intrinsics and extrinsics, the arm controller's action scale, the instruction text, the step and
  query index. Before the first query of an episode: the episode's name, task, the LIBERO suite its scene comes
  from, start state and noise seed.
- **Out:** a chunk of actions `[n, 7]` as `env.step` takes them (translation, axis-angle rotation, gripper with
  -1 open and +1 close) and `execute`, how many of them to run before the next query. A method may add small
  arrays for the record (`Actions.record`).
- **The adapter's own, all of it:** rotating, resizing, cropping, padding; whether and how the state is read; the
  prompt template; normalisation; the gripper's sign and binarisation; action ensembling (an adapter that
  ensembles returns one action per query with `execute = 1`); the number of sampling steps; the chunk length;
  what the noise seed controls.
- **The release's settings, as released.** Use the release's own inference code and its own values. A value that
  the shared side fixes for every row (step limit, settling steps, stop rule, environment seed, start states)
  is not the adapter's to set, even where the release's script differs.
- **`report()` is read from what was loaded**, not copied from `row.json`: the files' digests, the settings the
  objects actually hold, the installed packages' versions. A default nobody set still appears in it.
- No simulator in the adapter: it never imports `libero` or `robosuite`.

`python -m evaluation.rows.serve --row <row> --port-file <file>` is how the entry starts an adapter; nobody runs it
by hand.

## Tests of an adapter

A test of an adapter needs no weights: the transforms on a synthetic observation (shapes, the 180-degree
rotation, the gripper's sign, the action queue), in `tests/rows/test_<row>.py`. What needs the weights or a simulator is shown
by running the row: `python -m evaluation.run --block <block> --row <row> --tasks 0 --states 0 --smoke`.
