# The evaluation

One evaluation for every row of the paper's tables. A **unit** is one row on one evaluation block; one command
runs it, and it leaves a config record and an episode table, and raw episodes beside them.

```
python -m evaluation.run --block <block> --row <row>
        |
        |-- model process, in the row's environment          evaluation/rows/<row>/adapter.py
        |        ^  observation, instruction   |  actions [n, 7]
        |        |  (local connection)         v
        '-- simulator process, in the block's environment    evaluation/
                 reads evaluation/blocks/<block>/block.json, runs the episodes, records, scores
```

- **A block** is a test set: a benchmark suite with all its tasks, start states, instructions, step limit and
  Reach target per task ([`blocks/README.md`](blocks/README.md)). It says nothing about any model.
- **A row** is a model as released, or a method on one: its checkpoint (address, revision, digest), its
  environment, and a thin adapter that turns the shared observation into what the release expects
  ([`rows/README.md`](rows/README.md)). The row `pi05_psg` calls `psg/policy.py`, as the front page's example does.
- **The interface** between the two processes is `interface.py`: images, robot state, camera parameters and the
  instruction go in; `actions [n, 7]` and how many to execute come out, with small arrays for the record (PSG:
  v, v_g, v_d at every Euler step). Object poses, segmentation and simulator state never cross it.
- **The config record** is written by the program from what the two processes loaded, defaults included: the
  benchmark and its file digests, the checkpoint's digest, every inference setting, the method's settings, the
  seeds, the labeller. Scientific settings only: no host, no path, no time.

## 1. The tables from `results/`

```
python -m evaluation.tables.render            # Tables 1-3; --table 2 for one; --out <dir> writes them as CSV
python -m evaluation.tables.render --check    # compares all 222 numbers with the tables as printed (tables/printed/)
```

Any Python >= 3.8, standard library, no card. `--results <directory>` reads units collected elsewhere. The rules
are data (`tables/tables.json`; [`tables/README.md`](tables/README.md)): a number is 100 × hits / episodes of a
unit, to one decimal, a half rounded up; an average is the mean of its suites' percentages, not a count over
their episodes. Seven LIBERO averages of Table 1 are cited from their authors (`tables/cited_libero.csv`).

## 2. Set up a machine

**For PSG alone** (the rows `pi05` and `pi05_psg` on the LIBERO and LIBERO-PRO blocks): two environments and
one checkpoint.

| What | How |
|---|---|
| `openpi` | upstream openpi at the pinned commit with its own `uv sync`; the commands are in `environments/openpi.json`. The evaluation installs nothing of this repository in it: `run.py` puts its checkout first on the path. The front page's editable install of `psg` is for the Quick start |
| `sim-libero-pro` | LIBERO-PRO's `requirements.txt`, then `imageio` and `imageio-ffmpeg`; its code (`github.com/Zxy-MLlab/LIBERO-PRO` at the block's revision) on the path; a `config.yaml` naming its `assets` and the `bddl_files` and `init_files` directories (`environments/sim-libero-pro.json`) |
| LIBERO-PRO's `_swap` and `_task` files | `huggingface.co/datasets/zhouxueyang/LIBERO-Pro` at the revision the blocks record, `bddl_files/` and `init_files/`, into the directories `config.yaml` names |
| the checkpoint | `gs://openpi-assets/checkpoints/pi05_libero` (11.58 GB): openpi downloads it on first use into `$OPENPI_DATA_HOME`, anonymously. It must be a real directory there, not a symbolic link |

and `environments.local.json` in the checkout (not in git), where the two environments are on this machine:

```json
{
 "openpi":         {"python": "<python>", "env": {"OPENPI_DATA_HOME": "<weights>/openpi-cache"}},
 "sim-libero-pro": {"python": "<python>", "env": {"LIBERO_CONFIG_PATH": "<dir with config.yaml>", "MUJOCO_GL": "egl",
                                                  "PYOPENGL_PLATFORM": "egl", "MUJOCO_EGL_DEVICE_ID": "0"}}
}
```

**For every row**, the same with all ten environments: the simulators `sim-libero-pro` (blocks `L-*`, `PP-*`,
`PT-*`, `OS-*`), `sim-libero-plus` (`PL`) and `sim-libero-cf` (`CF-*`), all three a package named `libero`; and
the model environments `openpi`, `openpi-cf`, `rss`, `openvla-oft`, `lerobot`, `turbovla`, `apt`. Each is
declared in `environments/<name>.json` (Python version, sources at full revisions, build commands, and for the
released baselines the weight downloads under `"weights"`; the revisions are in each row's `row.json`). An NVIDIA
card with EGL (24 GB ran every row), `git`, and about 140 GB of disk. No login, except for `turbovla`: a
Hugging Face token with access to the gated encoder `facebook/dinov3-vitb16-pretrain-lvd1689m`; runs need none
(`HF_HUB_OFFLINE=1`). An entry of the machine file may set any variable its environment needs, for example
`OPENPI_DATA_HOME`, `HF_HOME`, `PYTHONPATH`, `MAGICK_HOME` (LIBERO-Plus), `XLA_PYTHON_CLIENT_MEM_FRACTION`
(see section 6), and `PSG_ENVIRONMENTS` may name another file than `environments.local.json`
([`environments/README.md`](environments/README.md)).

**Check the machine**, without a card and without loading a model:

```
python3 -m evaluation.check_machine                       # every environment and row; --row pi05_psg --block PP-O for a part
```

One line per check (`ok` or `FAIL`): every environment's interpreter, every path its variables name, the
deterministic flags, every row's weights on disk at the recorded revision; the last line says how many failed.

## 3. Run a unit

From the checkout, which must be clean (commit or remove changes; `--smoke` allows them). A first try, a few
episodes (about 4 minutes, of which 3.5 are loading):

```
python -m evaluation.run --block PP-O --row pi05_psg --tasks 0 --states 0-3 --smoke
```

`--tasks` and `--states` pick a part of the block; `--smoke` writes under `outputs/_smoke/`, which is never
collected. A unit is the same line with the two names only. Several blocks after `--block`, separated by commas,
share one loaded model.

| To | Do |
|---|---|
| choose the card | `CUDA_VISIBLE_DEVICES=1 python -m evaluation.run ...`: the model runs there and the simulator renders there too (`run.py` sets `MUJOCO_EGL_DEVICE_ID` to that card). Where a machine's EGL numbers its cards in another order than CUDA, the simulator's entry in the machine file gives the table, `"egl_device_of_card": {"0": 2, "1": 0}`; `raw/logs/launch.<shard>.jsonl` says which card each process was given |
| write somewhere else | `--out <directory>`, or `PSG_OUTPUT=<directory>` (default `./outputs`) |
| split a unit over two cards | `--shard 0of2` on one, `--shard 1of2` on the other: each takes every second episode |
| resume an interrupted unit | the same command again: episodes already in the table are skipped, a failed one is run again |
| collect finished units | `python -m evaluation.collect outputs` (a Python >= 3.8 with NumPy) checks that every shard is there, that the shards' records agree, that the checkout was clean, and that every episode is there once and none failed; then writes `config.json` and `episodes.csv` beside each unit's `shards/`, where `render --results outputs` reads them. `--results results` writes into this checkout's `results/`, over the paper's units |

A checkout that is no git checkout (a ZIP download) runs too; its records say revision `unknown`, and `collect`
does not take them.

**What a unit writes**, under `outputs/<block>/<row>/`, the same layout on every machine:

| File | Content |
|---|---|
| `shards/config.<k>of<n>.json` | the config record |
| `shards/episodes.<k>of<n>.csv` | one line per episode, the same columns for every block and row: identity, instruction, success and its step, the first close and the first contact with what they were on, Reach, the labeller's revision, the trace's digest |
| `raw/traces/<episode>.npz` | per step: the gripper's pose and command, the pose of every body of the scene, fixtures included, every contact of the gripper; the method's arrays at every query |
| `raw/videos/<episode>.mp4`, `raw/logs/` | the main camera's video; the processes' logs |

**Reach** asks whether the hand went for the requested thing first. On a task that begins with a grasp (rule
`first_close`): when the gripper first closes, the object or fixture nearest the hand is the task's target. On a
task without a grasp, such as opening a drawer (`first_contact`): the first thing the gripper touches is the
target. `reach.py` reads the trace, the task's rule and its target, and nothing else.

## 4. Blocks and rows

| Block | Test set | Tasks × start states | Episodes | Step limit | Reach | Table |
|---|---|---|---|---|---|---|
| `L-S` `L-O` `L-G` `L-L` | LIBERO Spatial, Object, Goal, Long | 10 × 50 each | 500 each | 220 / 280 / 300 / 520 | no | 1 |
| `PP-O` `PP-G` `PP-S` | LIBERO-PRO Position (objects' places exchanged) | 10 × 50 each | 500 each | 280 / 300 / 220 | yes | 1 |
| `PT-O` `PT-G` `PT-S` | LIBERO-PRO Task (the instruction changed) | 10 × 50 each | 500 each | 280 / 300 / 220 | yes | 1 |
| `PL` | LIBERO-Plus Layout, every variant | 1,525 × 1 | 1,525 | its suite's | yes | 1 |
| `OS-O` | our scenes, Object: ten foods at six new places and the training place | 70 × start states 6-15 | 689 (11 start states left out, where the food did not end at its place) | 280 | yes | none |
| `OS-G` | our scenes, Goal: bowl, cream cheese, wine bottle at six new places and the training place | 21 × start states 6-15 | 210 | 300 | yes | 2 |
| `CF-S` `CF-O` `CF-OOD` | LIBERO-CF: Spatial, Object, OOD | 15 / 10 / 15 × 50 | 750 / 500 / 750 | 220 / 280 / 220 | yes | 3 |

One noise seed (7) in every block.

| Row | What it is | Environment | Tables |
|---|---|---|---|
| `pi05` | π0.5, the released LIBERO checkpoint | `openpi` | 1, 2, 3 |
| `pi05_psg` | π0.5 + PSG (ours), on the same checkpoint | `openpi` | 1, 2, 3 |
| `pi05_cag` | π0.5 + CAG, training-free, with its authors' code | `openpi-cf` | 1, 2, 3 |
| `pi05_cag_va` | π0.5 + CAG with its authors' extra vision-action model | `openpi-cf` | 3 |
| `pi05_rss` | π0.5 + RSS, its authors' fine-tune and code | `rss` | 1, 2 |
| `pi0` | π0, released LIBERO checkpoint | `openpi` | 1, 2 |
| `pi0_fast` | π0-FAST, released LIBERO checkpoint | `openpi` | 1, 2 |
| `openvla_oft` | OpenVLA-OFT, released checkpoint for the four suites | `openvla-oft` | 1, 2 |
| `smolvla_exec10` | SmolVLA, released LIBERO checkpoint, 10 actions executed per query | `lerobot` | 1, 2 |
| `turbovla` | TurboVLA, released checkpoint | `turbovla` | 1, 2 |
| `apt` | APT, its authors' model and code | `apt` | 1, 2 |

The rows that belong to no table, the instruments' (the pathway cuts, PSG at other weights, the oracle push), are
in [`rows/README.md`](rows/README.md) and [`instruments/README.md`](../instruments/README.md).

## 5. Reproduce Tables 1-3

**Table 1** (ten rows; 82 units). The LIBERO column is run for three rows; for the other seven the table cites the
authors' averages. SmolVLA's LIBERO blocks were also run here (4 more units; 83.8 on average, in
`results/L-*/smolvla_exec10/`), but the table cites its authors' 87.3.

```
for ROW in pi05 pi05_psg pi05_cag; do
  python -m evaluation.run --block L-S,L-O,L-G,L-L --row $ROW
done
for ROW in pi05 pi05_psg pi05_cag pi05_rss pi0 pi0_fast openvla_oft smolvla_exec10 turbovla apt; do
  python -m evaluation.run --block PP-O,PP-G,PP-S --row $ROW
  python -m evaluation.run --block PT-O,PT-G,PT-S --row $ROW
  python -m evaluation.run --block PL --row $ROW
done
```

**Table 2** (the same ten rows on the Goal scenes; 10 units). It prints the 18 scenes where the object was moved,
in three groups by how far: below 20 cm, 20 to 30 cm, above 30 cm (4, 4 and 10 scenes; `tables/os_g_scenes.csv`).
The same rows were also run on `OS-O` (10 units), which no table prints.

```
for ROW in pi05 pi05_psg pi05_cag pi05_rss pi0 pi0_fast openvla_oft smolvla_exec10 turbovla apt; do
  python -m evaluation.run --block OS-G --row $ROW
done
```

**Table 3** (four rows; 12 units)

```
for ROW in pi05 pi05_psg pi05_cag pi05_cag_va; do
  python -m evaluation.run --block CF-S,CF-O,CF-OOD --row $ROW
done
```

Then `python -m evaluation.collect outputs` and `python -m evaluation.tables.render --results outputs`.

## 6. What repeats, and what it costs

What makes an episode the same wherever it runs: the start state; the scene's placement, checked at every reset
against the scene's first and rebuilt when it differs; the environment seed (7) and the block's `rollout`; and
the noise of a query, a function of (noise seed, task, start state, query index) alone
(`interface.noise_generator`). The openpi-based rows run with deterministic GPU operations
(`XLA_FLAGS=--xla_gpu_deterministic_ops=true`, `TF_CUDNN_DETERMINISTIC=1`, from `environments/openpi.json`):
with them two launches of the same episodes on one card gave identical traces, without them they did not.

That makes an episode repeat on one machine. **On another machine the same unit is the same definition, not the
same bits**: `pi05_psg` gave other traces on two machines in 3 of 3 episodes. The tables' rows ran on three
machines; a number run again elsewhere can differ in its last episodes, and two rows compared episode by episode
mean what they say only when both ran on one machine.

How long a unit takes on one 24 GB card (measured on a handful of episodes before the tables were run; every
episode counted to the step limit, so these are ceilings; add 3.5 minutes of loading per launch):

| Row | 500 episodes of Object / Goal / Spatial / Long |
|---|---|
| `pi05` | 1.5 / 1.5 / 1.4 / 2.8 h |
| `pi05_psg` | 2.9 / 2.4 / 2.1 / 3.8 h |
| `pi0` | 2.0 h (Object) |
| `pi0_fast` | 5.1 h (Object; three episodes measured) |
| `pi05_cag` | 1.9 h (Object) |

One `pi05_psg` unit needs 8.7 GB of the card; two units fit on 24 GB with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.45`
in the environment's entry. `pi05_cag_va` holds two π0.5 models and needs the whole card (`0.9` in the
`openpi-cf` entry). Per episode on disk: about 0.35 MB.

## 7. Tests

From the repository's root:

```
python -m pytest tests/                       # any Python >= 3.8 with NumPy and pytest; no card, no weights
PSG_HOST_TESTS=1 python -m pytest tests/      # the openpi environment's Python: also the samplers on small random models
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PSG_HOST_TESTS=1 python -m pytest tests/   # the same, held to the CPU
```

In the first line, three tests of `tests/rows/test_method_rows.py` are skipped where SciPy or JAX is not
installed. The host tests run against the openpi the environment has installed; `tests/conftest.py` sets for
them the deterministic settings of `environments/openpi.json`, as a unit gets them.
