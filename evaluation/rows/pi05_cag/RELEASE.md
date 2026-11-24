# π0.5 + CAG: the adapter against the release

Rows `pi05_cag` (training-free) and `pi05_cag_va` (with the vision-action model). Release: `github.com/yuffish/LIBERO-CF`
at `8280a500a01f7f7d45efe202cb416ca362aa8c1a` (MIT); the two server files are pinned by git blob
(`serve_policy_TF.py` `705c7ef5…`, `serve_policy_VA.py` `7f25d6f6…`) and the adapter refuses another.
Paper: CAG, arXiv 2602.17659. Written from reading the release, before the row was run.

| What | The release | Where (file, lines) | The adapter |
|---|---|---|---|
| Image preprocessing | both cameras rotated 180° (`[::-1, ::-1]`), `resize_with_pad` to 224, `convert_to_uint8` | `eval/main_cf.py` L201-210 | `openpi_libero.libero_request`, with openpi-client's own functions |
| State sent | eef_pos (3), quat→axis-angle (3), gripper_qpos (2) | `eval/main_cf.py` L222-229, L503-521 | same |
| State read by the model | no: `pi05_libero` has `discrete_state_input=False` | `openpi-cf/src/openpi/training/config.py` (`pi05_libero`) | reported |
| Prompt | the instruction; under LIBERO-CF the file-name (counterfactual) one | `eval/main_cf.py` L230, L441; `libero/libero/benchmark/__init__.py` L44-53 | the block's instruction |
| Actions per query | 5 of the 10-action chunk | `eval/main_cf.py` L38, L237-241 | `execute = 5` |
| Sampling | openpi Euler, 10 steps (policy default; no `sample_kwargs`) | `openpi-cf/src/openpi/models/pi0.py` `sample_actions`; `policies/policy_config.py` | reported from the loaded policy |
| TF: unconditioned input | the conditioned input with `tokenized_prompt` and `tokenized_prompt_mask` zeroed | `scripts/serve_policy_TF.py` L113-119, L148-151 | the authors' class |
| TF: unconditioned model | the same policy | `serve_policy_TF.py` L167-168 | the authors' class |
| VA: unconditioned model | config `pi05_libero_unconditioned` (`mask_language_tokens=True`), its own checkpoint | `serve_policy_VA.py` L228-238; `openpi-cf/src/openpi/training/config.py` L787-812; `models/pi0.py` (fork) `embed_prefix` | HF `VLA-CF/pi05-libero-VA` @ `7b7b6c22…` through the hub cache |
| VA: inputs | each policy's own transform; the unconditioned one then zeroed | `serve_policy_VA.py` L175-180 | the authors' class |
| Noise | one tensor for both samplings | TF L159-161; VA L188-192 | `query_noise(seed, query)` passed to `infer(..., noise=)` (TF L146, VA L174) |
| Mixing | a = a_u + ω (a_c − a_u), once, on the final chunk, 32 dims, normalised space | TF L170-172; VA L201-203 | the authors' class |
| Output | `LiberoOutputs`: un-normalise, first 7 dims | TF L174-178; VA L205-209; `policies/libero_policy.py` L100 | the authors' class |
| ω | paper 1.5 (TF and VA); README 2.0 for TF, 1.5 for VA | paper `main.tex` L609-611, `supl.tex` L192; `README.md` L274, L283, L308 | 1.5 for both (the paper's value) |
| The release's own noise | `jax.random` key split once per query in server order; never seeded per episode | TF L157-159; VA L121, L188 | replaced by the query noise (stated in the record) |
| Model code vs upstream openpi | the fork changes `pi0.py`, `pi0_config.py` only by opt-in masking fields (default off), `transforms.InjectDefaultPrompt` by two options (default off), `LiberoOutputs` by `[:, :7]` for `[..., :7]` | diff of the fork against openpi `upstream/main` (`215abfb`), read 2026-11-12 | none |

The VA checkpoint loads into `pi05_libero_unconditioned` and runs: the units in `results/CF-*/pi05_cag_va/`. Not
compared: whether its `assets/libero/norm_stats.json` (1,914 bytes on the hub) equals openpi's `pi05_libero`
statistics (also 1,914 bytes).
