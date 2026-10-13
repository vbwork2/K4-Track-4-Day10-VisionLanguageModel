# Day 10 implementation status

## Audit

- Base revision: `c2f3997` (`main` at audit time); implementation branch: `feat/day10-kaggle-qlora`.
- The supplied `INSTRUCTION_CODEX_DAY10_KAGGLE.md` and `kaggle_day10_qlora_resume.ipynb` were initially untracked user-provided files and are retained as implementation inputs.
- `README.md`, `GUIDE.md`, `RUBRIC.md`, `REPORT.md`, `configs/e0.yaml` through `configs/e5.yaml`, and the original 22-cell `lab_day10.ipynb` were inspected.
- The original `scripts/` and `tests/` files are protected. Their SHA-256 values were recorded before edits; implementation changes must not touch them.
- Dataset files are present locally: 2,009 images and 2,009 YOLO label files. No generated split metadata existed at audit time.
- Runtime: Windows 11, Python 3.11. Pillow 12.3.0 was installed to satisfy the original test dependency. PyTorch, Transformers, PEFT, Accelerate, bitsandbytes, Kaggle CLI, and Hugging Face Hub are absent. The visible GPU is an RTX 3050 with 4 GB VRAM, not the required Kaggle T4 16 GB. Kaggle and HF credential names were unset; no credential values were read.

## Baseline checks

- Initial `python -m pytest -q`: collection failed because Pillow was missing.
- After installing Pillow, the default pytest temp directory was inaccessible under the sandbox (`PermissionError` in `%LOCALAPPDATA%/Temp`).
- `python -m pytest -q --basetemp .pytest-tmp`: **68 passed, 1 skipped**. The workspace-local temp base is required in this environment.
- `nvidia-smi`: RTX 3050 4 GB detected. Model training and Qwen processor/model API checks are not possible in this runtime.
- Kaggle GPU training, Hugging Face remote backup, crash/reboot resume, and experiment evidence: **NOT_RUN**.

## Implementation plan and evidence policy

The `student/` package will own the sole implementation of label conversion, chat encoding, model setup, training, checkpoint recovery, inference, scoring orchestration, and optional reward. Both notebooks will call this package. The supplied Kaggle notebook remains the Kaggle entry point; `lab_day10.ipynb` keeps its original cell and section structure. The instructor's scripts, tests, split protocol, and metric implementation remain unchanged.

Evidence JSON and report measurements will be produced only by `scripts/score.py` from complete real predictions. CPU checks may validate deterministic transformations, checkpoint serialization, and notebook/package wiring. Results requiring the Qwen model, Kaggle GPU, Kaggle Secrets, or Hugging Face private repository will stay **NOT_RUN** until run in that environment.

## Checkpoint ledger

| Checkpoint | Commit | Files / checks | GPU / CPU | Status |
|---|---|---|---|---|
| CP00 | `96addb7` | `docs/IMPLEMENTATION_STATUS.md`; baseline `python -m pytest -q --basetemp .pytest-tmp` | CPU: 68 passed, 1 skipped | Complete |
| CP01 | `2a3e4a5` | Six files; `git show --stat`: 253 insertions, 4 deletions; grouped split and six local overlays | CPU: 75 passed, 1 skipped; split metadata and all 2,009 image paths validated | Complete |
| CP02 | `e692485` | Three files; `git show --stat`: 220 insertions, 3 deletions; prefix/mask/pad/image-token helpers | CPU: 80 passed, 1 skipped; Transformers/PyTorch processor integration **NOT_RUN** because those packages are absent | Complete |
| CP03 | `24c176e` | Three files; `git show --stat`: 322 insertions, 3 deletions; NF4 exclusions, real-module discovery assertions, LoRA setup, FP32 projector island/state | CPU target-pattern check; actual model load and GPU dtype/gradient smoke **NOT_RUN** | Code complete; hardware check pending |
| CP04 | `9c33a8e` | Three files; `git show --stat`: 203 insertions, 3 deletions; incremental JSONL, complete-set guard, original scorer invocation | CPU: 86 passed, 1 skipped overall; generation and evidence **NOT_RUN** | Code complete; GPU evidence pending |
| CP05 | `61dd35c` | Five files; `git show --stat`: 1,044 insertions, 3 deletions; atomic step directories, fingerprints, optimizer/scheduler/scaler/RNG state, metrics recovery, private-HF upload/download | CPU structural checks: 99 passed, 1 skipped; PyTorch save/load and Kaggle hard-reset remote recovery **NOT_RUN** | Code complete; durability verification pending |
| CP06 | `99057b0` | Seven files; `git show --stat`: 443 insertions, 5 deletions; E1/E2/E3 load/train/infer/score orchestration and annotated GT/Pred examples | CPU: 109 passed, 1 skipped; Kaggle predictions, evidence JSON, and six prediction overlays per setup **NOT_RUN** | Code complete; GPU evidence pending |
| CP07 | `6e25023` | Three files; `git show --stat`: 158 insertions, 3 deletions; fixed-control rank pair and evidence comparison | CPU: 111 passed, 1 skipped; both Kaggle rank runs **NOT_RUN** | Code complete; GPU evidence pending |
| CP08 | `98dae8f` | Three files; `git show --stat`: 111 insertions, 3 deletions; class-aware greedy IoU reward, record conversion, six reward/conversion cases | CPU: 112 passed, 1 skipped; GRPO **NOT_RUN** because local runtime has no PyTorch/CUDA or Kaggle session | Reward complete; optional GRPO pending |
| CP09 | `60ec5da` | 12 files; `git show --stat`: 1,000 insertions, 54 deletions; both notebooks, Kaggle runbook, truthful REPORT, split metadata validation, checkpoint integrity fix, final static/CPU checks | CPU: 113 passed, 2 skipped; compileall and both notebook JSON/Python-cell checks pass; official checker: 6 FAIL, 1 WARN due unrun Kaggle evidence, absent `data/splits/meta.json`, and unfilled name; GPU **NOT_RUN** | Code and notebook ready; experiment evidence pending |

## Requirement checklist

| Requirement | Implementation location | Verification / evidence |
|---|---|---|
| Do not modify instructor scripts, tests, README, GUIDE, RUBRIC, or configs | Existing files | Compare tracked diffs and pre-edit hashes |
| Keep original lab notebook structure | `lab_day10.ipynb` | Preserve 22 cells and all nine numbered sections; validate notebook JSON |
| Use one synchronized pipeline | `student/`, both notebooks | Both entry points import the same package; notebook static checks |
| Grouped split and dataset identity | Original `scripts/prepare_data.py`, `student/data.py` | `d10-21e4070301`; 18 sessions; train 1,422 / val 276 / test 311; class counts match README; all sessions remain in one split |
| Convert YOLO boxes to JSON coordinates | `student/labels.py` | CPU tests cover norm1000/pixel, pixel conversion, clipping, empty labels, stable order, and invalid input |
| Mask prompt and vision tokens | `student/dataset.py` | CPU tests cover prompt prefix, assistant-only labels, padding, and image token counting; image patch/grid concatenation is implemented but awaits a Qwen processor batch smoke test |
| QLoRA E1/E2/E3 variants | `student/model.py` | Synthetic module-tree tests assert 252 LLM and 96 ViT targets; runtime code asserts loaded model counts and quantization exclusions. Real trainable counts, dtype, and gradient flow remain unverified without the pinned model stack on Kaggle. |
| E0/E1/E2/E3/E4 inference | `student/experiments.py`, `student/inference.py`, `student/visualization.py` | CPU tests verify resume/ordering, complete-set scoring guard, and overlay rendering; E1–E4 results and test-set illustrations remain **NOT_RUN** |
| E4 single-variable ablation | `student/experiments.py` | Rank 4 and 64 use the same chosen baseline and controls; evidence comparison rejects incomplete or cross-dataset scorer files. Actual comparison is **NOT_RUN** until both Kaggle runs finish. |
| Durable training resume | `student/checkpointing.py`, `student/training.py` | CPU tests verify corrupt/incomplete checks, dataset/setup/config mismatches, state restoration order, monotonic logs, and missing-secret failure. Actual PyTorch checkpoint write/read and Kaggle private-HF upload → session reset → download → resume still require Kaggle credentials and runtime. |
| E5 reward | `student/reward.py` | CPU tests cover perfect match, malformed output, wrong class, hallucinated box, both empty, and split-record YOLO conversion. GRPO training is optional and remains **NOT_RUN**. |
| Vietnamese evidence-backed report | `REPORT.md` | Keep unanswered GPU-derived fields blank / NOT_RUN until actual evidence exists |
| Kaggle run instructions | `docs/KAGGLE_RUNBOOK.md` | Branch import, T4/Internet, Add Input, secrets, checkpoints, evidence paths, E4 pair, and unverified remote hard-reset boundary documented |

