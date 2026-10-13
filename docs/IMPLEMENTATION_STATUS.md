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
| CP03 | Pending | `student/model.py`, `student_tests/test_model_targets.py`; NF4 exclusions, real-module discovery assertions, LoRA setup, FP32 projector island/state | CPU target-pattern check: 83 passed, 1 skipped overall; actual model load and GPU dtype/gradient smoke **NOT_RUN** | Code complete; hardware check pending |
| CP04 | Pending | Resumable JSONL inference and canonical E0 scoring | Kaggle GPU | NOT_RUN |
| CP05 | Pending | Per-step logs, complete atomic checkpoints, local and HF recovery, fault-injection tests | CPU checkpoint tests; Kaggle remote recovery | Pending |
| CP06 | Pending | E1/E2/E3 inference, scorer evidence, metrics and annotated examples | Kaggle GPU | NOT_RUN |
| CP07 | Pending | E4 rank 4/64 runs with controlled settings and evidence | Kaggle GPU | NOT_RUN |
| CP08 | Pending | Pure Python IoU reward and unit tests; GRPO if resources permit | CPU reward tests; Kaggle GRPO | Pending |
| CP09 | Pending | Preserve REPORT format; final test and submission checker review | CPU plus real Kaggle evidence | Pending |

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
| E0/E1/E2/E3/E4 inference | `student/inference.py` and `student/experiments.py` | Incremental predictions; score only complete test runs |
| Durable training resume | `student/checkpointing.py`, `student/training.py` | Local fault injection; remote Kaggle kill/reboot test required |
| E5 reward | `student/reward.py` | Five required reward unit cases |
| Vietnamese evidence-backed report | `REPORT.md` | Keep unanswered GPU-derived fields blank / NOT_RUN until actual evidence exists |
| Kaggle run instructions | `docs/KAGGLE_RUNBOOK.md` | Manual consistency review against notebook and secrets setup |

