# Submission evidence

The submitted results come from the user-supplied `e0_e4_logs_evidence.zip`.
`IMPORT_MANIFEST.json` records its SHA-256, prediction hashes, original run IDs,
and checks against the original instructor scorer. The source ZIP and extracted
predictions stay local; no dataset, credential or model checkpoint is submitted.

| Original evidence name | Submitted evidence name | Actual setup |
|---|---|---|
| e0.json–e3.json | evidence/e0.json–e3.json | E0–E3 |
| e4_rank_4_base_e1.json | evidence/e4_rank_4.json | E1, rank 4, alpha 8 |
| e4_rank_64_base_e1.json | evidence/e4_rank_64.json | E1, rank 64, alpha 128 |

Only file and figure-directory names were normalized. Evidence `name` fields
and original log `run_id` fields retain the actual E1 baseline. E3 has the
highest observed F1, so the supplied E4 pair does not meet GUIDE's requirement
to ablate the best setup. The two E4 manifests otherwise differ only in rank
and its required alpha adjustment.

`logs/<submitted-name>/` contains the unmodified, small `run.json`,
`metrics.jsonl` files for each trained run. Each metrics file
contains steps 1–300 exactly once, with finite losses and gradient norms.
Evidence resource metadata matches the corresponding manifest; peak VRAM
matches the largest value in the step logs. There is no hardware inventory in
the ZIP. E1/E2 manifest start/end timestamps were rewritten after training and
do not describe training duration; the report uses measured elapsed minutes.
E3/E4 manifest durations include work after the last logged optimizer step.

To reproduce with the source ZIP extracted to `work/submission-import/`:

```powershell
python scripts/prepare_data.py --source data --out work/submission-data
python scripts/score.py --pred work/submission-import/runs/e3/predictions_test.jsonl --splits-dir work/submission-data/splits --name E3 --out work/rechecked_e3.json --meta trainable_params=148426752 --meta peak_vram_gb=10.177475929260254 --meta train_minutes=16.9951998789
python scripts/score.py --pred work/submission-import/runs/e3/predictions_test.jsonl --splits-dir work/submission-data/splits --coord pixel --name E3-wrong-coordinate-control --out work/rechecked_wrong_coord.json
python -m pytest -q tests student_tests --basetemp work/pytest-submission
python scripts/check_submission.py
```

The local grouped split reproduces `d10-21e4070301`. Rescored results match all
six supplied evidence files within 1e-12 floating-point tolerance (summation
order can slightly change mean IoU). `analysis/e3_wrong_coord_pixel.json` is
an intentional scoring control using the same E3 outputs with the wrong
coordinate interpretation; it is separate from the six experiment results.
The report includes six E3 images with per-image TP/FP/FN descriptions; the
six required E3 GT/Pred illustrations are retained in `evidence/figures/`.

Validation on 2026-10-14: full suite 118 passed, 4 skipped, 1 failed. The original test_report_template_has_student_markers_and_ten_questions requires the unfilled name marker, while the submission checker forbids it. Instructor tests are unchanged; excluding only this contradictory template test gives 118 passed, 4 skipped, 1 deselected. Submission checker:
0 FAIL, 0 WARN with locally regenerated split metadata. A fresh clone without
local dataset splits will report the expected metadata comparison warning.
Real remote recovery after a session reset, hardware comparability and E5
training are not demonstrated by this archive.
