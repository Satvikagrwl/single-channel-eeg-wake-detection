# Reproducibility and Result-Traceability Audit

Audit date: 2026-10-13

## Checks performed on the repository files

- All four primary Python source files compiled successfully with Python's `py_compile` check.
- All four `experiment_summary.json` files parsed successfully.
- Prediction-table row counts matched the saved cohort sizes: 27,533 rows for EXP003, EXP004, and EXP005; 220,264 rows for EXP006 (27,533 rows × 8 scenarios).
- EXP006 contained 80 outer-fold/scenario records (10 held-out subjects × 8 scenarios).
- Pooled accuracy for EXP003 was recomputed from the saved predictions and matched its summary.
- Pooled accuracy, Wake sensitivity, and specificity for EXP004 were recomputed from its saved predictions and matched its summary.
- Pooled Wake sensitivity and specificity for EXP005 baseline and selected-threshold predictions matched its summary.
- Pooled Wake sensitivity and specificity for each of the eight EXP006 scenarios matched its summary.

The current `python code/validate_results.py` checks saved-file integrity and registry consistency. Its latest recorded run passed 45 checks with zero failures. The pooled metric comparisons listed above were checked separately; the current validator does not independently recompute all those metrics from prediction rows.

## Important limitations

1. These are internal consistency checks. They do not independently reproduce feature extraction and model training from raw EEG.
2. Raw Sleep-EDF files are not included. `code/download.py` requests recording 1 for subject indices 0–9, but complete acquisition and end-to-end processing have not been verified on a clean machine.
3. `code/run_pipeline.py` is configured for EXP006 and protects an existing result directory from overwrite. EXP003–EXP005 are represented by their saved results and experiment-specific code snapshots; the current active script does not recreate all four experiments with one command.
4. The current cleaned source scripts are not byte-identical to the saved experiment snapshots. The snapshots are retained as experiment-specific historical records. Differences between a snapshot and the current active script can include experiment configuration changes as well as later console-message cleanup; do not substitute the current EXP006 script for the historical code when interpreting earlier experiments.
5. The experiment registry includes EXP003–EXP006. Its metric fields are compared with the intended saved summary or primary scenario by the current validator.
6. Dependency pins in `requirements.txt` come from the saved experiment metadata. A fresh dependency installation and full end-to-end rerun were not performed as part of this audit.

## Result interpretation

The saved EXP005/EXP006 pooled specificity for the nominal 0.90 inner-validation specificity constraint is approximately 0.8941. The constraint is applied to inner validation and does not guarantee the same specificity on outer held-out predictions. Cost-ratio scenarios are assumed analytical weights, not measured clinical or device costs.
