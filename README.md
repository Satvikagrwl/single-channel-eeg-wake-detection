# Single-Channel EEG Wake Detection

Research code and saved experiment outputs for evaluating Wake-versus-Non-Wake classification from a single frontal EEG channel using relative spectral bandpower features and Random Forest classifiers.

> **Status:** Research release candidate. The included result files have been checked for internal consistency against the saved prediction tables, but the complete analyses have not been independently rerun from a fresh environment as part of this repository audit. This is not a clinical system and is not validated for real-time or closed-loop deployment.

## Research question

Can a compact six-feature representation derived from the frontal Fpz-Cz EEG channel support subject-independent Wake-versus-Non-Wake classification, and how do nested threshold-selection rules change the trade-off between Wake sensitivity and specificity?

The project evaluates conventional fixed-threshold classification, nested subject-independent threshold selection, prespecified specificity constraints, and illustrative false-negative/false-positive cost ratios. The cost ratios are analytical scenarios, not empirically measured clinical or device costs.

## Dataset

The experiments use the **Sleep-EDF Expanded** dataset hosted by PhysioNet:

- Dataset page: <https://physionet.org/content/sleep-edfx/1.0.0/>
- EEG channel: `EEG Fpz-Cz`
- Cohort represented in saved outputs: subjects 1–10, recording 1
- Epoch duration: 30 seconds
- Retained epochs in each saved cohort-level prediction file: 27,533

Raw EEG is not included in this repository. Obtain the dataset from PhysioNet and follow its documentation and terms. The feature extractor expects EDF files under `~/mne_data/physionet-sleep-data/` with the filenames used in `code/feature_extractor.py`.

**Dataset-download note:** `code/download.py` requests recording 1 for all ten subjects using MNE's subject indices 0–9. Raw EEG is not bundled. After downloading, verify that the required files are present in the expected cache before running the feature extractor.

## Preprocessing and features

The current feature extractor documents and implements the following processing choices:

1. Load only `EEG Fpz-Cz`.
2. Divide the recording into consecutive, non-overlapping 30-second epochs.
3. Assign the annotation by the epoch midpoint.
4. Map `W` to Wake; `1` to N1; `2` to N2; `3` and `4` to N3; and `R` to REM.
5. Exclude unsupported, movement (`M`), unscored (`?`), and unmatched annotations rather than assigning them to Wake.
6. Compute six relative spectral bandpower features using Welch PSD with 4-second segments and 50% overlap:

| Feature | Frequency band |
|---|---:|
| Delta | 0.5–4 Hz |
| Theta | 4–8 Hz |
| Alpha | 8–12 Hz |
| Sigma | 12–16 Hz |
| Beta | 16–30 Hz |
| Gamma | 30–40 Hz |

The binary task used in EXP004–EXP006 defines Wake as the positive class and all retained non-Wake sleep stages as the negative class.

## Experiments and saved outputs

Each experiment directory contains an `experiment_summary.json`, prediction and/or fold-level tables, and a code snapshot. Inspect those files together when interpreting an experiment.

| Experiment | Purpose | Main saved result |
|---|---|---|
| `EXP003_corrected_baseline` | Corrected five-class sleep-stage baseline with unsupported labels excluded | 27,533 epochs; 10-subject LOSO; pooled accuracy 87.40%; mean fold accuracy 87.35% |
| `EXP004_wake_binary_baseline` | Binary Wake vs Non-Wake baseline at threshold 0.50 | Pooled Wake sensitivity 94.11%; pooled specificity 90.29%; pooled Wake miss rate 5.89% |
| `EXP005_nested_wake_threshold` | Select threshold within the outer training subjects using inner LOSO, with mean inner specificity constrained to at least 0.90 | Mean selected threshold 0.44; pooled Wake sensitivity 95.37%; pooled specificity 89.41%; pooled Wake miss rate 4.63% |
| `EXP006_operating_point_and_cost_analysis` | Compare specificity floors (0.85, 0.90, 0.95) and Wake-FN:Non-Wake-FP cost ratios (1:1, 2:1, 5:1, 10:1) | Eight analysis scenarios plus the fixed 0.50 baseline; 80 outer-fold/scenario rows |

The 0.90 specificity floor is enforced on inner validation data. It does **not** guarantee that pooled or per-subject outer-test specificity will reach 0.90. In the saved EXP005/EXP006 outputs, pooled specificity for the 0.90-floor scenario is approximately 89.41%.

The cost ratios are assumed analytical weights. They should not be described as measured real-world costs, clinical risk estimates, or validated device requirements.

## Repository layout

```text
Single-Channel-EEG-Wake-Detection/
├── README.md
├── .gitignore
├── requirements.txt
├── RESEARCH_LOG.md
├── REPRODUCIBILITY_AUDIT.md
├── code/
│   ├── download.py
│   ├── feature_extractor.py
│   ├── model_trainer.py
│   ├── run_pipeline.py
│   └── validate_results.py
└── results/
    ├── experiment_registry.csv
    ├── EXP003_corrected_baseline/
    ├── EXP004_wake_binary_baseline/
    ├── EXP005_nested_wake_threshold/
    └── EXP006_operating_point_and_cost_analysis/
```

The manuscript, figures, historical archive folders, raw EEG files, and local virtual environments are intentionally excluded from this code/results repository. Historical experiment-specific code snapshots are retained inside the result directories.

## Environment setup

The experiment summaries record the following software versions for the original runs: Python 3.14.6, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, MNE 1.13.2, and YASA 0.7.0. `requirements.txt` pins those recorded direct dependencies. These direct dependency pins were installed in the repository's virtual environment; library imports and `pip check` passed. A complete end-to-end rerun from raw EEG was not performed as part of this audit.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`.

## Checking saved result consistency

Run the standard-library validation utility from the repository root:

```bash
python code/validate_results.py
```

It checks JSON parsing, nonempty prediction CSVs, experiment identifiers, and selected registry metrics against the saved summaries. It does **not** independently recompute all prediction metrics or retrain models.

## Running the analysis code

The current `code/run_pipeline.py` is configured for **EXP006**. It uses the cohort data path described above and deliberately refuses to overwrite an existing experiment output directory. The included `results/EXP006_operating_point_and_cost_analysis/` directory already exists in this release, so running the script unchanged is expected to stop at the overwrite-protection check rather than regenerate those saved outputs.

For a new run, first preserve this release unchanged, make a separate working copy, ensure the required raw EEG files are available, and configure a new experiment ID/name and output location in that working copy. Do not disable overwrite protection to replace recorded results. The saved EXP003–EXP005 snapshots document earlier experiment configurations; the current main pipeline is not a single command that recreates all four historical experiments.

`code/model_trainer.py` is a legacy standalone five-class random 70/30 split trainer. It is not the nested LOSO evaluation pipeline used for EXP004–EXP006 and should not be treated as an alternative implementation of those experiments.

## Reproducibility and interpretation limits

- Saved predictions and summary metrics can be checked for internal consistency with `code/validate_results.py`.
- A consistency check is not the same as an independent rerun from raw data.
- The raw dataset is not bundled. The downloader requests all ten subjects, but a complete end-to-end acquisition and analysis run has not been verified on a clean machine.
- Runtime and model-size measurements in EXP006 were collected on the recorded workstation environment; they do not establish performance on wearable hardware.
- Results are from a ten-subject cohort and should not be generalized beyond the evaluation design without further validation.
- No clinical validation, safety validation, or real-time/closed-loop device validation is claimed.

## License

The MIT license in `LICENSE` applies to original project code. Third-party datasets and materials remain subject to their own terms.
