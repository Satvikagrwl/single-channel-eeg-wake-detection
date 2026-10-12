# Research Log

## Experiment 001 — Original Pilot

**Date:** 2026-09-15

**Dataset:**
Sleep-EDF, Subject 00 / Recording 1

**Pipeline:**

* Single-channel EEG Fpz-Cz
* 30-second epochs
* Welch PSD
* Random Forest, 100 trees
* Random 70/30 split

**Result:**
71% accuracy

**Purpose:**
Initial pilot experiment.

---

## Experiment 002 — 10-Subject LOSO

**Date:** 2026-09-26

**Dataset:**
10 subjects, recording 1

**Pipeline:**

* Single-channel EEG Fpz-Cz
* 30-second consecutive epochs
* Welch PSD
* 4-second Welch segments
* 50% overlap
* Six relative bandpower features
* Random Forest, 100 trees
* 10-fold Leave-One-Subject-Out (LOSO) validation

**Result:**
87.42% mean fold accuracy

**Observation:**
The initial pipeline used a Wake fallback for unmatched annotations.

**Decision:**
Investigate and correct the ground-truth labeling procedure before treating the result as the definitive baseline.

---

## Experiment 003 — Corrected Ground Truth and Feature Definition

**Date:** 2026-09-26

**Change:**
Unmatched / unsupported annotations were no longer assigned to Wake. Unscored (`?`), Movement (`M`), and unmatched annotations were excluded.

**Additional correction:**
The six frequency bands were explicitly specified as relative bandpower features:

* Delta: 0.5–4 Hz
* Theta: 4–8 Hz
* Alpha: 8–12 Hz
* Sigma: 12–16 Hz
* Beta: 16–30 Hz
* Gamma: 30–40 Hz

**Result:**

* 27,533 retained epochs
* 6 feature dimensions
* 10 subjects represented
* 87.35% mean LOSO accuracy

**Observation:**
The corrected labeling procedure produced a slightly lower mean accuracy than Experiment 002 (87.35% vs. 87.42%), indicating that the earlier Wake fallback had a measurable effect on the dataset.

**Decision:**
Use the corrected pipeline as the baseline for subsequent experiments.

**Next:**
Evaluate Wake-specific binary discrimination for potential use as a gating layer.

---

## Experiment 004 — Binary Wake vs. Non-Wake Classification

**Date:** 2026-09-26

**Purpose:**
Test whether the six-feature single-channel spectral representation is more effective for separating Wake from all non-Wake sleep stages than for five-class sleep-stage classification.

**Task:**
Binary classification:

* Wake = positive class
* N1, N2, N3, and REM = Non-Wake

**Pipeline:**

* Same corrected Experiment 003 feature extraction
* Single-channel EEG Fpz-Cz
* 30-second epochs
* Welch PSD
* 4-second segments
* 50% overlap
* Six relative bandpower features
* Random Forest, 100 trees
* 10-subject LOSO validation
* Fixed probability threshold = 0.50

**Dataset:**

* 27,533 retained epochs
* Wake: 19,084
* Non-Wake: 8,449

**Results:**

* Mean LOSO accuracy: 92.92%
* Pooled Wake sensitivity: 94.11%
* Wake miss rate: 5.89%
* Wake precision: 95.63%
* Wake F1: 94.87%
* Mean fold specificity: 90.99%
* ROC-AUC: 0.9739
* PR-AUC: 0.9838

**Observation:**
The binary Wake vs. Non-Wake task substantially outperformed the five-class task in overall discrimination. However, performance varied across held-out subjects, demonstrating subject-dependent variability despite LOSO validation.

**Interpretation:**
The result supports investigating Wake/non-Wake discrimination as a separate application-specific task rather than relying solely on five-stage classification performance.

**Next:**
Test whether the probability decision threshold can be selected without access to the held-out subject in order to increase Wake sensitivity while maintaining a predefined specificity constraint.

---

## Experiment 005 — Nested Wake-Threshold Selection

**Date:** 2026-09-26

**Purpose:**
Evaluate whether a Wake probability threshold can be selected using only the training subjects and then generalized to a completely unseen subject, with the objective of reducing missed Wake epochs.

**Validation design:**
Nested validation consisting of:

* Outer 10-subject LOSO evaluation
* Inner LOSO threshold selection performed only within the outer training subjects
* Outer test subject labels not used for threshold selection

**Threshold-selection rule:**

* Candidate thresholds evaluated from 0.00 to 1.00
* Objective: maximize mean inner-validation Wake sensitivity
* Constraint: mean inner-validation specificity ≥ 0.90
* Baseline threshold: 0.50

**Results — Fixed 0.50 baseline:**

* Mean fold accuracy: 92.9182%
* Mean fold Wake sensitivity: 94.3672%
* Mean fold specificity: 90.9925%
* Pooled Wake sensitivity: 94.1102%
* Pooled Wake miss rate: 5.8898%

**Results — Nested threshold selection:**

* Mean selected threshold: 0.4400
* Threshold SD: 0.0231
* Mean fold accuracy: 93.5184%
* Mean fold Wake sensitivity: 95.5957%
* Mean fold specificity: 90.1432%
* Pooled Wake sensitivity: 95.3731%
* Pooled Wake miss rate: 4.6269%

**Paired pooled change (nested − baseline):**

* Wake sensitivity: +1.2628 percentage points
* Wake miss rate: −1.2628 percentage points
* Specificity: −0.8877 percentage points
* Balanced accuracy: +0.1876 percentage points

**Observation:**
The nested procedure selected thresholds below the conventional 0.50 operating point for all ten outer folds, with selected thresholds ranging from 0.40 to 0.48.

On the outer held-out subjects, the nested thresholds increased Wake sensitivity relative to the fixed 0.50 threshold in every fold or maintained the same sensitivity. The corresponding reduction in specificity was modest at the cohort level.

**Interpretation:**
Under the predefined nested selection procedure, lowering the decision threshold to prioritize Wake sensitivity produced a reduction in the pooled Wake miss rate from 5.89% to 4.63% while maintaining approximately 90% mean specificity.

This result supports further investigation of threshold selection as an application-specific operating-point strategy. It does **not**, by itself, establish real-world stimulation safety or validate deployment in a physical wearable device.

**Decision:**
Retain the nested threshold-selection methodology and results for subsequent analysis.

**Next:**
Audit EXP005 result files and implementation for complete methodological reproducibility, then investigate the stability and practical implications of the Wake-sensitive operating point before making any claims regarding the proposed TMR gating application.
## Experiment 006 — Application-Specific Operating-Point and Cost Analysis
Date: 2026-09-26

Purpose:
Determine whether the Wake-sensitive operating-point behavior observed in EXP005
is robust across different predefined specificity constraints, and examine how
explicitly increasing the relative cost assigned to Wake false negatives changes
the selected decision threshold and resulting Wake/non-Wake trade-off.

Dataset:
Sleep-EDF
10 subjects, recording 1
27,533 retained 30-second epochs

Pipeline:
Single-channel EEG Fpz-Cz
30-second consecutive epochs
Welch PSD
4-second internal segments
50% overlap
Six relative spectral bandpower features:
Delta 0.5–4 Hz
Theta 4–8 Hz
Alpha 8–12 Hz
Sigma 12–16 Hz
Beta 16–30 Hz
Gamma 30–40 Hz
Random Forest, 100 trees
10-fold outer LOSO validation
9-fold inner LOSO threshold selection within each outer training set
Threshold grid: 0.00–1.00 in 0.01 increments

Analysis 1 — Specificity-Constrained Operating Points:
Specificity floors:
85%
90%
95%

Threshold-selection rule:
Maximize mean inner-subject Wake sensitivity subject to the specified
mean inner-subject specificity floor.

Results:

85% specificity floor:
Mean selected threshold: 0.230
Pooled Wake sensitivity: 98.276%
Pooled Wake miss rate: 1.724%
Pooled specificity: 84.507%

90% specificity floor:
Mean selected threshold: 0.440
Pooled Wake sensitivity: 95.373%
Pooled Wake miss rate: 4.627%
Pooled specificity: 89.407%

95% specificity floor:
Mean selected threshold: 0.785
Pooled Wake sensitivity: 86.580%
Pooled Wake miss rate: 13.420%
Pooled specificity: 94.650%

Observation:
Changing the predefined specificity requirement produced a systematic change
in the selected operating threshold and the resulting Wake sensitivity /
specificity trade-off. Lower specificity requirements permitted substantially
higher Wake sensitivity and lower Wake miss rates, while the higher specificity
requirement produced the opposite trade-off.

Analysis 2 — Asymmetric Error-Cost Sensitivity:
Wake false-negative : Non-Wake false-positive cost ratios:
1:1
2:1
5:1
10:1

Threshold-selection rule:
Minimize mean inner-subject cost per epoch across the threshold grid.

Results:

1:1 cost ratio:
Mean selected threshold: 0.279
Pooled Wake sensitivity: 97.254%
Pooled Wake miss rate: 2.746%
Pooled specificity: 85.939%

2:1 cost ratio:
Mean selected threshold: 0.201
Pooled Wake sensitivity: 98.528%
Pooled Wake miss rate: 1.472%
Pooled specificity: 83.359%

5:1 cost ratio:
Mean selected threshold: 0.115
Pooled Wake sensitivity: 99.371%
Pooled Wake miss rate: 0.629%
Pooled specificity: 77.879%

10:1 cost ratio:
Mean selected threshold: 0.070
Pooled Wake sensitivity: 99.675%
Pooled Wake miss rate: 0.325%
Pooled specificity: 71.559%

Observation:
Increasing the relative penalty assigned to Wake false negatives shifted the
selected operating threshold downward and produced progressively higher Wake
sensitivity, accompanied by progressively lower specificity.

Analysis 3 — Computational Characterization:
Feature extraction time for the 10-subject cohort: 8.63 s
Mean outer-fold Random Forest training time: 0.3517 s
Mean single-epoch inference time: 12.89 ms
Mean serialized model size: 14.38 MiB
Mean total Random Forest tree nodes: 183,624

Interpretation:
The computational analysis characterizes the workstation implementation and
provides measured software-level resource requirements. It does not constitute
hardware benchmarking or validation on a specific wearable or microcontroller.

Decision:
Retain Experiment 006 as the final major experimental analysis.
