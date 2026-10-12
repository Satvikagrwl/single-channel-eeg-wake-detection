import os
import csv
import json
import shutil
import sys
import platform
import importlib.metadata
from datetime import datetime

import numpy as np

from feature_extractor import extract_cohort_data

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)


# ==============================================================
# EXPERIMENT CONFIGURATION
# ==============================================================

EXPERIMENT_ID = "EXP005"
EXPERIMENT_NAME = "nested_wake_threshold"

# Binary target:
# 1 = Wake
# 0 = Non-Wake
POSITIVE_CLASS = 1

# EXP004 baseline operating point.
BASELINE_THRESHOLD = 0.50

# EXP005 pre-specified operating constraint.
# Among thresholds that achieve at least this mean specificity
# on the inner validation folds, choose the threshold with the
# highest mean Wake sensitivity.
SPECIFICITY_FLOOR = 0.90

# Fixed, pre-specified candidate threshold grid.
THRESHOLD_GRID = np.linspace(0.00, 1.00, 101)

# Same classifier as EXP004 so the experiment isolates threshold
# selection rather than changing the model family.
N_ESTIMATORS = 100
RANDOM_STATE = 42
N_JOBS = -1

# Never overwrite an existing experiment.
OVERWRITE_EXISTING = False


# ==============================================================
# PROJECT PATHS
# ==============================================================

CODE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    CODE_DIR
)

RESULTS_ROOT = os.path.join(
    PROJECT_ROOT,
    "results"
)

EXPERIMENT_DIR = os.path.join(
    RESULTS_ROOT,
    f"{EXPERIMENT_ID}_{EXPERIMENT_NAME}"
)

CODE_SNAPSHOT_DIR = os.path.join(
    EXPERIMENT_DIR,
    "code_snapshot"
)


# ==============================================================
# LABEL DEFINITIONS
# ==============================================================

ORIGINAL_CLASS_NAMES = {
    0: "Awake(W)",
    1: "Light(N1)",
    2: "Moderate(N2)",
    3: "Deep(N3)",
    4: "REM(R)"
}

BINARY_CLASS_NAMES = {
    0: "Non-Wake",
    1: "Wake"
}


# ==============================================================
# HELPER FUNCTIONS
# ==============================================================

def save_csv(filename, fieldnames, rows):
    """Save dictionaries as a CSV inside the experiment folder."""

    path = os.path.join(
        EXPERIMENT_DIR,
        filename
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


def get_package_version(package_name):
    """Safely obtain an installed package version."""

    try:
        return importlib.metadata.version(
            package_name
        )

    except Exception:
        return "unknown"


def safe_divide(numerator, denominator):
    """Avoid division-by-zero errors."""

    if denominator == 0:
        return 0.0

    return float(
        numerator / denominator
    )


def save_code_snapshot():
    """Preserve the exact code used for this experiment."""

    os.makedirs(
        CODE_SNAPSHOT_DIR,
        exist_ok=True
    )

    files_to_snapshot = [
        "run_pipeline.py",
        "feature_extractor.py",
        "model_trainer.py"
    ]

    for filename in files_to_snapshot:

        source = os.path.join(
            CODE_DIR,
            filename
        )

        if os.path.exists(source):

            destination = os.path.join(
                CODE_SNAPSHOT_DIR,
                filename
            )

            shutil.copy2(
                source,
                destination
            )


def append_registry_entry(summary):
    """
    Append this experiment to the master experiment registry.

    The existing registry columns are preserved so the experiment
    remains machine-readable alongside EXP003 and EXP004.
    """

    registry_path = os.path.join(
        RESULTS_ROOT,
        "experiment_registry.csv"
    )

    registry_exists = os.path.exists(
        registry_path
    )

    fieldnames = [
        "experiment_id",
        "experiment_name",
        "timestamp",
        "subjects",
        "epochs",
        "features",
        "validation",
        "mean_fold_accuracy",
        "accuracy_sd",
        "pooled_accuracy",
        "macro_f1",
        "weighted_f1"
    ]

    with open(
        registry_path,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        if not registry_exists:
            writer.writeheader()

        writer.writerow({
            "experiment_id": summary["experiment_id"],
            "experiment_name": summary["experiment_name"],
            "timestamp": summary["timestamp"],
            "subjects": summary["experiment"]["n_subjects"],
            "epochs": summary["experiment"]["n_epochs"],
            "features": summary["experiment"]["n_features"],
            "validation": summary["validation"]["outer_method"],
            "mean_fold_accuracy": summary["results"]["selected"]["mean_fold_accuracy"],
            "accuracy_sd": summary["results"]["selected"]["std_fold_accuracy"],
            "pooled_accuracy": summary["results"]["selected"]["pooled_accuracy"],
            "macro_f1": summary["results"]["selected"]["pooled_macro_f1"],
            "weighted_f1": summary["results"]["selected"]["pooled_weighted_f1"]
        })


def metric_bundle(y_true, y_pred, wake_probability):
    """
    Calculate the binary metrics used for the outer-fold and
    pooled analyses.
    """

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred
    )

    wake_precision = precision_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    wake_sensitivity = recall_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    wake_f1 = f1_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=[0, 1],
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=[0, 1],
        average="weighted",
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = safe_divide(
        tn,
        tn + fp
    )

    miss_rate = 1.0 - wake_sensitivity

    result = {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "wake_precision": float(wake_precision),
        "wake_sensitivity": float(wake_sensitivity),
        "wake_miss_rate": float(miss_rate),
        "wake_specificity": float(specificity),
        "wake_f1": float(wake_f1),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "true_wake": int(np.sum(y_true == 1)),
        "true_nonwake": int(np.sum(y_true == 0)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp)
    }

    # ROC-AUC and PR-AUC require both classes.
    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(
            roc_auc_score(
                y_true,
                wake_probability
            )
        )

        result["pr_auc"] = float(
            average_precision_score(
                y_true,
                wake_probability
            )
        )

    else:
        result["roc_auc"] = float("nan")
        result["pr_auc"] = float("nan")

    return result, cm


def train_random_forest(X_train, y_train):
    """Fit the fixed EXP004 Random Forest configuration."""

    if len(np.unique(y_train)) < 2:
        raise RuntimeError(
            "Training data do not contain both binary classes."
        )

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def get_wake_probability(model, X):
    """Extract the probability assigned to the Wake class."""

    probability_matrix = model.predict_proba(
        X
    )

    wake_column = np.where(
        model.classes_ == POSITIVE_CLASS
    )[0]

    if len(wake_column) != 1:
        raise RuntimeError(
            "Could not identify the Wake probability column."
        )

    return probability_matrix[:, wake_column[0]]


def evaluate_threshold_grid(
    y_true_by_subject,
    probability_by_subject,
    thresholds
):
    """
    Evaluate a threshold separately on each inner validation subject,
    then aggregate by equal-weight subject means.

    This keeps the threshold-selection objective at the subject level
    rather than allowing subjects with more epochs to dominate the
    selection criterion.
    """

    rows = []
    threshold_summaries = []

    subject_ids = sorted(
        y_true_by_subject.keys()
    )

    for threshold in thresholds:

        fold_sensitivities = []
        fold_specificities = []
        fold_balanced_accuracies = []
        fold_precisions = []
        fold_f1s = []

        for subject_id in subject_ids:

            y_true = y_true_by_subject[
                subject_id
            ]

            probabilities = probability_by_subject[
                subject_id
            ]

            y_pred = (
                probabilities >= threshold
            ).astype(int)

            metrics, _ = metric_bundle(
                y_true,
                y_pred,
                probabilities
            )

            fold_sensitivities.append(
                metrics["wake_sensitivity"]
            )

            fold_specificities.append(
                metrics["wake_specificity"]
            )

            fold_balanced_accuracies.append(
                metrics["balanced_accuracy"]
            )

            fold_precisions.append(
                metrics["wake_precision"]
            )

            fold_f1s.append(
                metrics["wake_f1"]
            )

            rows.append({
                "threshold": float(threshold),
                "inner_validation_subject": int(subject_id),
                "wake_sensitivity": float(
                    metrics["wake_sensitivity"]
                ),
                "wake_specificity": float(
                    metrics["wake_specificity"]
                ),
                "wake_miss_rate": float(
                    metrics["wake_miss_rate"]
                ),
                "wake_precision": float(
                    metrics["wake_precision"]
                ),
                "wake_f1": float(
                    metrics["wake_f1"]
                ),
                "balanced_accuracy": float(
                    metrics["balanced_accuracy"]
                )
            })

        mean_sensitivity = float(
            np.mean(fold_sensitivities)
        )

        mean_specificity = float(
            np.mean(fold_specificities)
        )

        mean_balanced_accuracy = float(
            np.mean(fold_balanced_accuracies)
        )

        mean_precision = float(
            np.mean(fold_precisions)
        )

        mean_f1 = float(
            np.mean(fold_f1s)
        )

        feasible = (
            mean_specificity >= SPECIFICITY_FLOOR
        )

        threshold_summaries.append({
            "threshold": float(threshold),
            "mean_wake_sensitivity": mean_sensitivity,
            "mean_wake_specificity": mean_specificity,
            "mean_wake_miss_rate": 1.0 - mean_sensitivity,
            "mean_wake_precision": mean_precision,
            "mean_wake_f1": mean_f1,
            "mean_balanced_accuracy": mean_balanced_accuracy,
            "specificity_floor": float(SPECIFICITY_FLOOR),
            "feasible": int(feasible)
        })

    return rows, threshold_summaries


def choose_nested_threshold(threshold_summaries):
    """
    Select the threshold using only inner-validation outcomes.

    Primary objective:
        maximize mean Wake sensitivity

    Constraint:
        mean specificity >= SPECIFICITY_FLOOR

    Deterministic tie-breaks:
        1. higher mean specificity
        2. threshold closest to 0.50
        3. lower threshold

    If no threshold satisfies the specificity floor, fall back to the
    threshold with the highest mean specificity, then highest mean
    sensitivity, then threshold closest to 0.50, then lower threshold.
    """

    feasible = [
        row
        for row in threshold_summaries
        if row["feasible"] == 1
    ]

    if feasible:

        selected = sorted(
            feasible,
            key=lambda row: (
                -row["mean_wake_sensitivity"],
                -row["mean_wake_specificity"],
                abs(row["threshold"] - BASELINE_THRESHOLD),
                row["threshold"]
            )
        )[0]

        selection_mode = (
            "specificity_constrained_sensitivity_maximization"
        )

    else:

        selected = sorted(
            threshold_summaries,
            key=lambda row: (
                -row["mean_wake_specificity"],
                -row["mean_wake_sensitivity"],
                abs(row["threshold"] - BASELINE_THRESHOLD),
                row["threshold"]
            )
        )[0]

        selection_mode = (
            "fallback_maximize_specificity_then_sensitivity"
        )

    return float(selected["threshold"]), selection_mode, selected


def collect_pooled_metrics(
    y_true,
    y_pred,
    wake_probabilities
):
    """Return the complete pooled metric dictionary plus confusion matrix."""

    metrics, cm = metric_bundle(
        y_true,
        y_pred,
        wake_probabilities
    )

    return metrics, cm


# ==============================================================
# MAIN
# ==============================================================

def main():

    # ==========================================================
    # 1. PREPARE EXPERIMENT DIRECTORY
    # ==========================================================

    os.makedirs(
        RESULTS_ROOT,
        exist_ok=True
    )

    if os.path.exists(EXPERIMENT_DIR):

        if not OVERWRITE_EXISTING:
            raise FileExistsError(
                "\n❌ EXPERIMENT ALREADY EXISTS\n\n"
                f"{EXPERIMENT_DIR}\n\n"
                "This experiment is protected from overwrite.\n"
                "Change EXPERIMENT_ID and EXPERIMENT_NAME for the next experiment."
            )

    else:
        os.makedirs(
            EXPERIMENT_DIR
        )

    # ==========================================================
    # 2. SNAPSHOT EXACT CODE
    # ==========================================================

    save_code_snapshot()

    run_timestamp = datetime.now().isoformat()

    print(
        "\n=========================================================="
    )
    print(
        f"🧪 {EXPERIMENT_ID}: {EXPERIMENT_NAME}"
    )
    print(
        "=========================================================="
    )

    print(
        f"📁 Results directory:\n{EXPERIMENT_DIR}\n"
    )

    print(
        "🎯 Task: Wake vs Non-Wake"
    )

    print(
        f"🎚️ Baseline threshold: {BASELINE_THRESHOLD:.2f}"
    )

    print(
        f"🛡️ Inner-validation specificity floor: "
        f"{SPECIFICITY_FLOOR:.2f}"
    )

    print(
        "🔁 Nested structure: outer LOSO + inner LOSO threshold selection\n"
    )

    # ==========================================================
    # 3. LOAD CORRECTED FEATURE MATRIX
    # ==========================================================

    cohort_subjects = [
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10
    ]

    X_matrix, original_y, subject_tracking = (
        extract_cohort_data(
            cohort_subjects
        )
    )

    print(
        f"\n📊 Feature matrix: {X_matrix.shape}"
    )

    print(
        f"🧬 Subjects: {np.unique(subject_tracking)}"
    )

    # ==========================================================
    # 4. INTEGRITY CHECKS
    # ==========================================================

    if X_matrix.ndim != 2:
        raise RuntimeError(
            f"Unexpected feature matrix shape: {X_matrix.shape}"
        )

    if X_matrix.shape[1] != 6:
        raise RuntimeError(
            f"Expected 6 features, got {X_matrix.shape[1]}"
        )

    if not (
        len(X_matrix)
        == len(original_y)
        == len(subject_tracking)
    ):
        raise RuntimeError(
            "X, y, and subject arrays have different lengths."
        )

    if not set(np.unique(original_y)).issubset(
        set(ORIGINAL_CLASS_NAMES.keys())
    ):
        raise RuntimeError(
            "Unexpected original sleep-stage labels detected."
        )

    # ==========================================================
    # 5. CONVERT FIVE-CLASS LABELS TO BINARY
    # ==========================================================

    binary_y = (
        original_y == 0
    ).astype(int)

    wake_count = int(
        np.sum(binary_y == 1)
    )

    nonwake_count = int(
        np.sum(binary_y == 0)
    )

    print(
        "\n📌 Binary class distribution:"
    )
    print(
        f"   Wake:     {wake_count}"
    )
    print(
        f"   Non-Wake: {nonwake_count}"
    )

    # ==========================================================
    # 6. OUTER-LOSO STORAGE
    # ==========================================================

    unique_subjects = np.unique(
        subject_tracking
    )

    outer_fold_results = []
    threshold_selection_rows = []
    inner_threshold_scan_rows = []
    prediction_rows = []

    baseline_y_true = []
    baseline_y_pred = []
    baseline_probabilities = []

    selected_y_true = []
    selected_y_pred = []
    selected_probabilities = []

    baseline_fold_accuracies = []
    selected_fold_accuracies = []

    baseline_fold_sensitivities = []
    selected_fold_sensitivities = []

    baseline_fold_specificities = []
    selected_fold_specificities = []

    baseline_fold_balanced_accuracies = []
    selected_fold_balanced_accuracies = []

    baseline_fold_precisions = []
    selected_fold_precisions = []

    baseline_fold_f1s = []
    selected_fold_f1s = []

    baseline_fold_roc_aucs = []
    selected_fold_roc_aucs = []

    baseline_fold_pr_aucs = []
    selected_fold_pr_aucs = []

    # ==========================================================
    # 7. OUTER LOSO
    # ==========================================================

    print(
        "\n🤖 Launching EXP005 nested threshold validation..."
    )

    for outer_test_sub in unique_subjects:

        print(
            "\n----------------------------------------------------------"
        )

        print(
            f"⚡ OUTER held-out Subject {outer_test_sub:02d}"
        )

        outer_train_subjects = [
            int(sub)
            for sub in unique_subjects
            if sub != outer_test_sub
        ]

        outer_train_mask = (
            subject_tracking != outer_test_sub
        )

        outer_test_mask = (
            subject_tracking == outer_test_sub
        )

        X_outer_train = X_matrix[
            outer_train_mask
        ]

        y_outer_train = binary_y[
            outer_train_mask
        ]

        X_outer_test = X_matrix[
            outer_test_mask
        ]

        y_outer_test = binary_y[
            outer_test_mask
        ]

        original_outer_test_stages = original_y[
            outer_test_mask
        ]

        if len(X_outer_train) == 0:
            raise RuntimeError(
                f"Empty outer training set for Subject {outer_test_sub}."
            )

        if len(X_outer_test) == 0:
            raise RuntimeError(
                f"Empty outer test set for Subject {outer_test_sub}."
            )

        if len(np.unique(y_outer_train)) < 2:
            raise RuntimeError(
                f"Outer training data for Subject {outer_test_sub} do not contain both classes."
            )

        if len(np.unique(y_outer_test)) < 2:
            raise RuntimeError(
                f"Outer test data for Subject {outer_test_sub} do not contain both classes."
            )

        print(
            f"   Outer training epochs: {len(X_outer_train)}"
        )
        print(
            f"   Outer testing epochs:  {len(X_outer_test)}"
        )

        # ======================================================
        # 7A. INNER LOSO FOR THRESHOLD SELECTION
        # ======================================================

        inner_y_true_by_subject = {}
        inner_probability_by_subject = {}

        print(
            "   🔎 Running inner LOSO threshold selection..."
        )

        for inner_val_sub in outer_train_subjects:

            inner_train_mask = (
                subject_tracking != outer_test_sub
            ) & (
                subject_tracking != inner_val_sub
            )

            inner_val_mask = (
                subject_tracking == inner_val_sub
            )

            X_inner_train = X_matrix[
                inner_train_mask
            ]

            y_inner_train = binary_y[
                inner_train_mask
            ]

            X_inner_val = X_matrix[
                inner_val_mask
            ]

            y_inner_val = binary_y[
                inner_val_mask
            ]

            if len(X_inner_train) == 0:
                raise RuntimeError(
                    f"Empty inner training set for inner subject {inner_val_sub}."
                )

            if len(X_inner_val) == 0:
                raise RuntimeError(
                    f"Empty inner validation set for inner subject {inner_val_sub}."
                )

            if len(np.unique(y_inner_train)) < 2:
                raise RuntimeError(
                    f"Inner training data for Subject {inner_val_sub} do not contain both classes."
                )

            model_inner = train_random_forest(
                X_inner_train,
                y_inner_train
            )

            inner_probability = get_wake_probability(
                model_inner,
                X_inner_val
            )

            inner_y_true_by_subject[
                int(inner_val_sub)
            ] = y_inner_val

            inner_probability_by_subject[
                int(inner_val_sub)
            ] = inner_probability

        inner_rows, threshold_summaries = evaluate_threshold_grid(
            inner_y_true_by_subject,
            inner_probability_by_subject,
            THRESHOLD_GRID
        )

        inner_threshold_scan_rows.extend(
            [
                {
                    "outer_held_out_subject": int(outer_test_sub),
                    **row
                }
                for row in inner_rows
            ]
        )

        selected_threshold, selection_mode, selected_inner_summary = (
            choose_nested_threshold(
                threshold_summaries
            )
        )

        # Store one compact summary row for the selected threshold.
        baseline_inner_summary = min(
            threshold_summaries,
            key=lambda row: abs(
                row["threshold"] - BASELINE_THRESHOLD
            )
        )

        threshold_selection_rows.append({
            "outer_held_out_subject": int(outer_test_sub),
            "baseline_threshold": float(BASELINE_THRESHOLD),
            "selected_threshold": float(selected_threshold),
            "specificity_floor": float(SPECIFICITY_FLOOR),
            "selection_mode": selection_mode,
            "baseline_inner_mean_wake_sensitivity": float(
                baseline_inner_summary["mean_wake_sensitivity"]
            ),
            "baseline_inner_mean_specificity": float(
                baseline_inner_summary["mean_wake_specificity"]
            ),
            "selected_inner_mean_wake_sensitivity": float(
                selected_inner_summary["mean_wake_sensitivity"]
            ),
            "selected_inner_mean_specificity": float(
                selected_inner_summary["mean_wake_specificity"]
            ),
            "selected_inner_mean_wake_miss_rate": float(
                selected_inner_summary["mean_wake_miss_rate"]
            ),
            "selected_threshold_feasible": int(
                selected_inner_summary["feasible"]
            )
        })

        print(
            f"   🎚️ Selected threshold: {selected_threshold:.2f}"
        )

        print(
            f"   📈 Inner mean Wake sensitivity: "
            f"{selected_inner_summary['mean_wake_sensitivity']:.2%}"
        )

        print(
            f"   📉 Inner mean specificity: "
            f"{selected_inner_summary['mean_wake_specificity']:.2%}"
        )

        # ======================================================
        # 7B. FIT ON ALL OUTER-TRAINING SUBJECTS
        # ======================================================

        model_outer = train_random_forest(
            X_outer_train,
            y_outer_train
        )

        outer_wake_probability = get_wake_probability(
            model_outer,
            X_outer_test
        )

        # IMPORTANT:
        # Both thresholds are applied to predictions from a model
        # trained only on the 9 outer-training subjects.
        # The outer test labels are never used for threshold choice.

        baseline_pred = (
            outer_wake_probability
            >= BASELINE_THRESHOLD
        ).astype(int)

        selected_pred = (
            outer_wake_probability
            >= selected_threshold
        ).astype(int)

        # ======================================================
        # 7C. OUTER-FOLD METRICS
        # ======================================================

        baseline_metrics, baseline_cm = metric_bundle(
            y_outer_test,
            baseline_pred,
            outer_wake_probability
        )

        selected_metrics, selected_cm = metric_bundle(
            y_outer_test,
            selected_pred,
            outer_wake_probability
        )

        outer_fold_results.append({
            "held_out_subject": int(outer_test_sub),
            "training_epochs": int(len(X_outer_train)),
            "test_epochs": int(len(X_outer_test)),
            "selected_threshold": float(selected_threshold),
            "baseline_accuracy": baseline_metrics["accuracy"],
            "baseline_balanced_accuracy": baseline_metrics["balanced_accuracy"],
            "baseline_wake_precision": baseline_metrics["wake_precision"],
            "baseline_wake_sensitivity": baseline_metrics["wake_sensitivity"],
            "baseline_wake_miss_rate": baseline_metrics["wake_miss_rate"],
            "baseline_wake_specificity": baseline_metrics["wake_specificity"],
            "baseline_wake_f1": baseline_metrics["wake_f1"],
            "baseline_roc_auc": baseline_metrics["roc_auc"],
            "baseline_pr_auc": baseline_metrics["pr_auc"],
            "baseline_true_negative": baseline_metrics["true_negative"],
            "baseline_false_positive": baseline_metrics["false_positive"],
            "baseline_false_negative": baseline_metrics["false_negative"],
            "baseline_true_positive": baseline_metrics["true_positive"],
            "selected_accuracy": selected_metrics["accuracy"],
            "selected_balanced_accuracy": selected_metrics["balanced_accuracy"],
            "selected_wake_precision": selected_metrics["wake_precision"],
            "selected_wake_sensitivity": selected_metrics["wake_sensitivity"],
            "selected_wake_miss_rate": selected_metrics["wake_miss_rate"],
            "selected_wake_specificity": selected_metrics["wake_specificity"],
            "selected_wake_f1": selected_metrics["wake_f1"],
            "selected_roc_auc": selected_metrics["roc_auc"],
            "selected_pr_auc": selected_metrics["pr_auc"],
            "selected_true_negative": selected_metrics["true_negative"],
            "selected_false_positive": selected_metrics["false_positive"],
            "selected_false_negative": selected_metrics["false_negative"],
            "selected_true_positive": selected_metrics["true_positive"]
        })

        # ======================================================
        # 7D. SAVE PREDICTIONS
        # ======================================================

        for i in range(len(y_outer_test)):

            original_stage = int(
                original_outer_test_stages[i]
            )

            prediction_rows.append({
                "held_out_subject": int(outer_test_sub),
                "epoch_index_within_fold": int(i),
                "original_stage_label": original_stage,
                "original_stage": ORIGINAL_CLASS_NAMES[original_stage],
                "binary_true_label": int(y_outer_test[i]),
                "binary_true_class": BINARY_CLASS_NAMES[int(y_outer_test[i])],
                "probability_wake": float(outer_wake_probability[i]),
                "baseline_threshold": float(BASELINE_THRESHOLD),
                "baseline_predicted_label": int(baseline_pred[i]),
                "baseline_predicted_class": BINARY_CLASS_NAMES[int(baseline_pred[i])],
                "selected_threshold": float(selected_threshold),
                "selected_predicted_label": int(selected_pred[i]),
                "selected_predicted_class": BINARY_CLASS_NAMES[int(selected_pred[i])],
                "baseline_epoch_correct": int(y_outer_test[i] == baseline_pred[i]),
                "selected_epoch_correct": int(y_outer_test[i] == selected_pred[i])
            })

        # ======================================================
        # 7E. SAVE SUBJECT-SPECIFIC CONFUSION MATRICES
        # ======================================================

        baseline_rows = []
        selected_rows = []

        for actual_index, actual_class in enumerate([0, 1]):

            for predicted_index, predicted_class in enumerate([0, 1]):

                baseline_rows.append({
                    "actual_class": BINARY_CLASS_NAMES[actual_class],
                    "predicted_class": BINARY_CLASS_NAMES[predicted_class],
                    "count": int(
                        baseline_cm[
                            actual_index,
                            predicted_index
                        ]
                    )
                })

                selected_rows.append({
                    "actual_class": BINARY_CLASS_NAMES[actual_class],
                    "predicted_class": BINARY_CLASS_NAMES[predicted_class],
                    "count": int(
                        selected_cm[
                            actual_index,
                            predicted_index
                        ]
                    )
                })

        save_csv(
            f"confusion_matrix_subject_{outer_test_sub:02d}_baseline.csv",
            ["actual_class", "predicted_class", "count"],
            baseline_rows
        )

        save_csv(
            f"confusion_matrix_subject_{outer_test_sub:02d}_selected.csv",
            ["actual_class", "predicted_class", "count"],
            selected_rows
        )

        # ======================================================
        # 7F. ACCUMULATE POOLED ARRAYS
        # ======================================================

        baseline_y_true.extend(
            y_outer_test
        )
        baseline_y_pred.extend(
            baseline_pred
        )
        baseline_probabilities.extend(
            outer_wake_probability
        )

        selected_y_true.extend(
            y_outer_test
        )
        selected_y_pred.extend(
            selected_pred
        )
        selected_probabilities.extend(
            outer_wake_probability
        )

        baseline_fold_accuracies.append(
            baseline_metrics["accuracy"]
        )
        selected_fold_accuracies.append(
            selected_metrics["accuracy"]
        )

        baseline_fold_sensitivities.append(
            baseline_metrics["wake_sensitivity"]
        )
        selected_fold_sensitivities.append(
            selected_metrics["wake_sensitivity"]
        )

        baseline_fold_specificities.append(
            baseline_metrics["wake_specificity"]
        )
        selected_fold_specificities.append(
            selected_metrics["wake_specificity"]
        )

        baseline_fold_balanced_accuracies.append(
            baseline_metrics["balanced_accuracy"]
        )
        selected_fold_balanced_accuracies.append(
            selected_metrics["balanced_accuracy"]
        )

        baseline_fold_precisions.append(
            baseline_metrics["wake_precision"]
        )
        selected_fold_precisions.append(
            selected_metrics["wake_precision"]
        )

        baseline_fold_f1s.append(
            baseline_metrics["wake_f1"]
        )
        selected_fold_f1s.append(
            selected_metrics["wake_f1"]
        )

        baseline_fold_roc_aucs.append(
            baseline_metrics["roc_auc"]
        )
        selected_fold_roc_aucs.append(
            selected_metrics["roc_auc"]
        )

        baseline_fold_pr_aucs.append(
            baseline_metrics["pr_auc"]
        )
        selected_fold_pr_aucs.append(
            selected_metrics["pr_auc"]
        )

        print(
            f"   Baseline 0.50  — Wake sensitivity: "
            f"{baseline_metrics['wake_sensitivity']:.2%} | "
            f"specificity: {baseline_metrics['wake_specificity']:.2%}"
        )

        print(
            f"   Nested {selected_threshold:.2f} — Wake sensitivity: "
            f"{selected_metrics['wake_sensitivity']:.2%} | "
            f"specificity: {selected_metrics['wake_specificity']:.2%}"
        )

    # ==========================================================
    # 8. CONVERT POOLED ARRAYS
    # ==========================================================

    baseline_y_true = np.asarray(
        baseline_y_true
    )
    baseline_y_pred = np.asarray(
        baseline_y_pred
    )
    baseline_probabilities = np.asarray(
        baseline_probabilities
    )

    selected_y_true = np.asarray(
        selected_y_true
    )
    selected_y_pred = np.asarray(
        selected_y_pred
    )
    selected_probabilities = np.asarray(
        selected_probabilities
    )

    # ==========================================================
    # 9. POOLED OUTER-OF-FOLD METRICS
    # ==========================================================

    pooled_baseline, baseline_pooled_cm = collect_pooled_metrics(
        baseline_y_true,
        baseline_y_pred,
        baseline_probabilities
    )

    pooled_selected, selected_pooled_cm = collect_pooled_metrics(
        selected_y_true,
        selected_y_pred,
        selected_probabilities
    )

    # ==========================================================
    # 10. FINAL REPORT
    # ==========================================================

    print(
        "\n=========================================================="
    )
    print(
        "📊 EXP005 FINAL RESULTS"
    )
    print(
        "=========================================================="
    )

    print(
        "\nBASELINE: fixed threshold 0.50"
    )
    print(
        f"   Mean fold accuracy:        {np.mean(baseline_fold_accuracies):.4%}"
    )
    print(
        f"   Mean fold Wake sensitivity:{np.mean(baseline_fold_sensitivities):.4%}"
    )
    print(
        f"   Mean fold specificity:     {np.mean(baseline_fold_specificities):.4%}"
    )
    print(
        f"   Pooled Wake sensitivity:   {pooled_baseline['wake_sensitivity']:.4%}"
    )
    print(
        f"   Pooled Wake miss rate:     {pooled_baseline['wake_miss_rate']:.4%}"
    )

    print(
        "\nNESTED: fold-specific threshold selected on inner LOSO"
    )
    print(
        f"   Mean selected threshold:    {np.mean([r['selected_threshold'] for r in outer_fold_results]):.4f}"
    )
    print(
        f"   Threshold SD:              {np.std([r['selected_threshold'] for r in outer_fold_results], ddof=1):.4f}"
    )
    print(
        f"   Mean fold accuracy:        {np.mean(selected_fold_accuracies):.4%}"
    )
    print(
        f"   Mean fold Wake sensitivity:{np.mean(selected_fold_sensitivities):.4%}"
    )
    print(
        f"   Mean fold specificity:     {np.mean(selected_fold_specificities):.4%}"
    )
    print(
        f"   Pooled Wake sensitivity:   {pooled_selected['wake_sensitivity']:.4%}"
    )
    print(
        f"   Pooled Wake miss rate:     {pooled_selected['wake_miss_rate']:.4%}"
    )

    print(
        "\nPaired pooled changes (nested minus baseline):"
    )
    print(
        f"   Wake sensitivity Δ:        "
        f"{pooled_selected['wake_sensitivity'] - pooled_baseline['wake_sensitivity']:+.4%}"
    )
    print(
        f"   Wake miss rate Δ:          "
        f"{pooled_selected['wake_miss_rate'] - pooled_baseline['wake_miss_rate']:+.4%}"
    )
    print(
        f"   Specificity Δ:             "
        f"{pooled_selected['wake_specificity'] - pooled_baseline['wake_specificity']:+.4%}"
    )
    print(
        f"   Balanced accuracy Δ:       "
        f"{pooled_selected['balanced_accuracy'] - pooled_baseline['balanced_accuracy']:+.4%}"
    )

    baseline_report = classification_report(
        baseline_y_true,
        baseline_y_pred,
        labels=[0, 1],
        target_names=["Non-Wake", "Wake"],
        zero_division=0
    )

    selected_report = classification_report(
        selected_y_true,
        selected_y_pred,
        labels=[0, 1],
        target_names=["Non-Wake", "Wake"],
        zero_division=0
    )

    # ==========================================================
    # 11. SAVE OUTER FOLD RESULTS
    # ==========================================================

    save_csv(
        "outer_fold_results.csv",
        list(outer_fold_results[0].keys()),
        outer_fold_results
    )

    # ==========================================================
    # 12. SAVE THRESHOLD SELECTION
    # ==========================================================

    save_csv(
        "threshold_selection.csv",
        list(threshold_selection_rows[0].keys()),
        threshold_selection_rows
    )

    save_csv(
        "inner_threshold_scan.csv",
        [
            "outer_held_out_subject",
            "threshold",
            "inner_validation_subject",
            "wake_sensitivity",
            "wake_specificity",
            "wake_miss_rate",
            "wake_precision",
            "wake_f1",
            "balanced_accuracy"
        ],
        inner_threshold_scan_rows
    )

    # ==========================================================
    # 13. SAVE ALL PREDICTIONS
    # ==========================================================

    save_csv(
        "loso_predictions.csv",
        [
            "held_out_subject",
            "epoch_index_within_fold",
            "original_stage_label",
            "original_stage",
            "binary_true_label",
            "binary_true_class",
            "probability_wake",
            "baseline_threshold",
            "baseline_predicted_label",
            "baseline_predicted_class",
            "selected_threshold",
            "selected_predicted_label",
            "selected_predicted_class",
            "baseline_epoch_correct",
            "selected_epoch_correct"
        ],
        prediction_rows
    )

    # ==========================================================
    # 14. SAVE POOLED CONFUSION MATRICES
    # ==========================================================

    def confusion_rows(cm):
        rows = []

        for actual_index, actual_class in enumerate([0, 1]):
            for predicted_index, predicted_class in enumerate([0, 1]):
                rows.append({
                    "actual_class": BINARY_CLASS_NAMES[actual_class],
                    "predicted_class": BINARY_CLASS_NAMES[predicted_class],
                    "count": int(
                        cm[
                            actual_index,
                            predicted_index
                        ]
                    )
                })

        return rows

    save_csv(
        "confusion_matrix_overall_baseline.csv",
        ["actual_class", "predicted_class", "count"],
        confusion_rows(baseline_pooled_cm)
    )

    save_csv(
        "confusion_matrix_overall_selected.csv",
        ["actual_class", "predicted_class", "count"],
        confusion_rows(selected_pooled_cm)
    )

    # ==========================================================
    # 15. SAVE ROC CURVE DATA
    # ==========================================================

    fpr, tpr, roc_thresholds = roc_curve(
        selected_y_true,
        selected_probabilities,
        pos_label=1
    )

    roc_rows = []

    for i in range(len(fpr)):
        roc_rows.append({
            "false_positive_rate": float(fpr[i]),
            "true_positive_rate": float(tpr[i]),
            "threshold": float(roc_thresholds[i])
        })

    save_csv(
        "roc_curve.csv",
        [
            "false_positive_rate",
            "true_positive_rate",
            "threshold"
        ],
        roc_rows
    )

    # ==========================================================
    # 16. SAVE PR CURVE DATA
    # ==========================================================

    precision_values, recall_values, pr_thresholds = (
        precision_recall_curve(
            selected_y_true,
            selected_probabilities,
            pos_label=1
        )
    )

    pr_rows = []

    for i in range(len(precision_values)):

        threshold = (
            float(pr_thresholds[i])
            if i < len(pr_thresholds)
            else ""
        )

        pr_rows.append({
            "precision": float(precision_values[i]),
            "recall": float(recall_values[i]),
            "threshold": threshold
        })

    save_csv(
        "precision_recall_curve.csv",
        [
            "precision",
            "recall",
            "threshold"
        ],
        pr_rows
    )

    # ==========================================================
    # 17. SAVE TEXT REPORT
    # ==========================================================

    mean_baseline_accuracy = float(
        np.mean(baseline_fold_accuracies)
    )

    sd_baseline_accuracy = float(
        np.std(baseline_fold_accuracies, ddof=1)
    )

    mean_selected_accuracy = float(
        np.mean(selected_fold_accuracies)
    )

    sd_selected_accuracy = float(
        np.std(selected_fold_accuracies, ddof=1)
    )

    mean_selected_threshold = float(
        np.mean(
            [row["selected_threshold"] for row in outer_fold_results]
        )
    )

    sd_selected_threshold = float(
        np.std(
            [row["selected_threshold"] for row in outer_fold_results],
            ddof=1
        )
    )

    with open(
        os.path.join(
            EXPERIMENT_DIR,
            "classification_report.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"{EXPERIMENT_ID}: {EXPERIMENT_NAME}\n"
        )
        f.write(
            "=" * 76 + "\n\n"
        )

        f.write(
            "Task: Wake vs Non-Wake\n"
        )
        f.write(
            "Positive class: Wake\n"
        )
        f.write(
            f"Baseline threshold: {BASELINE_THRESHOLD:.2f}\n"
        )
        f.write(
            f"Specificity floor for nested selection: {SPECIFICITY_FLOOR:.2f}\n"
        )
        f.write(
            "Threshold grid: 0.00 to 1.00 in 0.01 increments\n"
        )
        f.write(
            "Outer validation: Leave-One-Subject-Out\n"
        )
        f.write(
            "Inner threshold selection: Leave-One-Subject-Out over the 9 outer-training subjects\n\n"
        )

        f.write(
            "BASELINE (fixed threshold 0.50)\n"
        )
        f.write(
            f"Mean fold accuracy: {mean_baseline_accuracy:.6f}\n"
        )
        f.write(
            f"Fold accuracy SD: {sd_baseline_accuracy:.6f}\n"
        )
        f.write(
            f"Pooled accuracy: {pooled_baseline['accuracy']:.6f}\n"
        )
        f.write(
            f"Pooled Wake sensitivity: {pooled_baseline['wake_sensitivity']:.6f}\n"
        )
        f.write(
            f"Pooled Wake miss rate: {pooled_baseline['wake_miss_rate']:.6f}\n"
        )
        f.write(
            f"Pooled Wake specificity: {pooled_baseline['wake_specificity']:.6f}\n"
        )
        f.write(
            f"Pooled Wake precision: {pooled_baseline['wake_precision']:.6f}\n"
        )
        f.write(
            f"Pooled Wake F1: {pooled_baseline['wake_f1']:.6f}\n"
        )
        f.write(
            f"Pooled Macro-F1: {pooled_baseline['macro_f1']:.6f}\n"
        )
        f.write(
            f"Pooled Weighted-F1: {pooled_baseline['weighted_f1']:.6f}\n"
        )
        f.write(
            f"Pooled ROC-AUC: {pooled_baseline['roc_auc']:.6f}\n"
        )
        f.write(
            f"Pooled PR-AUC: {pooled_baseline['pr_auc']:.6f}\n\n"
        )

        f.write(
            "NESTED THRESHOLD (fold-specific threshold selected without outer test labels)\n"
        )
        f.write(
            f"Mean selected threshold: {mean_selected_threshold:.6f}\n"
        )
        f.write(
            f"Selected threshold SD: {sd_selected_threshold:.6f}\n"
        )
        f.write(
            f"Mean fold accuracy: {mean_selected_accuracy:.6f}\n"
        )
        f.write(
            f"Fold accuracy SD: {sd_selected_accuracy:.6f}\n"
        )
        f.write(
            f"Pooled accuracy: {pooled_selected['accuracy']:.6f}\n"
        )
        f.write(
            f"Pooled Wake sensitivity: {pooled_selected['wake_sensitivity']:.6f}\n"
        )
        f.write(
            f"Pooled Wake miss rate: {pooled_selected['wake_miss_rate']:.6f}\n"
        )
        f.write(
            f"Pooled Wake specificity: {pooled_selected['wake_specificity']:.6f}\n"
        )
        f.write(
            f"Pooled Wake precision: {pooled_selected['wake_precision']:.6f}\n"
        )
        f.write(
            f"Pooled Wake F1: {pooled_selected['wake_f1']:.6f}\n"
        )
        f.write(
            f"Pooled Macro-F1: {pooled_selected['macro_f1']:.6f}\n"
        )
        f.write(
            f"Pooled Weighted-F1: {pooled_selected['weighted_f1']:.6f}\n"
        )
        f.write(
            f"Pooled ROC-AUC: {pooled_selected['roc_auc']:.6f}\n"
        )
        f.write(
            f"Pooled PR-AUC: {pooled_selected['pr_auc']:.6f}\n\n"
        )

        f.write(
            "PAIRED CHANGE (nested minus baseline)\n"
        )
        f.write(
            f"Wake sensitivity delta: {pooled_selected['wake_sensitivity'] - pooled_baseline['wake_sensitivity']:+.6f}\n"
        )
        f.write(
            f"Wake miss rate delta: {pooled_selected['wake_miss_rate'] - pooled_baseline['wake_miss_rate']:+.6f}\n"
        )
        f.write(
            f"Specificity delta: {pooled_selected['wake_specificity'] - pooled_baseline['wake_specificity']:+.6f}\n"
        )
        f.write(
            f"Balanced accuracy delta: {pooled_selected['balanced_accuracy'] - pooled_baseline['balanced_accuracy']:+.6f}\n\n"
        )

        f.write(
            "BASELINE CLASSIFICATION REPORT\n"
        )
        f.write(
            "=" * 76 + "\n\n"
        )
        f.write(
            baseline_report
        )

        f.write(
            "\n\nNESTED THRESHOLD CLASSIFICATION REPORT\n"
        )
        f.write(
            "=" * 76 + "\n\n"
        )
        f.write(
            selected_report
        )

        f.write(
            "\n\nBASELINE POOLED CONFUSION MATRIX\n"
        )
        f.write(
            str(baseline_pooled_cm)
        )

        f.write(
            "\n\nNESTED THRESHOLD POOLED CONFUSION MATRIX\n"
        )
        f.write(
            str(selected_pooled_cm)
        )

    # ==========================================================
    # 18. SAVE EXPERIMENT SUMMARY
    # ==========================================================

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "timestamp": run_timestamp,

        "hypothesis": (
            "Selecting the Wake decision threshold using only inner "
            "subject-independent validation can reduce Wake misses "
            "on an unseen outer subject while maintaining a pre-specified "
            "specificity floor."
        ),

        "primary_metric": (
            "Wake sensitivity / recall"
        ),

        "secondary_metrics": [
            "Wake miss rate",
            "Wake specificity",
            "Wake precision",
            "Wake F1",
            "balanced accuracy",
            "ROC-AUC",
            "PR-AUC"
        ],

        "baseline_threshold": float(BASELINE_THRESHOLD),

        "threshold_selection": {
            "specificity_floor": float(SPECIFICITY_FLOOR),
            "candidate_grid_min": float(THRESHOLD_GRID.min()),
            "candidate_grid_max": float(THRESHOLD_GRID.max()),
            "candidate_grid_step": 0.01,
            "objective": (
                "maximize mean inner-fold Wake sensitivity subject to "
                f"mean inner-fold specificity >= {SPECIFICITY_FLOOR:.2f}"
            ),
            "tie_breaks": [
                "higher mean inner-fold specificity",
                "threshold closest to 0.50",
                "lower threshold"
            ],
            "fallback": (
                "If no candidate meets the specificity floor, maximize "
                "mean inner-fold specificity, then mean inner-fold Wake "
                "sensitivity, then closeness to 0.50, then lower threshold."
            )
        },

        "experiment": {
            "subjects": [
                int(x) for x in unique_subjects
            ],
            "n_subjects": int(
                len(unique_subjects)
            ),
            "n_epochs": int(
                len(X_matrix)
            ),
            "n_features": int(
                X_matrix.shape[1]
            ),
            "task": "Binary Wake vs Non-Wake",
            "wake_label": 1,
            "nonwake_label": 0
        },

        "original_labels": {
            "0": "Awake(W)",
            "1": "Light(N1)",
            "2": "Moderate(N2)",
            "3": "Deep(N3)",
            "4": "REM(R)"
        },

        "features": {
            "representation": "Relative spectral bandpower",
            "bands": {
                "Delta": "0.5-4 Hz",
                "Theta": "4-8 Hz",
                "Alpha": "8-12 Hz",
                "Sigma": "12-16 Hz",
                "Beta": "16-30 Hz",
                "Gamma": "30-40 Hz"
            },
            "welch_segment": "4 seconds",
            "welch_overlap": "50%"
        },

        "classifier": {
            "name": "Random Forest",
            "n_estimators": int(N_ESTIMATORS),
            "random_state": int(RANDOM_STATE),
            "n_jobs": int(N_JOBS)
        },

        "validation": {
            "outer_method": (
                "Leave-One-Subject-Out cross-validation"
            ),
            "outer_folds": int(
                len(unique_subjects)
            ),
            "inner_method": (
                "Leave-One-Subject-Out threshold selection within each outer training set"
            ),
            "inner_folds_per_outer_fold": int(
                len(unique_subjects) - 1
            ),
            "threshold_selection_uses_outer_test_labels": False,
            "outer_test_used_for_model_fitting": False
        },

        "results": {
            "baseline": {
                "mean_fold_accuracy": float(
                    np.mean(baseline_fold_accuracies)
                ),
                "std_fold_accuracy": float(
                    np.std(baseline_fold_accuracies, ddof=1)
                ),
                "mean_fold_wake_sensitivity": float(
                    np.mean(baseline_fold_sensitivities)
                ),
                "mean_fold_specificity": float(
                    np.mean(baseline_fold_specificities)
                ),
                "mean_fold_balanced_accuracy": float(
                    np.mean(baseline_fold_balanced_accuracies)
                ),
                "mean_fold_wake_precision": float(
                    np.mean(baseline_fold_precisions)
                ),
                "mean_fold_wake_f1": float(
                    np.mean(baseline_fold_f1s)
                ),
                "mean_fold_roc_auc": float(
                    np.mean(baseline_fold_roc_aucs)
                ),
                "mean_fold_pr_auc": float(
                    np.mean(baseline_fold_pr_aucs)
                ),
                "pooled_accuracy": float(
                    pooled_baseline["accuracy"]
                ),
                "pooled_balanced_accuracy": float(
                    pooled_baseline["balanced_accuracy"]
                ),
                "pooled_wake_precision": float(
                    pooled_baseline["wake_precision"]
                ),
                "pooled_wake_sensitivity": float(
                    pooled_baseline["wake_sensitivity"]
                ),
                "pooled_wake_miss_rate": float(
                    pooled_baseline["wake_miss_rate"]
                ),
                "pooled_wake_specificity": float(
                    pooled_baseline["wake_specificity"]
                ),
                "pooled_wake_f1": float(
                    pooled_baseline["wake_f1"]
                ),
                "pooled_macro_f1": float(
                    pooled_baseline["macro_f1"]
                ),
                "pooled_weighted_f1": float(
                    pooled_baseline["weighted_f1"]
                ),
                "pooled_roc_auc": float(
                    pooled_baseline["roc_auc"]
                ),
                "pooled_pr_auc": float(
                    pooled_baseline["pr_auc"]
                ),
                "true_negative": int(
                    pooled_baseline["true_negative"]
                ),
                "false_positive": int(
                    pooled_baseline["false_positive"]
                ),
                "false_negative": int(
                    pooled_baseline["false_negative"]
                ),
                "true_positive": int(
                    pooled_baseline["true_positive"]
                )
            },

            "selected": {
                "mean_fold_accuracy": float(
                    np.mean(selected_fold_accuracies)
                ),
                "std_fold_accuracy": float(
                    np.std(selected_fold_accuracies, ddof=1)
                ),
                "mean_fold_wake_sensitivity": float(
                    np.mean(selected_fold_sensitivities)
                ),
                "mean_fold_specificity": float(
                    np.mean(selected_fold_specificities)
                ),
                "mean_fold_balanced_accuracy": float(
                    np.mean(selected_fold_balanced_accuracies)
                ),
                "mean_fold_wake_precision": float(
                    np.mean(selected_fold_precisions)
                ),
                "mean_fold_wake_f1": float(
                    np.mean(selected_fold_f1s)
                ),
                "mean_fold_roc_auc": float(
                    np.mean(selected_fold_roc_aucs)
                ),
                "mean_fold_pr_auc": float(
                    np.mean(selected_fold_pr_aucs)
                ),
                "pooled_accuracy": float(
                    pooled_selected["accuracy"]
                ),
                "pooled_balanced_accuracy": float(
                    pooled_selected["balanced_accuracy"]
                ),
                "pooled_wake_precision": float(
                    pooled_selected["wake_precision"]
                ),
                "pooled_wake_sensitivity": float(
                    pooled_selected["wake_sensitivity"]
                ),
                "pooled_wake_miss_rate": float(
                    pooled_selected["wake_miss_rate"]
                ),
                "pooled_wake_specificity": float(
                    pooled_selected["wake_specificity"]
                ),
                "pooled_wake_f1": float(
                    pooled_selected["wake_f1"]
                ),
                "pooled_macro_f1": float(
                    pooled_selected["macro_f1"]
                ),
                "pooled_weighted_f1": float(
                    pooled_selected["weighted_f1"]
                ),
                "pooled_roc_auc": float(
                    pooled_selected["roc_auc"]
                ),
                "pooled_pr_auc": float(
                    pooled_selected["pr_auc"]
                ),
                "true_negative": int(
                    pooled_selected["true_negative"]
                ),
                "false_positive": int(
                    pooled_selected["false_positive"]
                ),
                "false_negative": int(
                    pooled_selected["false_negative"]
                ),
                "true_positive": int(
                    pooled_selected["true_positive"]
                )
            },

            "paired_change_selected_minus_baseline": {
                "wake_sensitivity": float(
                    pooled_selected["wake_sensitivity"]
                    - pooled_baseline["wake_sensitivity"]
                ),
                "wake_miss_rate": float(
                    pooled_selected["wake_miss_rate"]
                    - pooled_baseline["wake_miss_rate"]
                ),
                "wake_specificity": float(
                    pooled_selected["wake_specificity"]
                    - pooled_baseline["wake_specificity"]
                ),
                "balanced_accuracy": float(
                    pooled_selected["balanced_accuracy"]
                    - pooled_baseline["balanced_accuracy"]
                ),
                "wake_precision": float(
                    pooled_selected["wake_precision"]
                    - pooled_baseline["wake_precision"]
                ),
                "wake_f1": float(
                    pooled_selected["wake_f1"]
                    - pooled_baseline["wake_f1"]
                )
            },

            "selected_threshold_distribution": {
                "mean": mean_selected_threshold,
                "sd": sd_selected_threshold,
                "minimum": float(
                    np.min(
                        [row["selected_threshold"] for row in outer_fold_results]
                    )
                ),
                "maximum": float(
                    np.max(
                        [row["selected_threshold"] for row in outer_fold_results]
                    )
                )
            }
        },

        "software_environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "mne": get_package_version("mne"),
            "yasa": get_package_version("yasa"),
            "scipy": get_package_version("scipy"),
            "numpy": get_package_version("numpy"),
            "scikit_learn": get_package_version("scikit-learn")
        }
    }

    with open(
        os.path.join(
            EXPERIMENT_DIR,
            "experiment_summary.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    # ==========================================================
    # 19. UPDATE MASTER REGISTRY
    # ==========================================================

    append_registry_entry(
        summary
    )

    # ==========================================================
    # 20. FINISHED
    # ==========================================================

    print(
        "\n=========================================================="
    )
    print(
        "✅ EXP005 COMPLETE"
    )
    print(
        "=========================================================="
    )

    print(
        f"📁 Results saved to:\n{EXPERIMENT_DIR}"
    )

    print(
        "\n🔒 Experiment preserved."
    )
    print(
        "🧾 Exact code snapshot preserved."
    )
    print(
        "📊 Baseline + nested predictions, threshold scans, "
        "confusion matrices, ROC and PR data saved."
    )


# ==============================================================
# ENTRY POINT
# ==============================================================

if __name__ == "__main__":
    main()
