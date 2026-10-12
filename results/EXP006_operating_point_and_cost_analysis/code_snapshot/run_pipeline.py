import os
import csv
import json
import pickle
import sys
import time
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
    roc_curve as sklearn_roc_curve,
    precision_recall_curve as sklearn_pr_curve,
)

# ==============================================================
# EXPERIMENT CONFIGURATION
# ==============================================================

EXPERIMENT_ID = "EXP006"
EXPERIMENT_NAME = "operating_point_and_cost_analysis"

# Binary target:
# 1 = Wake
# 0 = Non-Wake
POSITIVE_CLASS = 1

# Fixed comparison operating point inherited from EXP004/EXP005.
BASELINE_THRESHOLD = 0.50

# --------------------------------------------------------------
# A. SPECIFICITY-CONSTRAINED ANALYSIS
# --------------------------------------------------------------
# These values are prespecified before looking at the EXP006
# outer-test results.
SPECIFICITY_FLOORS = [0.85, 0.90, 0.95]

# --------------------------------------------------------------
# B. COST-WEIGHTED DECISION ANALYSIS
# --------------------------------------------------------------
# Cost ratio = cost(Wake false negative) :
#              cost(Non-Wake false positive)
#
# 1:1 is included as a symmetric reference.
# 2:1, 5:1, and 10:1 represent increasingly asymmetric
# Wake-sensitive priorities.
#
# These are analysis scenarios, not empirically validated
# clinical/device costs.
WAKE_FN_TO_NONWAKE_FP_COST_RATIOS = [1.0, 2.0, 5.0, 10.0]

# Fixed, pre-specified threshold grid.
THRESHOLD_GRID = np.linspace(0.00, 1.00, 101)

# Same classifier as EXP004/EXP005.
N_ESTIMATORS = 100
RANDOM_STATE = 42
N_JOBS = -1

# Number of individual epochs used for an approximate
# single-epoch inference timing measurement per outer fold.
INFERENCE_TIMING_EPOCHS = 100

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
# GENERAL HELPERS
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

            import shutil

            shutil.copy2(
                source,
                destination
            )


def append_registry_entry(summary):
    """
    Append EXP006 to the master experiment registry.

    Because the registry is intentionally one-row-per-experiment,
    its headline metrics use the pre-specified 0.90 specificity
    scenario as the primary comparable scenario.
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

    primary = summary["primary_registry_scenario"]
    primary_results = summary["results"]["scenarios"][primary]

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
            "mean_fold_accuracy": primary_results["mean_fold_accuracy"],
            "accuracy_sd": primary_results["std_fold_accuracy"],
            "pooled_accuracy": primary_results["pooled_accuracy"],
            "macro_f1": primary_results["pooled_macro_f1"],
            "weighted_f1": primary_results["pooled_weighted_f1"]
        })


# ==============================================================
# METRICS
# ==============================================================

def metric_bundle(
    y_true,
    y_pred,
    wake_probability
):
    """
    Calculate binary metrics for one held-out subject or one
    pooled outer-of-fold prediction set.
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


# ==============================================================
# CLASSIFIER
# ==============================================================

def train_random_forest(X_train, y_train):
    """Fit the fixed Random Forest configuration."""

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
    """Extract the model probability assigned to Wake."""

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


# ==============================================================
# INNER THRESHOLD EVALUATION
# ==============================================================

def evaluate_thresholds_for_inner_subjects(
    y_true_by_subject,
    probability_by_subject
):
    """
    Evaluate every threshold on each inner validation subject.

    Returns a row-level table with subject results and a
    threshold-level table aggregated by equal-weight subject
    means.
    """

    rows = []
    summaries = []

    subject_ids = sorted(
        y_true_by_subject.keys()
    )

    for threshold in THRESHOLD_GRID:

        fold_sensitivities = []
        fold_specificities = []
        fold_balanced_accuracies = []
        fold_precisions = []
        fold_f1s = []
        fold_cost_components = {}

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

            # Per-epoch raw false-negative / false-positive counts
            # are retained so cost-weighted analyses can apply
            # different predefined ratios without retraining.
            n_epochs = len(y_true)

            fold_cost_components[subject_id] = {
                "n_epochs": int(n_epochs),
                "false_negative": int(
                    metrics["false_negative"]
                ),
                "false_positive": int(
                    metrics["false_positive"]
                )
            }

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
                ),
                "false_negative": int(
                    metrics["false_negative"]
                ),
                "false_positive": int(
                    metrics["false_positive"]
                ),
                "n_epochs": int(n_epochs)
            })

        # Equal weight for each inner subject.
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

        # Mean cost per epoch for each cost ratio.
        cost_summary = {}

        for ratio in WAKE_FN_TO_NONWAKE_FP_COST_RATIOS:

            subject_costs = []

            for subject_id in subject_ids:

                components = fold_cost_components[
                    subject_id
                ]

                numerator = (
                    ratio
                    * components["false_negative"]
                    + components["false_positive"]
                )

                denominator = components["n_epochs"]

                subject_costs.append(
                    safe_divide(
                        numerator,
                        denominator
                    )
                )

            cost_summary[f"{ratio:g}"] = float(
                np.mean(subject_costs)
            )

        summary_row = {
            "threshold": float(threshold),
            "mean_wake_sensitivity": mean_sensitivity,
            "mean_wake_specificity": mean_specificity,
            "mean_wake_miss_rate": 1.0 - mean_sensitivity,
            "mean_wake_precision": mean_precision,
            "mean_wake_f1": mean_f1,
            "mean_balanced_accuracy": mean_balanced_accuracy
        }

        for ratio in WAKE_FN_TO_NONWAKE_FP_COST_RATIOS:
            summary_row[
                f"mean_cost_ratio_{ratio:g}"
            ] = cost_summary[f"{ratio:g}"]

        summaries.append(
            summary_row
        )

    return rows, summaries


def choose_specificity_threshold(
    threshold_summaries,
    specificity_floor
):
    """
    Select threshold using only inner validation.

    Primary objective:
        maximize mean Wake sensitivity

    Constraint:
        mean inner specificity >= specificity_floor

    Deterministic tie-breaks:
        1. higher mean specificity
        2. threshold closest to 0.50
        3. lower threshold
    """

    feasible = [
        row
        for row in threshold_summaries
        if row["mean_wake_specificity"] >= specificity_floor
    ]

    if feasible:

        selected = sorted(
            feasible,
            key=lambda row: (
                -row["mean_wake_sensitivity"],
                -row["mean_wake_specificity"],
                abs(
                    row["threshold"]
                    - BASELINE_THRESHOLD
                ),
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
                abs(
                    row["threshold"]
                    - BASELINE_THRESHOLD
                ),
                row["threshold"]
            )
        )[0]

        selection_mode = (
            "fallback_maximize_specificity_then_sensitivity"
        )

    return (
        float(selected["threshold"]),
        selection_mode,
        selected
    )


def choose_cost_threshold(
    threshold_summaries,
    cost_ratio
):
    """
    Select threshold by minimizing mean inner subject-level
    cost per epoch:

        cost = ratio * FN + 1 * FP

    The outer-test subject is not involved.

    Tie-breaks:
        1. higher Wake sensitivity
        2. higher specificity
        3. threshold closest to 0.50
        4. lower threshold
    """

    field = (
        f"mean_cost_ratio_{cost_ratio:g}"
    )

    selected = sorted(
        threshold_summaries,
        key=lambda row: (
            row[field],
            -row["mean_wake_sensitivity"],
            -row["mean_wake_specificity"],
            abs(
                row["threshold"]
                - BASELINE_THRESHOLD
            ),
            row["threshold"]
        )
    )[0]

    return (
        float(selected["threshold"]),
        "cost_minimization",
        selected
    )


# ==============================================================
# COMPUTATIONAL CHARACTERIZATION
# ==============================================================

def model_resource_bundle(model):
    """
    Return model-footprint descriptors that can be measured
    directly from the fitted Random Forest.
    """

    tree_node_counts = [
        int(estimator.tree_.node_count)
        for estimator in model.estimators_
    ]

    serialized = pickle.dumps(
        model,
        protocol=pickle.HIGHEST_PROTOCOL
    )

    return {
        "n_trees": int(
            len(model.estimators_)
        ),
        "total_tree_nodes": int(
            np.sum(tree_node_counts)
        ),
        "mean_tree_nodes": float(
            np.mean(tree_node_counts)
        ),
        "min_tree_nodes": int(
            np.min(tree_node_counts)
        ),
        "max_tree_nodes": int(
            np.max(tree_node_counts)
        ),
        "serialized_model_bytes": int(
            len(serialized)
        ),
        "serialized_model_kib": float(
            len(serialized) / 1024.0
        ),
        "serialized_model_mib": float(
            len(serialized) / (1024.0 ** 2)
        )
    }


def time_single_epoch_inference(
    model,
    X_test,
    n_epochs
):
    """
    Approximate online-style single-epoch inference time by
    timing repeated one-epoch predict_proba calls after a
    one-call warm-up.

    This is a workstation measurement, not a microcontroller
    benchmark.
    """

    n = min(
        int(n_epochs),
        len(X_test)
    )

    if n == 0:
        return float("nan"), 0

    # Warm-up.
    model.predict_proba(
        X_test[0:1]
    )

    start = time.perf_counter()

    for index in range(n):
        model.predict_proba(
            X_test[index:index + 1]
        )

    elapsed = time.perf_counter() - start

    return (
        float(
            elapsed / n
        ),
        int(n)
    )


# ==============================================================
# CONFUSION-MATRIX EXPORT
# ==============================================================

def confusion_rows(
    scenario_id,
    held_out_subject,
    cm
):
    """Convert a 2x2 confusion matrix to long-form rows."""

    rows = []

    # Subject-level rows use integer subject IDs, while pooled rows
    # intentionally use the string label "POOLED". Preserve both
    # representations rather than forcing pooled output through int().
    if isinstance(held_out_subject, str) and held_out_subject.upper() == "POOLED":
        held_out_value = "POOLED"
    else:
        held_out_value = int(held_out_subject)

    for actual_index, actual_class in enumerate([0, 1]):

        for predicted_index, predicted_class in enumerate([0, 1]):

            rows.append({
                "scenario_id": scenario_id,
                "held_out_subject": held_out_value,
                "actual_class": (
                    BINARY_CLASS_NAMES[actual_class]
                ),
                "predicted_class": (
                    BINARY_CLASS_NAMES[predicted_class]
                ),
                "count": int(
                    cm[
                        actual_index,
                        predicted_index
                    ]
                )
            })

    return rows


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

    if os.path.exists(
        EXPERIMENT_DIR
    ):

        if not OVERWRITE_EXISTING:

            raise FileExistsError(
                "\n❌ EXPERIMENT ALREADY EXISTS\n\n"
                f"{EXPERIMENT_DIR}\n\n"
                "This experiment is protected from overwrite.\n"
                "Change EXPERIMENT_ID and EXPERIMENT_NAME "
                "for the next experiment."
            )

    else:

        os.makedirs(
            EXPERIMENT_DIR
        )

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
        f"📁 Results directory:\n"
        f"{EXPERIMENT_DIR}\n"
    )

    print(
        "🎯 Task: Wake vs Non-Wake"
    )

    print(
        f"🎚️ Baseline threshold: "
        f"{BASELINE_THRESHOLD:.2f}"
    )

    print(
        "🧭 Specificity floors: "
        + ", ".join(
            f"{x:.2f}"
            for x in SPECIFICITY_FLOORS
        )
    )

    print(
        "💰 Wake-FN : Non-Wake-FP cost ratios: "
        + ", ".join(
            f"{ratio:g}:1"
            for ratio in
            WAKE_FN_TO_NONWAKE_FP_COST_RATIOS
        )
    )

    print(
        "🔁 Validation: outer LOSO + inner LOSO"
    )

    print(
        "📏 Threshold grid: 0.00–1.00 in 0.01 increments\n"
    )

    # ==========================================================
    # 2. LOAD CORRECTED FEATURE MATRIX
    # ==========================================================

    cohort_subjects = [
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10
    ]

    feature_extraction_start = time.perf_counter()

    X_matrix, original_y, subject_tracking = (
        extract_cohort_data(
            cohort_subjects
        )
    )

    feature_extraction_seconds = (
        time.perf_counter()
        - feature_extraction_start
    )

    print(
        f"\n📊 Feature matrix: "
        f"{X_matrix.shape}"
    )

    print(
        f"🧬 Subjects: "
        f"{np.unique(subject_tracking)}"
    )

    # ==========================================================
    # 3. INTEGRITY CHECKS
    # ==========================================================

    if X_matrix.ndim != 2:
        raise RuntimeError(
            f"Unexpected feature matrix shape: "
            f"{X_matrix.shape}"
        )

    if X_matrix.shape[1] != 6:
        raise RuntimeError(
            f"Expected 6 features, "
            f"got {X_matrix.shape[1]}"
        )

    if not (
        len(X_matrix)
        == len(original_y)
        == len(subject_tracking)
    ):
        raise RuntimeError(
            "X, y, and subject arrays have different lengths."
        )

    if not set(
        np.unique(original_y)
    ).issubset(
        set(
            ORIGINAL_CLASS_NAMES.keys()
        )
    ):
        raise RuntimeError(
            "Unexpected original sleep-stage labels detected."
        )

    # ==========================================================
    # 4. BINARY CONVERSION
    # ==========================================================

    binary_y = (
        original_y == 0
    ).astype(int)

    wake_count = int(
        np.sum(
            binary_y == 1
        )
    )

    nonwake_count = int(
        np.sum(
            binary_y == 0
        )
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
    # 5. OUTER-LOSO STORAGE
    # ==========================================================

    unique_subjects = np.unique(
        subject_tracking
    )

    # One baseline scenario + specificity scenarios +
    # cost scenarios.
    scenario_definitions = [
        {
            "scenario_id": "baseline_0p50",
            "family": "baseline",
            "parameter": "0.50",
            "description": (
                "Fixed threshold 0.50"
            )
        }
    ]

    for floor in SPECIFICITY_FLOORS:
        scenario_definitions.append({
            "scenario_id": (
                f"specificity_floor_{int(round(floor * 100))}"
            ),
            "family": "specificity_constraint",
            "parameter": f"{floor:.2f}",
            "description": (
                "Nested threshold selected by maximizing "
                "Wake sensitivity subject to mean inner "
                f"specificity >= {floor:.2f}"
            )
        })

    for ratio in WAKE_FN_TO_NONWAKE_FP_COST_RATIOS:
        ratio_label = (
            f"{ratio:g}".replace(
                ".", "p"
            )
        )

        scenario_definitions.append({
            "scenario_id": (
                f"cost_ratio_{ratio_label}_to_1"
            ),
            "family": "cost_weighted",
            "parameter": f"{ratio:g}:1",
            "description": (
                "Nested threshold selected by minimizing "
                "mean inner per-epoch cost where Wake FN "
                f"cost = {ratio:g} and Non-Wake FP cost = 1"
            )
        })

    scenario_by_id = {
        item["scenario_id"]: item
        for item in scenario_definitions
    }

    scenario_true = {
        scenario["scenario_id"]: []
        for scenario in scenario_definitions
    }

    scenario_pred = {
        scenario["scenario_id"]: []
        for scenario in scenario_definitions
    }

    scenario_prob = {
        scenario["scenario_id"]: []
        for scenario in scenario_definitions
    }

    outer_fold_rows = []
    specificity_selection_rows = []
    cost_selection_rows = []
    inner_specificity_scan_rows = []
    inner_cost_scan_rows = []
    prediction_rows = []
    confusion_matrix_rows = []
    computational_rows = []

    # Per-scenario outer-fold metrics for summary.
    fold_metrics_by_scenario = {
        scenario["scenario_id"]: []
        for scenario in scenario_definitions
    }

    # Total timing across outer folds.
    outer_training_seconds = []
    outer_batch_inference_seconds = []
    single_epoch_inference_seconds = []

    print(
        "\n🤖 Launching EXP006 nested operating-point analysis..."
    )

    # ==========================================================
    # 6. OUTER LOSO
    # ==========================================================

    for outer_test_sub in unique_subjects:

        print(
            "\n----------------------------------------------------------"
        )

        print(
            f"⚡ OUTER held-out Subject "
            f"{outer_test_sub:02d}"
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
                f"Empty outer training set for "
                f"Subject {outer_test_sub}."
            )

        if len(X_outer_test) == 0:
            raise RuntimeError(
                f"Empty outer test set for "
                f"Subject {outer_test_sub}."
            )

        if len(np.unique(y_outer_train)) < 2:
            raise RuntimeError(
                f"Outer training data for Subject "
                f"{outer_test_sub} do not contain both classes."
            )

        if len(np.unique(y_outer_test)) < 2:
            raise RuntimeError(
                f"Outer test data for Subject "
                f"{outer_test_sub} do not contain both classes."
            )

        print(
            f"   Outer training epochs: "
            f"{len(X_outer_train)}"
        )

        print(
            f"   Outer testing epochs:  "
            f"{len(X_outer_test)}"
        )

        # ======================================================
        # 6A. INNER LOSO
        # ======================================================

        inner_y_true_by_subject = {}
        inner_probability_by_subject = {}

        inner_training_seconds = 0.0

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
                    f"Empty inner training set for "
                    f"inner subject {inner_val_sub}."
                )

            if len(X_inner_val) == 0:
                raise RuntimeError(
                    f"Empty inner validation set for "
                    f"inner subject {inner_val_sub}."
                )

            if len(np.unique(y_inner_train)) < 2:
                raise RuntimeError(
                    f"Inner training data for Subject "
                    f"{inner_val_sub} do not contain both classes."
                )

            train_start = time.perf_counter()

            model_inner = train_random_forest(
                X_inner_train,
                y_inner_train
            )

            inner_training_seconds += (
                time.perf_counter()
                - train_start
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

        (
            inner_subject_rows,
            threshold_summaries
        ) = evaluate_thresholds_for_inner_subjects(
            inner_y_true_by_subject,
            inner_probability_by_subject
        )

        # Save all inner rows twice: once in the specificity
        # analysis table and once in the cost analysis table.
        for row in inner_subject_rows:
            inner_specificity_scan_rows.append({
                "outer_held_out_subject": int(
                    outer_test_sub
                ),
                **row
            })

            inner_cost_scan_rows.append({
                "outer_held_out_subject": int(
                    outer_test_sub
                ),
                **row
            })

        # ======================================================
        # 6B. SELECT THRESHOLDS FROM INNER DATA ONLY
        # ======================================================

        selected_thresholds = {
            "baseline_0p50": BASELINE_THRESHOLD
        }

        # ---- Specificity-constrained scenarios ----

        for floor in SPECIFICITY_FLOORS:

            (
                selected_threshold,
                selection_mode,
                selected_inner_summary
            ) = choose_specificity_threshold(
                threshold_summaries,
                floor
            )

            scenario_id = (
                f"specificity_floor_{int(round(floor * 100))}"
            )

            selected_thresholds[
                scenario_id
            ] = selected_threshold

            specificity_selection_rows.append({
                "outer_held_out_subject": int(
                    outer_test_sub
                ),
                "specificity_floor": float(floor),
                "selected_threshold": float(
                    selected_threshold
                ),
                "selection_mode": selection_mode,
                "inner_mean_wake_sensitivity": float(
                    selected_inner_summary[
                        "mean_wake_sensitivity"
                    ]
                ),
                "inner_mean_wake_specificity": float(
                    selected_inner_summary[
                        "mean_wake_specificity"
                    ]
                ),
                "inner_mean_wake_miss_rate": float(
                    selected_inner_summary[
                        "mean_wake_miss_rate"
                    ]
                ),
                "inner_mean_balanced_accuracy": float(
                    selected_inner_summary[
                        "mean_balanced_accuracy"
                    ]
                ),
                "threshold_feasible": int(
                    selected_inner_summary[
                        "mean_wake_specificity"
                    ] >= floor
                )
            })

            print(
                f"   🛡️ Specificity floor {floor:.0%}"
                f" → threshold {selected_threshold:.2f}"
                f" | inner sensitivity "
                f"{selected_inner_summary['mean_wake_sensitivity']:.2%}"
                f" | inner specificity "
                f"{selected_inner_summary['mean_wake_specificity']:.2%}"
            )

        # ---- Cost-weighted scenarios ----

        for ratio in WAKE_FN_TO_NONWAKE_FP_COST_RATIOS:

            (
                selected_threshold,
                selection_mode,
                selected_inner_summary
            ) = choose_cost_threshold(
                threshold_summaries,
                ratio
            )

            ratio_label = (
                f"{ratio:g}".replace(
                    ".", "p"
                )
            )

            scenario_id = (
                f"cost_ratio_{ratio_label}_to_1"
            )

            selected_thresholds[
                scenario_id
            ] = selected_threshold

            cost_field = (
                f"mean_cost_ratio_{ratio:g}"
            )

            cost_selection_rows.append({
                "outer_held_out_subject": int(
                    outer_test_sub
                ),
                "wake_fn_to_nonwake_fp_cost_ratio": (
                    float(ratio)
                ),
                "selected_threshold": float(
                    selected_threshold
                ),
                "selection_mode": selection_mode,
                "inner_mean_cost_per_epoch": float(
                    selected_inner_summary[
                        cost_field
                    ]
                ),
                "inner_mean_wake_sensitivity": float(
                    selected_inner_summary[
                        "mean_wake_sensitivity"
                    ]
                ),
                "inner_mean_wake_specificity": float(
                    selected_inner_summary[
                        "mean_wake_specificity"
                    ]
                ),
                "inner_mean_wake_miss_rate": float(
                    selected_inner_summary[
                        "mean_wake_miss_rate"
                    ]
                ),
                "inner_mean_balanced_accuracy": float(
                    selected_inner_summary[
                        "mean_balanced_accuracy"
                    ]
                )
            })

            print(
                f"   💰 Cost ratio {ratio:g}:1"
                f" → threshold {selected_threshold:.2f}"
                f" | inner cost "
                f"{selected_inner_summary[cost_field]:.5f}"
            )

        # ======================================================
        # 6C. FIT ONE OUTER MODEL
        # ======================================================

        train_start = time.perf_counter()

        model_outer = train_random_forest(
            X_outer_train,
            y_outer_train
        )

        outer_training_time = (
            time.perf_counter()
            - train_start
        )

        outer_training_seconds.append(
            outer_training_time
        )

        # Batch prediction.
        batch_start = time.perf_counter()

        outer_wake_probability = (
            get_wake_probability(
                model_outer,
                X_outer_test
            )
        )

        outer_batch_inference_time = (
            time.perf_counter()
            - batch_start
        )

        outer_batch_inference_seconds.append(
            outer_batch_inference_time
        )

        # Approximate one-epoch online-style inference.
        (
            single_epoch_time,
            n_timed_epochs
        ) = time_single_epoch_inference(
            model_outer,
            X_outer_test,
            INFERENCE_TIMING_EPOCHS
        )

        single_epoch_inference_seconds.append(
            single_epoch_time
        )

        resource = model_resource_bundle(
            model_outer
        )

        computational_rows.append({
            "held_out_subject": int(
                outer_test_sub
            ),
            "outer_training_epochs": int(
                len(X_outer_train)
            ),
            "outer_test_epochs": int(
                len(X_outer_test)
            ),
            "inner_training_seconds_total": float(
                inner_training_seconds
            ),
            "outer_training_seconds": float(
                outer_training_time
            ),
            "batch_inference_seconds": float(
                outer_batch_inference_time
            ),
            "batch_inference_ms_per_epoch": float(
                outer_batch_inference_time
                / len(X_outer_test)
                * 1000.0
            ),
            "single_epoch_inference_seconds": float(
                single_epoch_time
            ),
            "single_epoch_inference_ms": float(
                single_epoch_time
                * 1000.0
            ),
            "single_epoch_timing_epochs": int(
                n_timed_epochs
            ),
            **resource
        })

        # ======================================================
        # 6D. APPLY ALL DECISION RULES TO SAME OUTER MODEL
        # ======================================================

        for scenario in scenario_definitions:

            scenario_id = (
                scenario["scenario_id"]
            )

            threshold = selected_thresholds[
                scenario_id
            ]

            y_pred = (
                outer_wake_probability
                >= threshold
            ).astype(int)

            metrics, cm = metric_bundle(
                y_outer_test,
                y_pred,
                outer_wake_probability
            )

            fold_metrics_by_scenario[
                scenario_id
            ].append({
                "held_out_subject": int(
                    outer_test_sub
                ),
                "threshold": float(
                    threshold
                ),
                **metrics
            })

            scenario_true[
                scenario_id
            ].extend(
                y_outer_test
            )

            scenario_pred[
                scenario_id
            ].extend(
                y_pred
            )

            scenario_prob[
                scenario_id
            ].extend(
                outer_wake_probability
            )

            confusion_matrix_rows.extend(
                confusion_rows(
                    scenario_id,
                    outer_test_sub,
                    cm
                )
            )

            # Store paired epoch-level predictions.
            for i in range(
                len(y_outer_test)
            ):

                original_stage = int(
                    original_outer_test_stages[i]
                )

                prediction_rows.append({
                    "held_out_subject": int(
                        outer_test_sub
                    ),
                    "epoch_index_within_fold": int(i),
                    "original_stage_label": original_stage,
                    "original_stage": (
                        ORIGINAL_CLASS_NAMES[
                            original_stage
                        ]
                    ),
                    "binary_true_label": int(
                        y_outer_test[i]
                    ),
                    "binary_true_class": (
                        BINARY_CLASS_NAMES[
                            int(y_outer_test[i])
                        ]
                    ),
                    "probability_wake": float(
                        outer_wake_probability[i]
                    ),
                    "scenario_id": scenario_id,
                    "scenario_family": (
                        scenario["family"]
                    ),
                    "scenario_parameter": (
                        scenario["parameter"]
                    ),
                    "threshold": float(
                        threshold
                    ),
                    "predicted_label": int(
                        y_pred[i]
                    ),
                    "predicted_class": (
                        BINARY_CLASS_NAMES[
                            int(y_pred[i])
                        ]
                    ),
                    "epoch_correct": int(
                        y_outer_test[i]
                        == y_pred[i]
                    )
                })

            # Compact outer-fold row.
            outer_fold_rows.append({
                "held_out_subject": int(
                    outer_test_sub
                ),
                "scenario_id": scenario_id,
                "scenario_family": (
                    scenario["family"]
                ),
                "scenario_parameter": (
                    scenario["parameter"]
                ),
                "selected_threshold": float(
                    threshold
                ),
                **metrics
            })

        # Console summary for the three specificity scenarios.
        print(
            f"   📊 Subject {outer_test_sub:02d} outer results:"
        )

        for floor in SPECIFICITY_FLOORS:

            scenario_id = (
                f"specificity_floor_{int(round(floor * 100))}"
            )

            metric = fold_metrics_by_scenario[
                scenario_id
            ][-1]

            print(
                f"      floor {floor:.0%}: "
                f"threshold={metric['threshold']:.2f}, "
                f"Wake sens={metric['wake_sensitivity']:.2%}, "
                f"spec={metric['wake_specificity']:.2%}"
            )

    # ==========================================================
    # 7. POOLED OUTER-OF-FOLD RESULTS
    # ==========================================================

    pooled_scenario_results = {}

    for scenario in scenario_definitions:

        scenario_id = (
            scenario["scenario_id"]
        )

        y_true = np.asarray(
            scenario_true[
                scenario_id
            ]
        )

        y_pred = np.asarray(
            scenario_pred[
                scenario_id
            ]
        )

        probabilities = np.asarray(
            scenario_prob[
                scenario_id
            ]
        )

        metrics, cm = metric_bundle(
            y_true,
            y_pred,
            probabilities
        )

        fold_rows = fold_metrics_by_scenario[
            scenario_id
        ]

        pooled_scenario_results[
            scenario_id
        ] = {
            "scenario_id": scenario_id,
            "family": scenario["family"],
            "parameter": scenario["parameter"],
            "description": scenario["description"],
            "mean_fold_accuracy": float(
                np.mean([
                    row["accuracy"]
                    for row in fold_rows
                ])
            ),
            "std_fold_accuracy": float(
                np.std([
                    row["accuracy"]
                    for row in fold_rows
                ], ddof=1)
            ),
            "mean_fold_wake_sensitivity": float(
                np.mean([
                    row["wake_sensitivity"]
                    for row in fold_rows
                ])
            ),
            "mean_fold_specificity": float(
                np.mean([
                    row["wake_specificity"]
                    for row in fold_rows
                ])
            ),
            "mean_fold_balanced_accuracy": float(
                np.mean([
                    row["balanced_accuracy"]
                    for row in fold_rows
                ])
            ),
            "mean_fold_wake_precision": float(
                np.mean([
                    row["wake_precision"]
                    for row in fold_rows
                ])
            ),
            "mean_fold_wake_f1": float(
                np.mean([
                    row["wake_f1"]
                    for row in fold_rows
                ])
            ),
            "mean_selected_threshold": float(
                np.mean([
                    row["threshold"]
                    for row in fold_rows
                ])
            ),
            "sd_selected_threshold": float(
                np.std([
                    row["threshold"]
                    for row in fold_rows
                ], ddof=1)
            ),
            "min_selected_threshold": float(
                np.min([
                    row["threshold"]
                    for row in fold_rows
                ])
            ),
            "max_selected_threshold": float(
                np.max([
                    row["threshold"]
                    for row in fold_rows
                ])
            ),
            "pooled_accuracy": float(
                metrics["accuracy"]
            ),
            "pooled_balanced_accuracy": float(
                metrics["balanced_accuracy"]
            ),
            "pooled_wake_precision": float(
                metrics["wake_precision"]
            ),
            "pooled_wake_sensitivity": float(
                metrics["wake_sensitivity"]
            ),
            "pooled_wake_miss_rate": float(
                metrics["wake_miss_rate"]
            ),
            "pooled_wake_specificity": float(
                metrics["wake_specificity"]
            ),
            "pooled_wake_f1": float(
                metrics["wake_f1"]
            ),
            "pooled_macro_f1": float(
                metrics["macro_f1"]
            ),
            "pooled_weighted_f1": float(
                metrics["weighted_f1"]
            ),
            "pooled_roc_auc": float(
                metrics["roc_auc"]
            ),
            "pooled_pr_auc": float(
                metrics["pr_auc"]
            ),
            "true_negative": int(
                metrics["true_negative"]
            ),
            "false_positive": int(
                metrics["false_positive"]
            ),
            "false_negative": int(
                metrics["false_negative"]
            ),
            "true_positive": int(
                metrics["true_positive"]
            )
        }

        confusion_matrix_rows.extend(
            confusion_rows(
                scenario_id,
                "POOLED",
                cm
            )
        )

    # ==========================================================
    # 8. SAVE SCORE-BASED ROC / PR DATA
    # ==========================================================
    # All decision scenarios use the same outer-test Wake
    # probabilities within each held-out subject. Therefore ROC-AUC
    # and PR-AUC are invariant to the decision threshold and are saved
    # once for the common outer-of-fold score set.

    baseline_probabilities = np.asarray(
        scenario_prob["baseline_0p50"]
    )

    baseline_true_for_curves = np.asarray(
        scenario_true["baseline_0p50"]
    )

    roc_fpr, roc_tpr, roc_thresholds = sklearn_roc_curve(
        baseline_true_for_curves,
        baseline_probabilities,
        pos_label=1
    )

    save_csv(
        "roc_curve.csv",
        [
            "false_positive_rate",
            "true_positive_rate",
            "threshold"
        ],
        [
            {
                "false_positive_rate": float(roc_fpr[i]),
                "true_positive_rate": float(roc_tpr[i]),
                "threshold": float(roc_thresholds[i])
            }
            for i in range(len(roc_fpr))
        ]
    )

    pr_precision, pr_recall, pr_thresholds = sklearn_pr_curve(
        baseline_true_for_curves,
        baseline_probabilities,
        pos_label=1
    )

    save_csv(
        "precision_recall_curve.csv",
        [
            "precision",
            "recall",
            "threshold"
        ],
        [
            {
                "precision": float(pr_precision[i]),
                "recall": float(pr_recall[i]),
                "threshold": (
                    float(pr_thresholds[i])
                    if i < len(pr_thresholds)
                    else ""
                )
            }
            for i in range(len(pr_precision))
        ]
    )

    # ==========================================================
    # 9. PAIRED CHANGES RELATIVE TO FIXED 0.50
    # ==========================================================

    baseline = pooled_scenario_results[
        "baseline_0p50"
    ]

    for scenario_id, result in (
        pooled_scenario_results.items()
    ):

        result["paired_change_vs_baseline"] = {
            "wake_sensitivity": (
                result["pooled_wake_sensitivity"]
                - baseline["pooled_wake_sensitivity"]
            ),
            "wake_miss_rate": (
                result["pooled_wake_miss_rate"]
                - baseline["pooled_wake_miss_rate"]
            ),
            "wake_specificity": (
                result["pooled_wake_specificity"]
                - baseline["pooled_wake_specificity"]
            ),
            "balanced_accuracy": (
                result["pooled_balanced_accuracy"]
                - baseline["pooled_balanced_accuracy"]
            ),
            "wake_precision": (
                result["pooled_wake_precision"]
                - baseline["pooled_wake_precision"]
            ),
            "wake_f1": (
                result["pooled_wake_f1"]
                - baseline["pooled_wake_f1"]
            ),
            "accuracy": (
                result["pooled_accuracy"]
                - baseline["pooled_accuracy"]
            )
        }

    # ==========================================================
    # 10. SAVE OUTPUT TABLES
    # ==========================================================

    # ---- Outer fold results ----

    save_csv(
        "outer_fold_results.csv",
        list(
            outer_fold_rows[0].keys()
        ),
        outer_fold_rows
    )

    # ---- Specificity threshold selection ----

    save_csv(
        "threshold_selection_specificity.csv",
        list(
            specificity_selection_rows[0].keys()
        ),
        specificity_selection_rows
    )

    # ---- Cost threshold selection ----

    save_csv(
        "threshold_selection_cost.csv",
        list(
            cost_selection_rows[0].keys()
        ),
        cost_selection_rows
    )

    # ---- Inner scans ----

    # Subject-level validation rows preserve the full inner-fold
    # evidence for every threshold.
    inner_subject_fields = [
        "outer_held_out_subject",
        "threshold",
        "inner_validation_subject",
        "wake_sensitivity",
        "wake_specificity",
        "wake_miss_rate",
        "wake_precision",
        "wake_f1",
        "balanced_accuracy",
        "false_negative",
        "false_positive",
        "n_epochs"
    ]

    save_csv(
        "inner_subject_threshold_scan.csv",
        inner_subject_fields,
        inner_specificity_scan_rows
    )

    # Threshold-level summaries are the actual objects used for
    # specificity-constrained and cost-weighted selection.
    # Reconstruct them deterministically from the saved row structure.
    # We generate them again here rather than relying on a transient
    # in-memory object that is not otherwise persisted.
    inner_summary_rows = []

    # Group the subject-level rows by outer subject and threshold.
    grouped = {}

    for row in inner_specificity_scan_rows:
        key = (
            int(row["outer_held_out_subject"]),
            float(row["threshold"])
        )
        grouped.setdefault(key, []).append(row)

    for (outer_subject, threshold), rows in sorted(grouped.items()):

        mean_sensitivity = float(np.mean([
            row["wake_sensitivity"] for row in rows
        ]))

        mean_specificity = float(np.mean([
            row["wake_specificity"] for row in rows
        ]))

        mean_miss_rate = float(np.mean([
            row["wake_miss_rate"] for row in rows
        ]))

        mean_precision = float(np.mean([
            row["wake_precision"] for row in rows
        ]))

        mean_f1 = float(np.mean([
            row["wake_f1"] for row in rows
        ]))

        mean_balanced_accuracy = float(np.mean([
            row["balanced_accuracy"] for row in rows
        ]))

        summary = {
            "outer_held_out_subject": int(outer_subject),
            "threshold": float(threshold),
            "mean_wake_sensitivity": mean_sensitivity,
            "mean_wake_specificity": mean_specificity,
            "mean_wake_miss_rate": mean_miss_rate,
            "mean_wake_precision": mean_precision,
            "mean_wake_f1": mean_f1,
            "mean_balanced_accuracy": mean_balanced_accuracy
        }

        for ratio in WAKE_FN_TO_NONWAKE_FP_COST_RATIOS:
            ratio_field = f"mean_cost_ratio_{ratio:g}"
            subject_costs = []

            for row in rows:
                cost = safe_divide(
                    ratio * row["false_negative"]
                    + row["false_positive"],
                    row["n_epochs"]
                )
                subject_costs.append(cost)

            summary[ratio_field] = float(
                np.mean(subject_costs)
            )

        inner_summary_rows.append(summary)

    save_csv(
        "inner_threshold_summary.csv",
        list(inner_summary_rows[0].keys()),
        inner_summary_rows
    )

    # Retain a dedicated copy for cost analysis naming so a reviewer
    # can immediately identify which rows contain the cost objective.
    save_csv(
        "inner_cost_scan.csv",
        list(inner_summary_rows[0].keys()),
        inner_summary_rows
    )

    # ---- All pooled/outer predictions ----

    save_csv(
        "loso_predictions.csv",
        list(
            prediction_rows[0].keys()
        ),
        prediction_rows
    )

    # ---- Long-form confusion matrices ----

    save_csv(
        "confusion_matrices_long.csv",
        list(
            confusion_matrix_rows[0].keys()
        ),
        confusion_matrix_rows
    )

    # ---- Computational characterization ----

    save_csv(
        "computational_resources.csv",
        list(
            computational_rows[0].keys()
        ),
        computational_rows
    )

    # ==========================================================
    # 11. CLASSIFICATION REPORTS
    # ==========================================================

    report_blocks = []

    for scenario in scenario_definitions:

        scenario_id = (
            scenario["scenario_id"]
        )

        y_true = np.asarray(
            scenario_true[
                scenario_id
            ]
        )

        y_pred = np.asarray(
            scenario_pred[
                scenario_id
            ]
        )

        report = classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=[
                "Non-Wake",
                "Wake"
            ],
            zero_division=0
        )

        report_blocks.append(
            "\n"
            + "=" * 76
            + "\n"
            + f"SCENARIO: {scenario_id}\n"
            + f"Family: {scenario['family']}\n"
            + f"Parameter: {scenario['parameter']}\n"
            + f"Description: {scenario['description']}\n\n"
            + report
        )

    with open(
        os.path.join(
            EXPERIMENT_DIR,
            "classification_reports.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"{EXPERIMENT_ID}: {EXPERIMENT_NAME}\n"
        )

        f.write(
            "=" * 76
            + "\n"
        )

        f.write(
            "\n".join(
                report_blocks
            )
        )

    # ==========================================================
    # 12. COMPUTATIONAL SUMMARY
    # ==========================================================

    computational_summary = {
        "feature_extraction_seconds_total": float(
            feature_extraction_seconds
        ),
        "feature_extraction_minutes_total": float(
            feature_extraction_seconds / 60.0
        ),
        "outer_training_seconds_mean": float(
            np.mean(
                outer_training_seconds
            )
        ),
        "outer_training_seconds_sd": float(
            np.std(
                outer_training_seconds,
                ddof=1
            )
        ),
        "outer_batch_inference_seconds_mean": float(
            np.mean(
                outer_batch_inference_seconds
            )
        ),
        "outer_batch_inference_ms_per_epoch_mean": float(
            np.mean([
                value / len(
                    X_matrix[
                        subject_tracking
                        == subject
                    ]
                ) * 1000.0
                for value, subject
                in zip(
                    outer_batch_inference_seconds,
                    unique_subjects
                )
            ])
        ),
        "single_epoch_inference_ms_mean": float(
            np.mean(
                np.asarray(
                    single_epoch_inference_seconds
                ) * 1000.0
            )
        ),
        "single_epoch_inference_ms_sd": float(
            np.std(
                np.asarray(
                    single_epoch_inference_seconds
                ) * 1000.0,
                ddof=1
            )
        ),
        "serialized_model_kib_mean": float(
            np.mean([
                row["serialized_model_kib"]
                for row in computational_rows
            ])
        ),
        "serialized_model_kib_sd": float(
            np.std([
                row["serialized_model_kib"]
                for row in computational_rows
            ], ddof=1)
        ),
        "total_tree_nodes_mean": float(
            np.mean([
                row["total_tree_nodes"]
                for row in computational_rows
            ])
        ),
        "mean_tree_nodes_mean": float(
            np.mean([
                row["mean_tree_nodes"]
                for row in computational_rows
            ])
        ),
        "max_tree_nodes_mean": float(
            np.mean([
                row["max_tree_nodes"]
                for row in computational_rows
            ])
        )
    }

    # ==========================================================
    # 13. FINAL CONSOLE SUMMARY
    # ==========================================================

    print(
        "\n=========================================================="
    )

    print(
        "📊 EXP006 FINAL RESULTS"
    )

    print(
        "=========================================================="
    )

    for scenario in scenario_definitions:

        result = pooled_scenario_results[
            scenario["scenario_id"]
        ]

        print(
            f"\n{scenario['scenario_id']}"
        )

        print(
            f"   Mean threshold: "
            f"{result['mean_selected_threshold']:.4f}"
        )

        print(
            f"   Pooled accuracy: "
            f"{result['pooled_accuracy']:.4%}"
        )

        print(
            f"   Pooled Wake sensitivity: "
            f"{result['pooled_wake_sensitivity']:.4%}"
        )

        print(
            f"   Pooled Wake miss rate: "
            f"{result['pooled_wake_miss_rate']:.4%}"
        )

        print(
            f"   Pooled specificity: "
            f"{result['pooled_wake_specificity']:.4%}"
        )

        print(
            f"   Pooled Wake F1: "
            f"{result['pooled_wake_f1']:.4%}"
        )

        if (
            scenario["scenario_id"]
            != "baseline_0p50"
        ):

            delta = (
                result[
                    "paired_change_vs_baseline"
                ]
            )

            print(
                f"   Δ Wake sensitivity vs baseline: "
                f"{delta['wake_sensitivity']:+.4%}"
            )

            print(
                f"   Δ Wake miss rate vs baseline: "
                f"{delta['wake_miss_rate']:+.4%}"
            )

            print(
                f"   Δ specificity vs baseline: "
                f"{delta['wake_specificity']:+.4%}"
            )

    print(
        "\n=========================================================="
    )

    print(
        "💻 COMPUTATIONAL CHARACTERIZATION"
    )

    print(
        "=========================================================="
    )

    print(
        f"   Feature extraction: "
        f"{computational_summary['feature_extraction_seconds_total']:.2f} s"
    )

    print(
        f"   Mean outer training time: "
        f"{computational_summary['outer_training_seconds_mean']:.4f} s"
    )

    print(
        f"   Mean single-epoch inference: "
        f"{computational_summary['single_epoch_inference_ms_mean']:.4f} ms"
    )

    print(
        f"   Mean serialized model size: "
        f"{computational_summary['serialized_model_kib_mean']:.2f} KiB"
    )

    print(
        f"   Mean total RF tree nodes: "
        f"{computational_summary['total_tree_nodes_mean']:.0f}"
    )

    # ==========================================================
    # 14. EXPERIMENT SUMMARY JSON
    # ==========================================================

    primary_registry_scenario = (
        "specificity_floor_90"
    )

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "timestamp": run_timestamp,

        "hypotheses": [
            (
                "The Wake-sensitive operating-point behavior observed "
                "in EXP005 is robust across different predefined "
                "specificity constraints under nested subject-independent "
                "validation."
            ),
            (
                "Explicitly weighting Wake false negatives more heavily "
                "than Non-Wake false positives shifts the selected "
                "operating point toward greater Wake sensitivity in a "
                "predictable way."
            ),
            (
                "The fixed six-feature single-channel Random Forest "
                "framework has a measurable computational footprint that "
                "can be characterized from the workstation implementation, "
                "without claiming hardware validation."
            )
        ],

        "primary_metrics": [
            "Wake sensitivity / recall",
            "Wake miss rate",
            "Wake specificity"
        ],

        "secondary_metrics": [
            "Wake precision",
            "Wake F1",
            "balanced accuracy",
            "accuracy",
            "ROC-AUC",
            "PR-AUC"
        ],

        "experiment": {
            "subjects": [
                int(x)
                for x in unique_subjects
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
            "n_estimators": int(
                N_ESTIMATORS
            ),
            "random_state": int(
                RANDOM_STATE
            ),
            "n_jobs": int(
                N_JOBS
            )
        },

        "decision_analysis": {
            "baseline_threshold": float(
                BASELINE_THRESHOLD
            ),
            "specificity_floors": [
                float(x)
                for x in SPECIFICITY_FLOORS
            ],
            "cost_ratios_wake_fn_to_nonwake_fp": [
                float(x)
                for x
                in WAKE_FN_TO_NONWAKE_FP_COST_RATIOS
            ],
            "threshold_grid_min": float(
                THRESHOLD_GRID.min()
            ),
            "threshold_grid_max": float(
                THRESHOLD_GRID.max()
            ),
            "threshold_grid_step": 0.01,
            "specificity_selection_objective": (
                "maximize mean inner-subject Wake sensitivity "
                "subject to mean inner-subject specificity >= floor"
            ),
            "cost_selection_objective": (
                "minimize mean inner-subject cost per epoch where "
                "cost = ratio * Wake FN + 1 * Non-Wake FP"
            ),
            "cost_analysis_note": (
                "Cost ratios are predefined sensitivity-analysis "
                "scenarios, not empirically validated clinical or "
                "device costs."
            )
        },

        "validation": {
            "outer_method": (
                "Leave-One-Subject-Out cross-validation"
            ),
            "outer_folds": int(
                len(unique_subjects)
            ),
            "inner_method": (
                "Leave-One-Subject-Out threshold selection "
                "within each outer training set"
            ),
            "inner_folds_per_outer_fold": int(
                len(unique_subjects) - 1
            ),
            "outer_test_labels_used_for_threshold_selection": False,
            "outer_test_used_for_model_fitting": False
        },

        "results": {
            "scenarios": pooled_scenario_results
        },

        "computational_resources": computational_summary,

        "primary_registry_scenario": (
            primary_registry_scenario
        ),

        "software_environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "mne": get_package_version("mne"),
            "yasa": get_package_version("yasa"),
            "scipy": get_package_version("scipy"),
            "numpy": get_package_version("numpy"),
            "scikit_learn": get_package_version(
                "scikit-learn"
            )
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
    # 15. SAVE COMPUTATIONAL PROTOCOL TEXT
    # ==========================================================

    with open(
        os.path.join(
            EXPERIMENT_DIR,
            "computational_note.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "EXP006 Computational Characterization\n"
        )

        f.write(
            "=====================================\n\n"
        )

        f.write(
            "This analysis reports workstation-measured timing and "
            "serialized Random Forest model footprint. It is not a "
            "hardware benchmark and does not establish compatibility "
            "with any specific microcontroller or wearable device.\n\n"
        )

        f.write(
            "Measurements:\n"
        )

        f.write(
            "- total feature-extraction wall-clock time\n"
        )

        f.write(
            "- outer-fold Random Forest training time\n"
        )

        f.write(
            "- batch inference time on outer-test epochs\n"
        )

        f.write(
            "- approximate single-epoch predict_proba time after warm-up\n"
        )

        f.write(
            "- serialized model size\n"
        )

        f.write(
            "- Random Forest tree-node counts\n"
        )

        f.write(
            "\n"
        )

        f.write(
            "Environment:\n"
        )

        f.write(
            f"{platform.platform()}\n"
        )

    # ==========================================================
    # 16. MASTER REGISTRY
    # ==========================================================

    append_registry_entry(
        summary
    )

    # ==========================================================
    # 17. FINISHED
    # ==========================================================

    print(
        "\n=========================================================="
    )

    print(
        "✅ EXP006 COMPLETE"
    )

    print(
        "=========================================================="
    )

    print(
        f"📁 Results saved to:\n"
        f"{EXPERIMENT_DIR}"
    )

    print(
        "\n🔒 Experiment preserved."
    )

    print(
        "🧾 Exact code snapshot preserved."
    )

    print(
        "📊 Specificity, cost, predictions, confusion matrices, "
        "and computational-characterization outputs saved."
    )


# ==============================================================
# ENTRY POINT
# ==============================================================

if __name__ == "__main__":
    main()
