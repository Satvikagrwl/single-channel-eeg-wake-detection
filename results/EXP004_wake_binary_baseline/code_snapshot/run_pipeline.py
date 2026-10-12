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
    precision_recall_curve
)


# ==============================================================
# EXPERIMENT CONFIGURATION
# ==============================================================

EXPERIMENT_ID = "EXP004"

EXPERIMENT_NAME = "wake_binary_baseline"

# Positive class:
# 1 = Wake
# 0 = Non-Wake
POSITIVE_CLASS = 1

# Baseline probability threshold.
# Experiment 5 will investigate alternative thresholds.
WAKE_THRESHOLD = 0.50

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
# ORIGINAL FIVE-CLASS DEFINITIONS
# ==============================================================

ORIGINAL_CLASS_NAMES = {
    0: "Awake(W)",
    1: "Light(N1)",
    2: "Moderate(N2)",
    3: "Deep(N3)",
    4: "REM(R)"
}


# ==============================================================
# BINARY DEFINITIONS
# ==============================================================

BINARY_CLASS_NAMES = {
    0: "Non-Wake",
    1: "Wake"
}

BINARY_CLASS_IDS = [0, 1]


# ==============================================================
# HELPER FUNCTIONS
# ==============================================================

def save_csv(filename, fieldnames, rows):
    """
    Save dictionaries as a CSV inside the experiment folder.
    """

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
    """
    Safely obtain installed package version.
    """

    try:
        return importlib.metadata.version(
            package_name
        )

    except Exception:
        return "unknown"


def save_code_snapshot():
    """
    Preserve the exact code used for this experiment.
    """

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

    The registry deliberately keeps the same broad columns used
    by earlier experiments so that EXP003 and EXP004 remain in
    one machine-readable table.
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
            "validation": summary["validation"]["method"],
            "mean_fold_accuracy": summary["results"]["mean_fold_accuracy"],
            "accuracy_sd": summary["results"]["std_fold_accuracy"],
            "pooled_accuracy": summary["results"]["pooled_accuracy"],
            "macro_f1": summary["results"]["pooled_macro_f1"],
            "weighted_f1": summary["results"]["pooled_weighted_f1"]
        })


def safe_divide(numerator, denominator):
    """
    Avoid division-by-zero errors.
    """

    if denominator == 0:
        return 0.0

    return float(
        numerator / denominator
    )


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
                "Change EXPERIMENT_ID and EXPERIMENT_NAME "
                "for the next experiment."
            )

    else:

        os.makedirs(
            EXPERIMENT_DIR
        )

    # ==========================================================
    # 2. SNAPSHOT THE EXACT CODE
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
        f"📁 Results directory:\n"
        f"{EXPERIMENT_DIR}\n"
    )

    print(
        "🎯 Task: Wake vs Non-Wake"
    )

    print(
        f"🎚️ Baseline Wake probability threshold: "
        f"{WAKE_THRESHOLD:.2f}\n"
    )

    # ==========================================================
    # 3. LOAD THE SAME CORRECTED FEATURE MATRIX
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
        f"\n📊 Feature matrix: "
        f"{X_matrix.shape}"
    )

    print(
        f"🧬 Subjects: "
        f"{np.unique(subject_tracking)}"
    )

    # ==========================================================
    # 4. INTEGRITY CHECKS
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
            "X, y, and subject arrays have "
            "different lengths."
        )

    if not set(np.unique(original_y)).issubset(
        set(ORIGINAL_CLASS_NAMES.keys())
    ):
        raise RuntimeError(
            "Unexpected original sleep-stage labels detected."
        )

    # ==========================================================
    # 5. CONVERT FIVE-CLASS LABELS TO BINARY LABELS
    # ==========================================================
    #
    # Wake = 1
    # N1/N2/N3/REM = 0
    #
    # This is the ONLY target-definition change from EXP003.
    #

    binary_y = (
        original_y == 0
    ).astype(int)

    print(
        "\n📌 Binary class distribution:"
    )

    wake_count = int(
        np.sum(binary_y == 1)
    )

    nonwake_count = int(
        np.sum(binary_y == 0)
    )

    print(
        f"   Wake:     {wake_count}"
    )

    print(
        f"   Non-Wake: {nonwake_count}"
    )

    # ==========================================================
    # 6. STORAGE
    # ==========================================================

    all_y_true = []
    all_y_pred = []
    all_wake_probabilities = []

    fold_results = []
    fold_class_metrics = []
    fold_class_distributions = []
    prediction_rows = []

    fold_wake_sensitivities = []
    fold_specificities = []
    fold_balanced_accuracies = []
    fold_wake_precisions = []
    fold_wake_f1s = []
    fold_roc_aucs = []
    fold_pr_aucs = []

    overall_confusion = np.zeros(
        (2, 2),
        dtype=int
    )

    # ==========================================================
    # 7. LOSO VALIDATION
    # ==========================================================

    unique_subjects = np.unique(
        subject_tracking
    )

    print(
        "\n🤖 Launching binary Wake-vs-Non-Wake "
        "LOSO validation..."
    )

    for test_sub in unique_subjects:

        print(
            "\n--------------------------------------------------"
        )

        print(
            f"⚡ Held-out Subject {test_sub:02d}"
        )

        # ------------------------------------------------------
        # Subject-level split
        # ------------------------------------------------------

        train_mask = (
            subject_tracking != test_sub
        )

        test_mask = (
            subject_tracking == test_sub
        )

        X_train = X_matrix[
            train_mask
        ]

        y_train = binary_y[
            train_mask
        ]

        X_test = X_matrix[
            test_mask
        ]

        y_test = binary_y[
            test_mask
        ]

        original_test_stages = original_y[
            test_mask
        ]

        if len(X_train) == 0:
            raise RuntimeError(
                f"Empty training set for Subject "
                f"{test_sub}."
            )

        if len(X_test) == 0:
            raise RuntimeError(
                f"Empty test set for Subject "
                f"{test_sub}."
            )

        # ------------------------------------------------------
        # Make sure both binary classes are represented
        # in training and testing.
        # ------------------------------------------------------

        if len(np.unique(y_train)) < 2:
            raise RuntimeError(
                f"Training data for Subject "
                f"{test_sub} does not contain both "
                f"Wake and Non-Wake classes."
            )

        if len(np.unique(y_test)) < 2:
            raise RuntimeError(
                f"Testing data for Subject "
                f"{test_sub} does not contain both "
                f"Wake and Non-Wake classes."
            )

        print(
            f"   Training epochs: {len(X_train)}"
        )

        print(
            f"   Testing epochs:  {len(X_test)}"
        )

        # ======================================================
        # Random Forest
        # ======================================================

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            X_train,
            y_train
        )

        # ======================================================
        # Wake probability
        # ======================================================

        probability_matrix = model.predict_proba(
            X_test
        )

        wake_column = np.where(
            model.classes_ == POSITIVE_CLASS
        )[0]

        if len(wake_column) != 1:
            raise RuntimeError(
                "Could not identify the Wake probability "
                "column."
            )

        wake_probability = (
            probability_matrix[
                :,
                wake_column[0]
            ]
        )

        # ======================================================
        # Explicit 0.50 baseline threshold
        # ======================================================

        y_pred = (
            wake_probability
            >= WAKE_THRESHOLD
        ).astype(int)

        # ======================================================
        # Fold metrics
        # ======================================================

        fold_accuracy = accuracy_score(
            y_test,
            y_pred
        )

        fold_balanced_accuracy = (
            balanced_accuracy_score(
                y_test,
                y_pred
            )
        )

        fold_wake_precision = (
            precision_score(
                y_test,
                y_pred,
                pos_label=1,
                zero_division=0
            )
        )

        fold_wake_sensitivity = (
            recall_score(
                y_test,
                y_pred,
                pos_label=1,
                zero_division=0
            )
        )

        fold_wake_f1 = (
            f1_score(
                y_test,
                y_pred,
                pos_label=1,
                zero_division=0
            )
        )

        # ------------------------------------------------------
        # Confusion matrix
        #
        # [[TN, FP],
        #  [FN, TP]]
        # ------------------------------------------------------

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        )

        tn, fp, fn, tp = (
            cm.ravel()
        )

        fold_specificity = safe_divide(
            tn,
            tn + fp
        )

        fold_wake_miss_rate = (
            1.0
            - fold_wake_sensitivity
        )

        # ------------------------------------------------------
        # ROC / PR metrics
        # ------------------------------------------------------

        fold_roc_auc = roc_auc_score(
            y_test,
            wake_probability
        )

        fold_pr_auc = (
            average_precision_score(
                y_test,
                wake_probability
            )
        )

        # ------------------------------------------------------
        # Store fold metrics
        # ------------------------------------------------------

        fold_results.append({
            "held_out_subject": int(test_sub),
            "training_epochs": int(len(X_train)),
            "test_epochs": int(len(X_test)),
            "accuracy": float(fold_accuracy),
            "balanced_accuracy": float(
                fold_balanced_accuracy
            ),
            "wake_precision": float(
                fold_wake_precision
            ),
            "wake_sensitivity": float(
                fold_wake_sensitivity
            ),
            "wake_miss_rate": float(
                fold_wake_miss_rate
            ),
            "wake_specificity": float(
                fold_specificity
            ),
            "wake_f1": float(
                fold_wake_f1
            ),
            "roc_auc": float(
                fold_roc_auc
            ),
            "pr_auc": float(
                fold_pr_auc
            ),
            "true_wake": int(
                np.sum(y_test == 1)
            ),
            "true_nonwake": int(
                np.sum(y_test == 0)
            ),
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        })

        fold_wake_sensitivities.append(
            fold_wake_sensitivity
        )

        fold_specificities.append(
            fold_specificity
        )

        fold_balanced_accuracies.append(
            fold_balanced_accuracy
        )

        fold_wake_precisions.append(
            fold_wake_precision
        )

        fold_wake_f1s.append(
            fold_wake_f1
        )

        fold_roc_aucs.append(
            fold_roc_auc
        )

        fold_pr_aucs.append(
            fold_pr_auc
        )

        print(
            f"   Accuracy:            {fold_accuracy:.2%}"
        )

        print(
            f"   Wake sensitivity:    "
            f"{fold_wake_sensitivity:.2%}"
        )

        print(
            f"   Wake specificity:    "
            f"{fold_specificity:.2%}"
        )

        print(
            f"   Wake precision:      "
            f"{fold_wake_precision:.2%}"
        )

        print(
            f"   Balanced accuracy:   "
            f"{fold_balanced_accuracy:.2%}"
        )

        print(
            f"   ROC-AUC:             "
            f"{fold_roc_auc:.4f}"
        )

        print(
            f"   PR-AUC:              "
            f"{fold_pr_auc:.4f}"
        )

        # ======================================================
        # Class metrics
        # ======================================================

        report = classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=[
                "Non-Wake",
                "Wake"
            ],
            output_dict=True,
            zero_division=0
        )

        for class_id in [0, 1]:

            class_name = BINARY_CLASS_NAMES[
                class_id
            ]

            fold_class_metrics.append({
                "held_out_subject": int(test_sub),
                "class_id": int(class_id),
                "class_name": class_name,
                "precision": float(
                    report[class_name]["precision"]
                ),
                "recall": float(
                    report[class_name]["recall"]
                ),
                "f1_score": float(
                    report[class_name]["f1-score"]
                ),
                "support": int(
                    report[class_name]["support"]
                )
            })

        # ======================================================
        # Binary class distributions
        # ======================================================

        for class_id in [0, 1]:

            fold_class_distributions.append({
                "held_out_subject": int(test_sub),
                "class_id": int(class_id),
                "class_name": BINARY_CLASS_NAMES[
                    class_id
                ],
                "training_count": int(
                    np.sum(
                        y_train == class_id
                    )
                ),
                "test_true_count": int(
                    np.sum(
                        y_test == class_id
                    )
                ),
                "test_predicted_count": int(
                    np.sum(
                        y_pred == class_id
                    )
                )
            })

        # ======================================================
        # Save every out-of-fold prediction
        # ======================================================

        for i in range(
            len(y_test)
        ):

            original_stage = int(
                original_test_stages[i]
            )

            prediction_rows.append({
                "held_out_subject": int(test_sub),
                "epoch_index_within_fold": int(i),

                "original_stage_label": original_stage,

                "original_stage": ORIGINAL_CLASS_NAMES[
                    original_stage
                ],

                "binary_true_label": int(
                    y_test[i]
                ),

                "binary_true_class": BINARY_CLASS_NAMES[
                    int(y_test[i])
                ],

                "binary_predicted_label": int(
                    y_pred[i]
                ),

                "binary_predicted_class": BINARY_CLASS_NAMES[
                    int(y_pred[i])
                ],

                "probability_wake": float(
                    wake_probability[i]
                ),

                "baseline_threshold": float(
                    WAKE_THRESHOLD
                ),

                "epoch_correct": int(
                    y_test[i] == y_pred[i]
                )
            })

        # ======================================================
        # Save pooled arrays
        # ======================================================

        all_y_true.extend(
            y_test
        )

        all_y_pred.extend(
            y_pred
        )

        all_wake_probabilities.extend(
            wake_probability
        )

        overall_confusion += cm

        # ======================================================
        # Save subject-specific confusion matrix
        # ======================================================

        subject_confusion_rows = []

        for actual_index, actual_class in enumerate(
            [0, 1]
        ):

            for predicted_index, predicted_class in enumerate(
                [0, 1]
            ):

                subject_confusion_rows.append({
                    "actual_class": BINARY_CLASS_NAMES[
                        actual_class
                    ],
                    "predicted_class": BINARY_CLASS_NAMES[
                        predicted_class
                    ],
                    "count": int(
                        cm[
                            actual_index,
                            predicted_index
                        ]
                    )
                })

        save_csv(
            f"confusion_matrix_subject_{test_sub:02d}.csv",

            [
                "actual_class",
                "predicted_class",
                "count"
            ],

            subject_confusion_rows
        )

    # ==========================================================
    # 8. POOLED OUT-OF-FOLD RESULTS
    # ==========================================================

    all_y_true = np.asarray(
        all_y_true
    )

    all_y_pred = np.asarray(
        all_y_pred
    )

    all_wake_probabilities = np.asarray(
        all_wake_probabilities
    )

    pooled_accuracy = accuracy_score(
        all_y_true,
        all_y_pred
    )

    pooled_balanced_accuracy = (
        balanced_accuracy_score(
            all_y_true,
            all_y_pred
        )
    )

    pooled_wake_precision = (
        precision_score(
            all_y_true,
            all_y_pred,
            pos_label=1,
            zero_division=0
        )
    )

    pooled_wake_sensitivity = (
        recall_score(
            all_y_true,
            all_y_pred,
            pos_label=1,
            zero_division=0
        )
    )

    pooled_wake_f1 = (
        f1_score(
            all_y_true,
            all_y_pred,
            pos_label=1,
            zero_division=0
        )
    )

    pooled_macro_f1 = (
        f1_score(
            all_y_true,
            all_y_pred,
            labels=[0, 1],
            average="macro",
            zero_division=0
        )
    )

    pooled_weighted_f1 = (
        f1_score(
            all_y_true,
            all_y_pred,
            labels=[0, 1],
            average="weighted",
            zero_division=0
        )
    )

    pooled_roc_auc = roc_auc_score(
        all_y_true,
        all_wake_probabilities
    )

    pooled_pr_auc = (
        average_precision_score(
            all_y_true,
            all_wake_probabilities
        )
    )

    pooled_cm = confusion_matrix(
        all_y_true,
        all_y_pred,
        labels=[0, 1]
    )

    pooled_tn, pooled_fp, pooled_fn, pooled_tp = (
        pooled_cm.ravel()
    )

    pooled_specificity = safe_divide(
        pooled_tn,
        pooled_tn + pooled_fp
    )

    pooled_wake_miss_rate = (
        1.0
        - pooled_wake_sensitivity
    )

    # ==========================================================
    # 9. FINAL REPORT
    # ==========================================================

    final_report = classification_report(
        all_y_true,
        all_y_pred,
        labels=[0, 1],
        target_names=[
            "Non-Wake",
            "Wake"
        ],
        zero_division=0
    )

    print(
        "\n=========================================================="
    )

    print(
        "📊 EXP004 FINAL RESULTS"
    )

    print(
        "=========================================================="
    )

    print(
        f"Mean fold accuracy:       "
        f"{np.mean([x['accuracy'] for x in fold_results]):.4%}"
    )

    print(
        f"Accuracy SD:              "
        f"{np.std([x['accuracy'] for x in fold_results], ddof=1):.4%}"
    )

    print(
        f"Mean fold Wake sensitivity:"
        f" {np.mean(fold_wake_sensitivities):.4%}"
    )

    print(
        f"Mean fold specificity:     "
        f"{np.mean(fold_specificities):.4%}"
    )

    print(
        f"Mean fold balanced accuracy:"
        f" {np.mean(fold_balanced_accuracies):.4%}"
    )

    print(
        f"Pooled accuracy:           "
        f"{pooled_accuracy:.4%}"
    )

    print(
        f"Pooled Wake sensitivity:   "
        f"{pooled_wake_sensitivity:.4%}"
    )

    print(
        f"Pooled Wake miss rate:     "
        f"{pooled_wake_miss_rate:.4%}"
    )

    print(
        f"Pooled Wake specificity:   "
        f"{pooled_specificity:.4%}"
    )

    print(
        f"Pooled Wake precision:     "
        f"{pooled_wake_precision:.4%}"
    )

    print(
        f"Pooled Wake F1:            "
        f"{pooled_wake_f1:.4%}"
    )

    print(
        f"Pooled Macro-F1:           "
        f"{pooled_macro_f1:.4%}"
    )

    print(
        f"Pooled ROC-AUC:            "
        f"{pooled_roc_auc:.4f}"
    )

    print(
        f"Pooled PR-AUC:             "
        f"{pooled_pr_auc:.4f}"
    )

    print(
        "\nFinal classification report:\n"
    )

    print(
        final_report
    )

    print(
        "\nPooled confusion matrix:"
    )

    print(
        pooled_cm
    )

    # ==========================================================
    # 10. SAVE FOLD RESULTS
    # ==========================================================

    save_csv(
        "fold_results.csv",

        [
            "held_out_subject",
            "training_epochs",
            "test_epochs",
            "accuracy",
            "balanced_accuracy",
            "wake_precision",
            "wake_sensitivity",
            "wake_miss_rate",
            "wake_specificity",
            "wake_f1",
            "roc_auc",
            "pr_auc",
            "true_wake",
            "true_nonwake",
            "true_negative",
            "false_positive",
            "false_negative",
            "true_positive"
        ],

        fold_results
    )

    # ==========================================================
    # 11. SAVE CLASS METRICS
    # ==========================================================

    save_csv(
        "fold_class_metrics.csv",

        [
            "held_out_subject",
            "class_id",
            "class_name",
            "precision",
            "recall",
            "f1_score",
            "support"
        ],

        fold_class_metrics
    )

    # ==========================================================
    # 12. SAVE CLASS DISTRIBUTIONS
    # ==========================================================

    save_csv(
        "fold_class_distributions.csv",

        [
            "held_out_subject",
            "class_id",
            "class_name",
            "training_count",
            "test_true_count",
            "test_predicted_count"
        ],

        fold_class_distributions
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
            "binary_predicted_label",
            "binary_predicted_class",
            "probability_wake",
            "baseline_threshold",
            "epoch_correct"
        ],

        prediction_rows
    )

    # ==========================================================
    # 14. SAVE OVERALL CONFUSION MATRIX
    # ==========================================================

    overall_confusion_rows = []

    for actual_index, actual_class in enumerate(
        [0, 1]
    ):

        for predicted_index, predicted_class in enumerate(
            [0, 1]
        ):

            overall_confusion_rows.append({
                "actual_class": BINARY_CLASS_NAMES[
                    actual_class
                ],
                "predicted_class": BINARY_CLASS_NAMES[
                    predicted_class
                ],
                "count": int(
                    overall_confusion[
                        actual_index,
                        predicted_index
                    ]
                )
            })

    save_csv(
        "confusion_matrix_overall.csv",

        [
            "actual_class",
            "predicted_class",
            "count"
        ],

        overall_confusion_rows
    )

    # ==========================================================
    # 15. SAVE ROC CURVE DATA
    # ==========================================================

    fpr, tpr, roc_thresholds = roc_curve(
        all_y_true,
        all_wake_probabilities,
        pos_label=1
    )

    roc_rows = []

    for i in range(
        len(fpr)
    ):

        roc_rows.append({
            "false_positive_rate": float(
                fpr[i]
            ),
            "true_positive_rate": float(
                tpr[i]
            ),
            "threshold": float(
                roc_thresholds[i]
            )
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
    # 16. SAVE PRECISION-RECALL CURVE DATA
    # ==========================================================

    precision_values, recall_values, pr_thresholds = (
        precision_recall_curve(
            all_y_true,
            all_wake_probabilities,
            pos_label=1
        )
    )

    pr_rows = []

    for i in range(
        len(precision_values)
    ):

        threshold = (
            float(pr_thresholds[i])
            if i < len(pr_thresholds)
            else ""
        )

        pr_rows.append({
            "precision": float(
                precision_values[i]
            ),
            "recall": float(
                recall_values[i]
            ),
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

    mean_fold_accuracy = float(
        np.mean(
            [
                x["accuracy"]
                for x in fold_results
            ]
        )
    )

    std_fold_accuracy = float(
        np.std(
            [
                x["accuracy"]
                for x in fold_results
            ],
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
            f"{EXPERIMENT_ID}: "
            f"{EXPERIMENT_NAME}\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            "Task: Wake vs Non-Wake\n"
        )

        f.write(
            "Positive class: Wake\n"
        )

        f.write(
            f"Baseline threshold: "
            f"{WAKE_THRESHOLD:.2f}\n\n"
        )

        f.write(
            f"Mean fold accuracy: "
            f"{mean_fold_accuracy:.6f}\n"
        )

        f.write(
            f"Fold accuracy SD: "
            f"{std_fold_accuracy:.6f}\n"
        )

        f.write(
            f"Pooled accuracy: "
            f"{pooled_accuracy:.6f}\n"
        )

        f.write(
            f"Pooled Wake sensitivity: "
            f"{pooled_wake_sensitivity:.6f}\n"
        )

        f.write(
            f"Pooled Wake miss rate: "
            f"{pooled_wake_miss_rate:.6f}\n"
        )

        f.write(
            f"Pooled Wake specificity: "
            f"{pooled_specificity:.6f}\n"
        )

        f.write(
            f"Pooled Wake precision: "
            f"{pooled_wake_precision:.6f}\n"
        )

        f.write(
            f"Pooled Wake F1: "
            f"{pooled_wake_f1:.6f}\n"
        )

        f.write(
            f"Pooled Macro-F1: "
            f"{pooled_macro_f1:.6f}\n"
        )

        f.write(
            f"Pooled ROC-AUC: "
            f"{pooled_roc_auc:.6f}\n"
        )

        f.write(
            f"Pooled PR-AUC: "
            f"{pooled_pr_auc:.6f}\n\n"
        )

        f.write(
            "Classification Report\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            final_report
        )

        f.write(
            "\n\nPooled Confusion Matrix\n"
        )

        f.write(
            str(pooled_cm)
        )

    # ==========================================================
    # 18. EXPERIMENT SUMMARY
    # ==========================================================

    summary = {

        "experiment_id": EXPERIMENT_ID,

        "experiment_name": EXPERIMENT_NAME,

        "timestamp": run_timestamp,

        "hypothesis": (
            "A six-feature single-channel frontal EEG "
            "representation can distinguish Wake from "
            "Non-Wake across previously unseen subjects "
            "under subject-independent LOSO validation."
        ),

        "primary_metric": (
            "Wake sensitivity / recall"
        ),

        "baseline_threshold": WAKE_THRESHOLD,

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

            "task": (
                "Binary Wake vs Non-Wake"
            ),

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

            "representation": (
                "Relative spectral bandpower"
            ),

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

            "n_estimators": 100,

            "random_state": 42,

            "n_jobs": -1
        },

        "validation": {

            "method": (
                "Leave-One-Subject-Out "
                "cross-validation"
            ),

            "n_folds": int(
                len(unique_subjects)
            )
        },

        "results": {

            "mean_fold_accuracy": (
                mean_fold_accuracy
            ),

            "std_fold_accuracy": (
                std_fold_accuracy
            ),

            "mean_fold_wake_sensitivity": float(
                np.mean(
                    fold_wake_sensitivities
                )
            ),

            "std_fold_wake_sensitivity": float(
                np.std(
                    fold_wake_sensitivities,
                    ddof=1
                )
            ),

            "mean_fold_specificity": float(
                np.mean(
                    fold_specificities
                )
            ),

            "mean_fold_balanced_accuracy": float(
                np.mean(
                    fold_balanced_accuracies
                )
            ),

            "mean_fold_wake_precision": float(
                np.mean(
                    fold_wake_precisions
                )
            ),

            "mean_fold_wake_f1": float(
                np.mean(
                    fold_wake_f1s
                )
            ),

            "mean_fold_roc_auc": float(
                np.mean(
                    fold_roc_aucs
                )
            ),

            "mean_fold_pr_auc": float(
                np.mean(
                    fold_pr_aucs
                )
            ),

            "pooled_accuracy": (
                float(pooled_accuracy)
            ),

            "pooled_balanced_accuracy": (
                float(pooled_balanced_accuracy)
            ),

            "pooled_wake_precision": (
                float(pooled_wake_precision)
            ),

            "pooled_wake_sensitivity": (
                float(pooled_wake_sensitivity)
            ),

            "pooled_wake_miss_rate": (
                float(pooled_wake_miss_rate)
            ),

            "pooled_wake_specificity": (
                float(pooled_specificity)
            ),

            "pooled_wake_f1": (
                float(pooled_wake_f1)
            ),

            "pooled_macro_f1": (
                float(pooled_macro_f1)
            ),

            "pooled_weighted_f1": (
                float(pooled_weighted_f1)
            ),

            "pooled_roc_auc": (
                float(pooled_roc_auc)
            ),

            "pooled_pr_auc": (
                float(pooled_pr_auc)
            ),

            "true_negative": int(
                pooled_tn
            ),

            "false_positive": int(
                pooled_fp
            ),

            "false_negative": int(
                pooled_fn
            ),

            "true_positive": int(
                pooled_tp
            )
        },

        "software_environment": {

            "python": (
                sys.version
            ),

            "platform": (
                platform.platform()
            ),

            "mne": get_package_version(
                "mne"
            ),

            "yasa": get_package_version(
                "yasa"
            ),

            "scipy": get_package_version(
                "scipy"
            ),

            "numpy": get_package_version(
                "numpy"
            ),

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
        "✅ EXP004 COMPLETE"
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
        "📊 Predictions, probabilities, confusion matrices, "
        "ROC and PR data saved."
    )


# ==============================================================
# ENTRY POINT
# ==============================================================

if __name__ == "__main__":
    main()
