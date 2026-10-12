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
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix
)


# ==============================================================
# EXPERIMENT CONFIGURATION
# ==============================================================

# This is officially Experiment 3.
EXPERIMENT_ID = "EXP003"

# Short descriptive name.
EXPERIMENT_NAME = "corrected_baseline"

# IMPORTANT:
# The script will NEVER overwrite an existing experiment folder.
# If the folder already exists, the program stops.
OVERWRITE_EXISTING = False


# ==============================================================
# PROJECT PATHS
# ==============================================================

# run_pipeline.py is inside:
# Sleep_Research_Project/code/
#
# Therefore the project root is one level above this file's folder.

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
# CLASS DEFINITIONS
# ==============================================================

CLASS_IDS = [0, 1, 2, 3, 4]

CLASS_NAMES = {
    0: "Awake(W)",
    1: "Light(N1)",
    2: "Moderate(N2)",
    3: "Deep(N3)",
    4: "REM(R)"
}


# ==============================================================
# HELPER FUNCTIONS
# ==============================================================

def save_csv(filename, fieldnames, rows):
    """
    Save dictionaries as a CSV inside the current experiment folder.
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
    Safely obtain an installed package version.
    """

    try:
        return importlib.metadata.version(
            package_name
        )
    except Exception:
        return "unknown"


def save_code_snapshot():
    """
    Preserve the exact analysis code used for this experiment.
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
    Append a one-line summary of the experiment to the master registry.
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


# ==============================================================
# MAIN EXPERIMENT
# ==============================================================

def main():

    # ----------------------------------------------------------
    # Prepare directories
    # ----------------------------------------------------------

    os.makedirs(
        RESULTS_ROOT,
        exist_ok=True
    )

    # ----------------------------------------------------------
    # Protect existing experiments
    # ----------------------------------------------------------

    if os.path.exists(EXPERIMENT_DIR):

        if not OVERWRITE_EXISTING:

            raise FileExistsError(
                "\n❌ EXPERIMENT ALREADY EXISTS\n\n"
                f"Folder:\n{EXPERIMENT_DIR}\n\n"
                "This script is intentionally refusing to "
                "overwrite an existing experiment.\n\n"
                "To run a NEW experiment, change:\n"
                "EXPERIMENT_ID\n"
                "EXPERIMENT_NAME\n\n"
                "Do NOT delete the existing experiment folder."
            )

    else:

        os.makedirs(
            EXPERIMENT_DIR
        )

    # ----------------------------------------------------------
    # Snapshot exact code
    # ----------------------------------------------------------

    save_code_snapshot()

    # ----------------------------------------------------------
    # Experiment timestamp
    # ----------------------------------------------------------

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
        f"📁 Experiment directory:\n"
        f"{EXPERIMENT_DIR}\n"
    )

    print(
        "🔄 Initializing 10-subject LOSO validation..."
    )

    # ==========================================================
    # 1. LOAD COHORT
    # ==========================================================

    cohort_subjects = [
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10
    ]

    X_matrix, y_vector, subject_tracking = (
        extract_cohort_data(
            cohort_subjects
        )
    )

    print(
        f"\n📊 Master dataset compiled: "
        f"{len(X_matrix)} epochs."
    )

    unique_subs = np.unique(
        subject_tracking
    )

    print(
        f"🧬 Subjects detected: "
        f"{unique_subs}"
    )

    # ==========================================================
    # 2. INTEGRITY CHECKS
    # ==========================================================

    if len(unique_subs) == 0:

        raise RuntimeError(
            "No subjects were extracted."
        )

    if X_matrix.ndim != 2:

        raise RuntimeError(
            f"Unexpected X matrix shape: "
            f"{X_matrix.shape}"
        )

    if X_matrix.shape[1] != 6:

        raise RuntimeError(
            f"Expected 6 features, "
            f"got {X_matrix.shape[1]}."
        )

    if not (
        len(X_matrix)
        == len(y_vector)
        == len(subject_tracking)
    ):

        raise RuntimeError(
            "X, y, and subject arrays have "
            "different lengths."
        )

    # ==========================================================
    # 3. STORAGE
    # ==========================================================

    all_accuracies = []

    all_y_true = []
    all_y_pred = []

    fold_results = []
    fold_class_metrics = []
    fold_class_distributions = []
    prediction_rows = []

    overall_confusion = np.zeros(
        (5, 5),
        dtype=int
    )

    # ==========================================================
    # 4. LOSO
    # ==========================================================

    print(
        "\n🤖 Launching Leave-One-Subject-Out "
        "(LOSO) validation..."
    )

    for test_sub in unique_subs:

        print(
            f"\n⚡ Held-out Subject "
            f"{test_sub:02d}"
        )

        # ------------------------------------------------------
        # Split by subject
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

        y_train = y_vector[
            train_mask
        ]

        X_test = X_matrix[
            test_mask
        ]

        y_test = y_vector[
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
        # Predictions
        # ======================================================

        y_pred = model.predict(
            X_test
        )

        y_prob = model.predict_proba(
            X_test
        )

        # ======================================================
        # Fold metrics
        # ======================================================

        fold_accuracy = accuracy_score(
            y_test,
            y_pred
        )

        fold_macro_f1 = f1_score(
            y_test,
            y_pred,
            labels=CLASS_IDS,
            average="macro",
            zero_division=0
        )

        fold_weighted_f1 = f1_score(
            y_test,
            y_pred,
            labels=CLASS_IDS,
            average="weighted",
            zero_division=0
        )

        all_accuracies.append(
            fold_accuracy
        )

        all_y_true.extend(
            y_test
        )

        all_y_pred.extend(
            y_pred
        )

        print(
            f"   Accuracy: {fold_accuracy:.2%}"
        )

        print(
            f"   Macro-F1: {fold_macro_f1:.2%}"
        )

        # ======================================================
        # Fold summary
        # ======================================================

        fold_results.append({
            "held_out_subject": int(test_sub),
            "training_epochs": int(len(X_train)),
            "test_epochs": int(len(X_test)),
            "accuracy": float(fold_accuracy),
            "macro_f1": float(fold_macro_f1),
            "weighted_f1": float(fold_weighted_f1)
        })

        # ======================================================
        # Per-class metrics
        # ======================================================

        report = classification_report(
            y_test,
            y_pred,
            labels=CLASS_IDS,
            target_names=[
                CLASS_NAMES[i]
                for i in CLASS_IDS
            ],
            output_dict=True,
            zero_division=0
        )

        for class_id in CLASS_IDS:

            class_name = CLASS_NAMES[
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
        # Class distributions
        # ======================================================

        for class_id in CLASS_IDS:

            fold_class_distributions.append({
                "held_out_subject": int(test_sub),
                "class_id": int(class_id),
                "class_name": CLASS_NAMES[class_id],
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
        # Individual predictions
        # ======================================================

        for row_index in range(
            len(y_test)
        ):

            probability_map = {
                class_id: 0.0
                for class_id in CLASS_IDS
            }

            for column_index, class_id in enumerate(
                model.classes_
            ):

                probability_map[
                    int(class_id)
                ] = float(
                    y_prob[
                        row_index,
                        column_index
                    ]
                )

            prediction_rows.append({
                "held_out_subject": int(test_sub),
                "epoch_index_within_fold": int(
                    row_index
                ),
                "true_label": int(
                    y_test[row_index]
                ),
                "true_class": CLASS_NAMES[
                    int(y_test[row_index])
                ],
                "predicted_label": int(
                    y_pred[row_index]
                ),
                "predicted_class": CLASS_NAMES[
                    int(y_pred[row_index])
                ],
                "prob_awake": probability_map[0],
                "prob_n1": probability_map[1],
                "prob_n2": probability_map[2],
                "prob_n3": probability_map[3],
                "prob_rem": probability_map[4]
            })

        # ======================================================
        # Subject confusion matrix
        # ======================================================

        subject_confusion = confusion_matrix(
            y_test,
            y_pred,
            labels=CLASS_IDS
        )

        overall_confusion += (
            subject_confusion
        )

        confusion_rows = []

        for actual_index, actual_class in enumerate(
            CLASS_IDS
        ):

            for predicted_index, predicted_class in enumerate(
                CLASS_IDS
            ):

                confusion_rows.append({
                    "actual_class": CLASS_NAMES[
                        actual_class
                    ],
                    "predicted_class": CLASS_NAMES[
                        predicted_class
                    ],
                    "count": int(
                        subject_confusion[
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
            confusion_rows
        )

    # ==========================================================
    # 5. FINAL AGGREGATION
    # ==========================================================

    all_y_true = np.asarray(
        all_y_true
    )

    all_y_pred = np.asarray(
        all_y_pred
    )

    mean_fold_accuracy = float(
        np.mean(
            all_accuracies
        )
    )

    std_fold_accuracy = float(
        np.std(
            all_accuracies,
            ddof=1
        )
    )

    pooled_accuracy = float(
        accuracy_score(
            all_y_true,
            all_y_pred
        )
    )

    pooled_macro_f1 = float(
        f1_score(
            all_y_true,
            all_y_pred,
            labels=CLASS_IDS,
            average="macro",
            zero_division=0
        )
    )

    pooled_weighted_f1 = float(
        f1_score(
            all_y_true,
            all_y_pred,
            labels=CLASS_IDS,
            average="weighted",
            zero_division=0
        )
    )

    # ==========================================================
    # 6. FINAL CLASSIFICATION REPORT
    # ==========================================================

    final_report = classification_report(
        all_y_true,
        all_y_pred,
        labels=CLASS_IDS,
        target_names=[
            CLASS_NAMES[i]
            for i in CLASS_IDS
        ],
        zero_division=0
    )

    print(
        "\n=========================================================="
    )

    print(
        "📊 FINAL EXPERIMENT RESULTS"
    )

    print(
        "=========================================================="
    )

    print(
        f"Mean fold accuracy: "
        f"{mean_fold_accuracy:.4%}"
    )

    print(
        f"Fold accuracy SD:   "
        f"{std_fold_accuracy:.4%}"
    )

    print(
        f"Pooled accuracy:    "
        f"{pooled_accuracy:.4%}"
    )

    print(
        f"Macro-F1:            "
        f"{pooled_macro_f1:.4%}"
    )

    print(
        f"Weighted-F1:         "
        f"{pooled_weighted_f1:.4%}"
    )

    print(
        "\nClassification report:\n"
    )

    print(
        final_report
    )

    # ==========================================================
    # 7. SAVE CSV FILES
    # ==========================================================

    save_csv(
        "fold_results.csv",
        [
            "held_out_subject",
            "training_epochs",
            "test_epochs",
            "accuracy",
            "macro_f1",
            "weighted_f1"
        ],
        fold_results
    )

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

    save_csv(
        "loso_predictions.csv",
        [
            "held_out_subject",
            "epoch_index_within_fold",
            "true_label",
            "true_class",
            "predicted_label",
            "predicted_class",
            "prob_awake",
            "prob_n1",
            "prob_n2",
            "prob_n3",
            "prob_rem"
        ],
        prediction_rows
    )

    # ==========================================================
    # 8. OVERALL CONFUSION MATRIX
    # ==========================================================

    overall_confusion_rows = []

    for actual_index, actual_class in enumerate(
        CLASS_IDS
    ):

        for predicted_index, predicted_class in enumerate(
            CLASS_IDS
        ):

            overall_confusion_rows.append({
                "actual_class": CLASS_NAMES[
                    actual_class
                ],
                "predicted_class": CLASS_NAMES[
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
    # 9. SAVE TEXT CLASSIFICATION REPORT
    # ==========================================================

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
            f"Pooled Macro-F1: "
            f"{pooled_macro_f1:.6f}\n"
        )

        f.write(
            f"Pooled Weighted-F1: "
            f"{pooled_weighted_f1:.6f}\n\n"
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

    # ==========================================================
    # 10. EXPERIMENT SUMMARY
    # ==========================================================

    summary = {

        "experiment_id": EXPERIMENT_ID,

        "experiment_name": EXPERIMENT_NAME,

        "timestamp": run_timestamp,

        "experiment": {

            "subjects": [
                int(x)
                for x in unique_subs
            ],

            "n_subjects": int(
                len(unique_subs)
            ),

            "n_epochs": int(
                len(X_matrix)
            ),

            "n_features": int(
                X_matrix.shape[1]
            )
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

            "welch_segment": (
                "4 seconds"
            ),

            "welch_overlap": (
                "50%"
            )
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
                len(unique_subs)
            )
        },

        "results": {

            "mean_fold_accuracy": (
                mean_fold_accuracy
            ),

            "std_fold_accuracy": (
                std_fold_accuracy
            ),

            "pooled_accuracy": (
                pooled_accuracy
            ),

            "pooled_macro_f1": (
                pooled_macro_f1
            ),

            "pooled_weighted_f1": (
                pooled_weighted_f1
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
    # 11. UPDATE MASTER EXPERIMENT REGISTRY
    # ==========================================================

    append_registry_entry(
        summary
    )

    # ==========================================================
    # 12. FINISHED
    # ==========================================================

    print(
        "\n=========================================================="
    )

    print(
        "✅ EXPERIMENT COMPLETE"
    )

    print(
        "=========================================================="
    )

    print(
        f"📁 Saved to:\n"
        f"{EXPERIMENT_DIR}"
    )

    print(
        "\n🧾 Code snapshot saved."
    )

    print(
        "📊 Results archived."
    )

    print(
        "📚 Experiment registry updated."
    )

    print(
        "\n🔒 This experiment is now preserved."
    )


# ==============================================================
# ENTRY POINT
# ==============================================================

if __name__ == "__main__":
    main()
