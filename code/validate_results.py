
"""Read-only integrity and registry consistency checks for saved experiments.

This script does not retrain models or modify experiment artifacts.
It checks saved-file integrity and selected registry metrics only.
"""

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

EXPERIMENTS = {
    "EXP003": "EXP003_corrected_baseline",
    "EXP004": "EXP004_wake_binary_baseline",
    "EXP005": "EXP005_nested_wake_threshold",
    "EXP006": "EXP006_operating_point_and_cost_analysis",
}

REGISTRY_METRICS = {
    "mean_fold_accuracy": ("mean_fold_accuracy",),
    "accuracy_sd": ("std_fold_accuracy", "accuracy_sd"),
    "pooled_accuracy": ("pooled_accuracy",),
    "macro_f1": ("pooled_macro_f1", "macro_f1"),
    "weighted_f1": ("pooled_weighted_f1", "weighted_f1"),
}


passed = 0
failed = 0
warnings = 0


def report(status, message):
    global passed, failed, warnings
    print(f"{status} {message}")
    if status == "PASS":
        passed += 1
    elif status == "FAIL":
        failed += 1
    elif status == "WARN":
        warnings += 1


def read_json(path, description):
    if not path.is_file():
        report("FAIL", f"Missing {description}: {path.relative_to(ROOT)}")
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        report("PASS", f"Valid JSON: {path.relative_to(ROOT)}")
        return data
    except (OSError, json.JSONDecodeError) as exc:
        report("FAIL", f"Cannot read {description}: {exc}")
        return None


def get_registry_metrics(experiment_id, summary):
    """Return the intended metrics for registry comparison."""
    if experiment_id in ("EXP003", "EXP004"):
        return summary.get("results", {})

    if experiment_id == "EXP005":
        return summary.get("results", {}).get("selected", {})

    if experiment_id == "EXP006":
        scenario = summary.get("primary_registry_scenario")
        scenarios = summary.get("results", {}).get("scenarios", {})
        if not scenario:
            report("FAIL", "EXP006 has no primary_registry_scenario")
            return {}
        if scenario not in scenarios:
            report("FAIL", f"EXP006 primary scenario missing: {scenario}")
            return {}
        report("PASS", f"EXP006 registry scenario identified: {scenario}")
        return scenarios[scenario]

    return {}


def finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def main():
    print("READ-ONLY RESULTS VALIDATION")
    print(f"Repository: {ROOT}")
    print("No models will be trained and no results will be modified.\n")

    report("PASS" if RESULTS.is_dir() else "FAIL",
           "Results directory exists" if RESULTS.is_dir()
           else "Results directory is missing")

    summaries = {}

    for experiment_id, folder in EXPERIMENTS.items():
        directory = RESULTS / folder
        report("PASS" if directory.is_dir() else "FAIL",
               f"{experiment_id} directory exists" if directory.is_dir()
               else f"Missing experiment directory: {folder}")

        if not directory.is_dir():
            continue

        summary_path = directory / "experiment_summary.json"
        summary = read_json(summary_path, f"{experiment_id} summary")
        if summary is not None:
            summaries[experiment_id] = summary

            saved_id = summary.get("experiment_id")
            if saved_id == experiment_id:
                report("PASS", f"{experiment_id} summary ID matches folder")
            elif saved_id is None:
                report("WARN", f"{experiment_id} summary has no experiment_id")
            else:
                report("FAIL", f"{experiment_id} summary ID is {saved_id!r}")

            experiment = summary.get("experiment", {})
            n_subjects = experiment.get("n_subjects")
            if isinstance(n_subjects, int) and n_subjects > 0:
                report("PASS", f"{experiment_id} positive subject count recorded")
            else:
                report("FAIL", f"{experiment_id} missing/invalid subject count")

        predictions = directory / "loso_predictions.csv"
        if not predictions.is_file():
            report("FAIL", f"Missing predictions CSV: {predictions.relative_to(ROOT)}")
        else:
            try:
                with predictions.open("r", encoding="utf-8-sig", newline="") as handle:
                    reader = csv.DictReader(handle)
                    rows = sum(1 for _ in reader)
                if rows > 0:
                    report("PASS", f"{experiment_id} predictions CSV readable ({rows} rows)")
                else:
                    report("FAIL", f"{experiment_id} predictions CSV is empty")
            except (OSError, csv.Error) as exc:
                report("FAIL", f"{experiment_id} predictions CSV unreadable: {exc}")

    registry_path = RESULTS / "experiment_registry.csv"
    registry_rows = None
    if registry_path.is_file():
        try:
            with registry_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                registry_fields = reader.fieldnames or []
                registry_rows = list(reader)
            report("PASS", "Registry CSV readable")
        except (OSError, csv.Error) as exc:
            report("FAIL", f"Registry CSV unreadable: {exc}")

    if registry_rows is not None:
        required_columns = {
            "experiment_id",
            *REGISTRY_METRICS.keys(),
        }
        missing_columns = required_columns - set(registry_fields)
        if not missing_columns:
            report("PASS", "Registry contains required columns")
        else:
            report("FAIL", f"Registry missing columns: {sorted(missing_columns)}")

        ids = [row.get("experiment_id", "").strip() for row in registry_rows]
        if len(ids) == len(set(ids)):
            report("PASS", "Registry experiment IDs are unique")
        else:
            report("FAIL", "Registry contains duplicate experiment IDs")

        for experiment_id in EXPERIMENTS:
            matches = [row for row in registry_rows
                       if row.get("experiment_id", "").strip() == experiment_id]
            if len(matches) != 1:
                report("FAIL", f"Expected exactly one registry row for {experiment_id}; found {len(matches)}")
                continue

            summary = summaries.get(experiment_id)
            if summary is None:
                report("WARN", f"Cannot compare {experiment_id}: summary unavailable")
                continue

            metrics = get_registry_metrics(experiment_id, summary)
            if not metrics:
                report("FAIL", f"No metric mapping available for {experiment_id}")
                continue

            row = matches[0]
            for registry_field, aliases in REGISTRY_METRICS.items():
                source_field = next((field for field in aliases
                                     if field in metrics), None)
                if source_field is None:
                    report("FAIL", f"{experiment_id}: no source metric for {registry_field}")
                    continue

                try:
                    registry_value = float(row[registry_field])
                    source_value = metrics[source_field]
                except (ValueError, TypeError, KeyError):
                    report("FAIL", f"{experiment_id}: invalid {registry_field} value")
                    continue

                if not finite_number(source_value) or not math.isfinite(registry_value):
                    report("FAIL", f"{experiment_id}: non-finite {registry_field}")
                elif math.isclose(registry_value, source_value, rel_tol=1e-9, abs_tol=1e-10):
                    report("PASS", f"{experiment_id} registry {registry_field} matches {source_field}")
                else:
                    report("FAIL", f"{experiment_id} registry {registry_field} mismatch: "
                                   f"{registry_value} vs {source_value}")

    ignored_dirs = {".venv", "venv", "__pycache__", ".git"}
    hygiene_items = []
    for path in ROOT.rglob("*"):
        if any(part in ignored_dirs for part in path.parts):
            continue
        if path.name == ".DS_Store" or path.name.endswith(".bak"):
            hygiene_items.append(path.relative_to(ROOT))

    if hygiene_items:
        for item in hygiene_items:
            report("WARN", f"Review repository hygiene item: {item}")
    else:
        report("PASS", "No .DS_Store or .bak files found outside excluded directories")

    print("\n" + "=" * 60)
    print(f"Checks passed: {passed}")
    print(f"Checks failed: {failed}")
    print(f"Warnings: {warnings}")
    print("Scope: saved-artifact integrity and registry consistency only.")
    print("This does not rerun or independently reproduce the experiments.")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
