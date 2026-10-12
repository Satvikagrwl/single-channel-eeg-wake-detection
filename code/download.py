
"""Download the ten-subject Sleep-EDF cohort used in this project."""

import sys

import mne


SUBJECTS = list(range(10))
RECORDINGS = [1]


def main():
    """Download recording 1 for all ten subjects."""
    print("Downloading Sleep-EDF data from PhysioNet...")
    print(f"Subjects: {[subject + 1 for subject in SUBJECTS]}")
    print(f"Recording(s): {RECORDINGS}")
    print("MNE will use its configured dataset cache.\n")

    try:
        files = mne.datasets.sleep_physionet.age.fetch_data(
            subjects=SUBJECTS,
            recording=RECORDINGS,
            on_missing="raise",
            verbose=True,
        )
    except Exception as exc:
        print(f"\nERROR: Dataset download failed: {exc}", file=sys.stderr)
        print(
            "Check your internet connection and rerun the script.",
            file=sys.stderr,
        )
        return 1

    expected_subjects = len(SUBJECTS)
    returned_subjects = len(files)

    if returned_subjects != expected_subjects:
        print(
            f"\nERROR: Expected {expected_subjects} subject results, "
            f"but received {returned_subjects}.",
            file=sys.stderr,
        )
        return 1

    print(
        f"\nDownload request completed successfully for "
        f"{expected_subjects} subjects."
    )
    print("The returned file paths are available in MNE's dataset cache.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
