import os

import mne
import yasa
import numpy as np

from scipy.signal import welch


def extract_cohort_data(subject_ids):
    """
    Extract single-channel EEG features from consecutive 30-second epochs
    for multiple subjects.

    Ground-truth labels
    -------------------
    0 = Wake (W)
    1 = N1
    2 = N2
    3 = N3 (Stages 3 and 4 collapsed)
    4 = REM (R)

    Unsupported / unscored annotations such as:
    - Movement (M)
    - Unscored (?)
    - Any unmatched annotation

    are excluded from the dataset rather than being assigned to Wake.

    Feature representation
    ----------------------
    Six relative spectral bandpower features:
    - Delta: 0.5-4 Hz
    - Theta: 4-8 Hz
    - Alpha: 8-12 Hz
    - Sigma: 12-16 Hz
    - Beta: 16-30 Hz
    - Gamma: 30-40 Hz

    Spectral estimation uses Welch's method with:
    - 4-second internal segments
    - 50% overlap

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (n_epochs, 6).

    y : np.ndarray
        Integer sleep-stage labels.

    subjects : np.ndarray
        Subject identifier for every retained epoch.
    """

    # ============================================================
    # Dataset location
    # ============================================================
    data_dir = os.path.expanduser(
        os.path.join("~", "mne_data", "physionet-sleep-data")
    )

    # ============================================================
    # Output containers
    # ============================================================
    all_X = []
    all_y = []
    all_subjects = []

    # ============================================================
    # Sleep-stage mapping
    # ============================================================
    stage_map = {
        "W": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 3,   # R&K Stage 4 collapsed into N3
        "R": 4,
    }

    # ============================================================
    # Explicit frequency-band definitions
    # ============================================================
    bands = [
        (0.5, 4.0, "Delta"),
        (4.0, 8.0, "Theta"),
        (8.0, 12.0, "Alpha"),
        (12.0, 16.0, "Sigma"),
        (16.0, 30.0, "Beta"),
        (30.0, 40.0, "Gamma"),
    ]

    # ============================================================
    # Process each subject
    # ============================================================
    for sub in subject_ids:

        print(
            f"\n Processing Subject {sub}: "
            f"Extracting raw timelines..."
        )

        # --------------------------------------------------------
        # Resolve PSG and hypnogram filenames
        # --------------------------------------------------------
        if sub == 10:
            data_file = os.path.join(
                data_dir,
                "SC4101E0-PSG.edf"
            )

            label_file = os.path.join(
                data_dir,
                "SC4101EC-Hypnogram.edf"
            )

        else:
            data_file = os.path.join(
                data_dir,
                f"SC40{sub}1E0-PSG.edf"
            )

            label_file = os.path.join(
                data_dir,
                f"SC40{sub}1EC-Hypnogram.edf"
            )

        # Some recordings may use the "H" hypnogram filename.
        if not os.path.exists(label_file) and sub != 10:
            label_file = os.path.join(
                data_dir,
                f"SC40{sub}1EH-Hypnogram.edf"
            )

        # --------------------------------------------------------
        # Safety checks
        # --------------------------------------------------------
        if not os.path.exists(data_file):
            raise FileNotFoundError(
                f"PSG file not found for Subject {sub}: "
                f"{data_file}"
            )

        if not os.path.exists(label_file):
            raise FileNotFoundError(
                f"Hypnogram file not found for Subject {sub}: "
                f"{label_file}"
            )

        # ========================================================
        # 1. Load ONLY the Fpz-Cz channel
        # ========================================================
        #
        # Loading only the required channel avoids combining the
        # incompatible filter metadata from unrelated EDF channels.
        #
        raw = mne.io.read_raw_edf(
            data_file,
            include=["EEG Fpz-Cz"],
            preload=True,
            verbose=False
        )

        # Confirm the requested channel was actually loaded.
        if "EEG Fpz-Cz" not in raw.ch_names:
            raise RuntimeError(
                f"EEG Fpz-Cz was not found in Subject {sub}. "
                f"Available channels: {raw.ch_names}"
            )

        # --------------------------------------------------------
        # Sampling frequency
        # --------------------------------------------------------
        sfreq = raw.info["sfreq"]

        # --------------------------------------------------------
        # Extract the single EEG channel
        #
        # MNE stores EEG amplitudes in Volts.
        # Convert to microvolts for YASA compatibility.
        # --------------------------------------------------------
        data = raw.get_data()[0] * 1e6

        # ========================================================
        # 2. Divide recording into consecutive 30-second epochs
        # ========================================================
        window_size_samples = int(30 * sfreq)

        total_windows = (
            len(data) // window_size_samples
        )

        print(
            f"   Sampling frequency: {sfreq:.2f} Hz"
        )

        print(
            f"   Available 30-second epochs: "
            f"{total_windows}"
        )

        # ========================================================
        # 3. Read hypnogram annotations
        # ========================================================
        annotations = mne.read_annotations(
            label_file
        )

        onsets = annotations.onset
        durations = annotations.duration
        descriptions = annotations.description

        # ========================================================
        # 4. Process every 30-second epoch
        # ========================================================
        for i in range(total_windows):

            # ----------------------------------------------------
            # Epoch boundaries
            # ----------------------------------------------------
            start_sample = (
                i * window_size_samples
            )

            end_sample = (
                start_sample + window_size_samples
            )

            window_data = data[
                start_sample:end_sample
            ]

            # ----------------------------------------------------
            # Welch PSD
            #
            # Internal segment length = 4 seconds
            # Overlap = 50%
            # ----------------------------------------------------
            welch_window = int(
                4 * sfreq
            )

            freqs, psd = welch(
                window_data,
                fs=sfreq,
                nperseg=welch_window,
                noverlap=welch_window // 2
            )

            # ----------------------------------------------------
            # Relative spectral bandpower
            #
            # Six explicitly defined bands:
            # Delta, Theta, Alpha, Sigma, Beta, Gamma
            # ----------------------------------------------------
            bp = yasa.bandpower_from_psd_ndarray(
                psd,
                freqs,
                bands=bands,
                relative=True
            )

            bp = np.asarray(
                bp,
                dtype=float
            ).reshape(-1)

            # ----------------------------------------------------
            # Verify feature dimensionality
            # ----------------------------------------------------
            if bp.shape[0] != len(bands):
                raise ValueError(
                    f"Unexpected feature count for "
                    f"Subject {sub}, epoch {i}: "
                    f"got {bp.shape[0]}, "
                    f"expected {len(bands)}."
                )

            # ====================================================
            # Synchronize annotation to epoch center
            # ====================================================
            #
            # The midpoint of each 30-second epoch is used for
            # stage assignment.
            #
            current_time_sec = (
                i * 30.0
            ) + 15.0

            # ----------------------------------------------------
            # -1 = unsupported / unscored / unmatched
            #
            # IMPORTANT:
            # We do NOT default this to Wake.
            # ----------------------------------------------------
            assigned_stage = -1

            for start, dur, desc in zip(
                onsets,
                durations,
                descriptions
            ):

                if (
                    start
                    <= current_time_sec
                    < (start + dur)
                ):

                    clean_desc = str(desc).strip()

                    # ------------------------------------------------
                    # Match only valid scored sleep stages
                    # ------------------------------------------------
                    for code, stage_value in stage_map.items():

                        if (
                            clean_desc.endswith(code)
                            or
                            f"stage {code}"
                            in clean_desc.lower()
                        ):
                            assigned_stage = stage_value
                            break

                    # Stop searching once the annotation containing
                    # the epoch center has been identified.
                    break

            # ====================================================
            # Exclude unsupported / unscored epochs
            # ====================================================
            if assigned_stage == -1:
                continue

            # ====================================================
            # Exclude invalid numerical features
            # ====================================================
            if not np.all(
                np.isfinite(bp)
            ):
                continue

            # ====================================================
            # Store valid epoch
            # ====================================================
            all_X.append(bp)
            all_y.append(assigned_stage)
            all_subjects.append(sub)

    # ============================================================
    # Convert to NumPy arrays
    # ============================================================
    X = np.asarray(
        all_X,
        dtype=float
    )

    y = np.asarray(
        all_y,
        dtype=int
    )

    subjects = np.asarray(
        all_subjects,
        dtype=int
    )

    # ============================================================
    # Final integrity checks
    # ============================================================
    if X.ndim != 2:
        raise RuntimeError(
            f"Unexpected feature matrix shape: {X.shape}"
        )

    if X.shape[1] != 6:
        raise RuntimeError(
            f"Expected 6 spectral features, "
            f"got {X.shape[1]}"
        )

    if not (
        len(X)
        == len(y)
        == len(subjects)
    ):
        raise RuntimeError(
            "Feature, label, and subject arrays "
            "have inconsistent lengths."
        )

    print(
        "\n Feature extraction complete."
    )

    print(
        f"   Total retained epochs: {len(X)}"
    )

    print(
        f"   Feature dimensions: {X.shape[1]}"
    )

    print(
        f"   Subjects represented: "
        f"{np.unique(subjects)}"
    )

    return X, y, subjects
