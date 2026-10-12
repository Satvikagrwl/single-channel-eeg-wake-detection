import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_and_evaluate_model(X_data, y_true):
    """
    Partitions data arrays into a 70/30 split, trains a Random Forest Classifier,
    and prints out an empirical metrics table.
    """
    # 1. Partition the dataset arrays into isolated Train (70%) and Test (30%) fields
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_true, test_size=0.3, random_state=42
    )

    print(f" Training Random Forest on {len(X_train)} windows and testing on {len(X_test)} windows...")

    # 2. Initialize and fit the ensemble model (100 independent decision trees)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 3. Generate predictions on hidden test data
    y_pred = model.predict(X_test)

    # 4. Generate structured academic metric strings
    print("\n --- SLEEP-STAGE CLASSIFICATION RESULTS ---")
    unique_labels = np.unique(np.concatenate((y_test, y_pred)))
    all_names = {0: 'Awake(W)', 1: 'Light(N1)', 2: 'Moderate(N2)', 3: 'Deep(N3)', 4: 'REM(R)'}
    target_names = [all_names[lbl] for lbl in unique_labels]

    print(classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names, zero_division=0))
    print(" Pipeline completed successfully.")

    return model
