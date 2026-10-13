from sklearn.metrics import classification_report
import time
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, hamming_loss
from sklearn.multioutput import MultiOutputClassifier

print("Loading dataset...")

data = np.load("Wafer_Map_Datasets.npz")
wafers = data["arr_0"]
y = data["arr_1"].astype(np.int32)

# Keep wafer-map values as numeric features
X = wafers.reshape(len(wafers), -1).astype(np.float32) / 2.0
del wafers, data

# Same split as previous experiments
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

del X, X_temp, y_temp

print("Training Random Forest...")

start = time.time()

base_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model = MultiOutputClassifier(base_model, n_jobs=1)
model.fit(X_train, y_train)

print(f"Training time: {(time.time() - start) / 60:.2f} minutes")

# Each output column is a separate binary classifier.
# Use validation probabilities to select thresholds.
print("Tuning thresholds on validation data...")

val_probabilities = np.column_stack([
    estimator.predict_proba(X_val)[:, 1]
    for estimator in model.estimators_
])

thresholds = []

for j in range(y.shape[1]):
    best_threshold = 0.5
    best_f1 = -1.0

    # Probability thresholds from 0.10 to 0.90
    for threshold in np.arange(0.10, 0.91, 0.05):
        val_pred = (
            val_probabilities[:, j] >= threshold
        ).astype(int)

        score = f1_score(
            y_val[:, j], val_pred, zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    thresholds.append(best_threshold)
    print(
        f"Label {j}: threshold={best_threshold:.2f}, "
        f"validation F1={best_f1:.3f}"
    )

# Evaluate test set only after threshold selection
print("\nEvaluating test set...")

test_probabilities = np.column_stack([
    estimator.predict_proba(X_test)[:, 1]
    for estimator in model.estimators_
])

y_pred = (
    test_probabilities >= np.array(thresholds)
).astype(int)

print("\n===== RANDOM FOREST RESULTS =====")
print(f"Macro F1: {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
print(f"Micro F1: {f1_score(y_test, y_pred, average='micro', zero_division=0):.4f}")
print(f"Hamming loss: {hamming_loss(y_test, y_pred):.4f}")

print("\nThresholds:", np.round(thresholds, 2))
print(f"Total elapsed time: {(time.time() - start) / 60:.2f} minutes")
print("\n===== PER-LABEL TEST REPORT =====")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0,
    target_names=[f"Label {i}" for i in range(y_test.shape[1])]
))
# Save the trained model and decision thresholds
joblib.dump(
    {
        "model": model,
        "thresholds": np.array(thresholds),
        "input_shape": (52, 52),
        "labels": [f"Label {i}" for i in range(8)]
    },
    "wafer_random_forest.joblib"
)

print("\nSaved model to wafer_random_forest.joblib")