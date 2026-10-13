
import time
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    hamming_loss,
)

start = time.time()
RANDOM_STATE = 42

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------
print("Loading dataset...", flush=True)

with np.load("Wafer_Map_Datasets.npz") as data:
    wafers = data["arr_0"]
    labels = data["arr_1"]

print("Wafer maps:", wafers.shape)
print("Labels:", labels.shape)

# Validate the expected dataset structure
if wafers.ndim != 3 or wafers.shape[1:] != (52, 52):
    raise ValueError("Expected wafer maps with shape (N, 52, 52).")

if labels.shape != (len(wafers), 8):
    raise ValueError("Expected eight binary labels per wafer.")

if not np.isin(labels, [0, 1]).all():
    raise ValueError("Labels must contain only 0 and 1.")

print("\nPositive examples per label:", labels.sum(axis=0))

# --------------------------------------------------
# 2. Prepare features
# --------------------------------------------------
# Each 52x52 wafer becomes 2704 numeric features.
# Scale values 0, 1, 2 to 0, 0.5, 1.
X = wafers.reshape(len(wafers), -1).astype(np.float32)
X /= 2.0
y = labels.astype(np.uint8)

del wafers, labels

# --------------------------------------------------
# 3. Split into train, validation and test sets
# --------------------------------------------------
# Two-stage split: 70% train, 15% validation, 15% test.
# This is an ordinary split, not multilabel-stratified.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE
)

del X, y, X_temp, y_temp

print("\nTrain:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

print("Training label counts:", y_train.sum(axis=0))
print("Test label counts:", y_test.sum(axis=0))

if np.any(y_train.sum(axis=0) == 0):
    raise ValueError(
        "At least one label has no positive training examples. "
        "Use a multilabel-stratified split."
    )

if np.any(y_test.sum(axis=0) == 0):
    print(
        "WARNING: At least one label has no positive test examples. "
        "Its evaluation will not be reliable."
    )

# --------------------------------------------------
# 4. Build and train the multi-label classifier
# --------------------------------------------------
print("\nTraining model...", flush=True)

model = OneVsRestClassifier(
    SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=0.0001,
        max_iter=30,
        tol=1e-3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    n_jobs=1,
)

model.fit(X_train, y_train)

print(
    f"Training finished in {(time.time() - start) / 60:.2f} minutes.",
    flush=True,
)

# --------------------------------------------------
# 5. Validation evaluation
# --------------------------------------------------
val_pred = model.predict(X_val)

print("\nVALIDATION RESULTS")
print("Macro F1:", round(
    f1_score(y_val, val_pred, average="macro", zero_division=0), 4
))
print("Micro F1:", round(
    f1_score(y_val, val_pred, average="micro", zero_division=0), 4
))
print("Hamming loss:", round(hamming_loss(y_val, val_pred), 4))

# --------------------------------------------------
# 6. Final test evaluation
# --------------------------------------------------
test_pred = model.predict(X_test)

macro_f1 = f1_score(
    y_test, test_pred, average="macro", zero_division=0
)
micro_f1 = f1_score(
    y_test, test_pred, average="micro", zero_division=0
)
hloss = hamming_loss(y_test, test_pred)

print("\nFINAL TEST RESULTS")
print("Macro F1:", round(macro_f1, 4))
print("Micro F1:", round(micro_f1, 4))
print("Hamming loss:", round(hloss, 4))

print("\nPer-label classification report:")
print(classification_report(
    y_test,
    test_pred,
    labels=list(range(8)),
    target_names=[f"Label {i}" for i in range(8)],
    zero_division=0,
))

# --------------------------------------------------
# 7. Save the evaluation chart
# --------------------------------------------------
label_ids = np.arange(8)
actual_counts = y_test.sum(axis=0)
predicted_counts = test_pred.sum(axis=0)

width = 0.38
plt.figure(figsize=(10, 5))
plt.bar(label_ids - width / 2, actual_counts, width, label="Actual")
plt.bar(label_ids + width / 2, predicted_counts, width, label="Predicted")
plt.xlabel("Label index")
plt.ylabel("Positive labels in test set")
plt.title("Actual vs Predicted Wafer Defect Labels")
plt.xticks(label_ids)
plt.legend()
plt.tight_layout()
plt.savefig("model_results.png", dpi=150)
plt.close()

# --------------------------------------------------
# 8. Save model and metrics
# --------------------------------------------------
joblib.dump(model, "wafer_fault_model.joblib")

np.savez(
    "test_results.npz",
    y_true=y_test,
    y_pred=test_pred,
)

print("\nSaved files:")
print("- wafer_fault_model.joblib")
print("- model_results.png")
print("- test_results.npz")
print(f"\nTotal elapsed time: {(time.time() - start) / 60:.2f} minutes.")
