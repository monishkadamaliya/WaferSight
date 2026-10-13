
import time
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    classification_report,
)

RANDOM_STATE = 42
start = time.time()

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------
print("Loading dataset...", flush=True)

with np.load("Wafer_Map_Datasets.npz") as data:
    wafers = data["arr_0"]
    labels = data["arr_1"]

X = wafers.reshape(len(wafers), -1).astype(np.float32) / 2.0
y = labels.astype(np.uint8)

del wafers, labels

# Reproduce the original split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE
)
del X, y, X_temp, y_temp

# --------------------------------------------------
# 2. Train the baseline model
# --------------------------------------------------
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

print("Training model...", flush=True)
model.fit(X_train, y_train)

# --------------------------------------------------
# 3. Tune thresholds on validation data only
# --------------------------------------------------
print("Tuning thresholds on validation set...", flush=True)

# Each binary SGD classifier supports decision_function.
val_scores = model.decision_function(X_val)
test_scores = model.decision_function(X_test)

thresholds = np.zeros(y_val.shape[1], dtype=float)

# Search thresholds for macro-F1 of each individual label.
# This gives rare labels equal weight during threshold selection.
candidates = np.arange(-2.0, 2.01, 0.1)

for j in range(y_val.shape[1]):
    best_f1 = -1.0
    best_threshold = 0.0

    for threshold in candidates:
        pred_j = (val_scores[:, j] >= threshold).astype(np.uint8)
        score = f1_score(
            y_val[:, j], pred_j, zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    thresholds[j] = best_threshold

    print(
        f"Label {j}: threshold={best_threshold:.2f}, "
        f"validation F1={best_f1:.3f}",
        flush=True,
    )

# --------------------------------------------------
# 4. Evaluate default vs tuned thresholds on test set
# --------------------------------------------------
default_pred = (test_scores >= 0.0).astype(np.uint8)
tuned_pred = (test_scores >= thresholds).astype(np.uint8)

def evaluate(name, truth, pred):
    print(f"\n{name}")
    print("Macro F1:", round(
        f1_score(truth, pred, average="macro", zero_division=0), 4
    ))
    print("Micro F1:", round(
        f1_score(truth, pred, average="micro", zero_division=0), 4
    ))
    print("Hamming loss:", round(hamming_loss(truth, pred), 4))

evaluate("TEST RESULTS — DEFAULT THRESHOLD", y_test, default_pred)
evaluate("TEST RESULTS — TUNED THRESHOLDS", y_test, tuned_pred)

print("\nTuned classification report:")
print(classification_report(
    y_test,
    tuned_pred,
    labels=list(range(8)),
    target_names=[f"Label {i}" for i in range(8)],
    zero_division=0,
))

# --------------------------------------------------
# 5. Save model, thresholds, and test predictions
# --------------------------------------------------
joblib.dump(
    {"model": model, "thresholds": thresholds},
    "wafer_fault_tuned.joblib",
)

np.savez(
    "tuned_test_results.npz",
    y_true=y_test,
    y_pred_default=default_pred,
    y_pred_tuned=tuned_pred,
    thresholds=thresholds,
)

print("\nThresholds:", thresholds)
print("Saved wafer_fault_tuned.joblib")
print("Saved tuned_test_results.npz")
print(f"Elapsed time: {(time.time() - start) / 60:.2f} minutes")

# Plot label-wise F1 before and after tuning
default_f1 = [
    f1_score(y_test[:, j], default_pred[:, j], zero_division=0)
    for j in range(y_test.shape[1])
]
tuned_f1 = [
    f1_score(y_test[:, j], tuned_pred[:, j], zero_division=0)
    for j in range(y_test.shape[1])
]

ids = np.arange(y_test.shape[1])
width = 0.38

plt.figure(figsize=(10, 5))
plt.bar(ids - width / 2, default_f1, width, label="Default")
plt.bar(ids + width / 2, tuned_f1, width, label="Tuned")
plt.xticks(ids)
plt.xlabel("Label index")
plt.ylabel("Test F1 score")
plt.ylim(0, 1)
plt.title("Per-label F1: Default vs Tuned Thresholds")
plt.legend()
plt.tight_layout()
plt.savefig("threshold_comparison.png", dpi=150)
plt.close()

print("Saved threshold_comparison.png")
