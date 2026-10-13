
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

print("Loading dataset...")

data = np.load("Wafer_Map_Datasets.npz")
wafers = data["arr_0"]
y = data["arr_1"].astype(np.int32)

X = wafers.reshape(len(wafers), -1).astype(np.float32) / 2.0
del wafers, data

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

del X, X_temp, y_temp

print("Training alpha=0.001 model...")

model = OneVsRestClassifier(
    SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=0.001,
        max_iter=100,
        tol=1e-3,
        class_weight="balanced",
        random_state=42
    ),
    n_jobs=1
)

model.fit(X_train, y_train)

# Label 2 is column index 2
val_scores = model.decision_function(X_val)[:, 2]
test_scores = model.decision_function(X_test)[:, 2]

print("\nLABEL 2: VALIDATION THRESHOLD COMPARISON")
print("Threshold | Precision | Recall | F1")
print("-" * 43)

best_threshold = 0.0
best_f1 = -1.0

for threshold in np.arange(-2.0, 3.01, 0.2):
    val_pred = (val_scores >= threshold).astype(int)

    precision = precision_score(
        y_val[:, 2], val_pred, zero_division=0
    )
    recall = recall_score(
        y_val[:, 2], val_pred, zero_division=0
    )
    f1 = f1_score(
        y_val[:, 2], val_pred, zero_division=0
    )

    print(
        f"{threshold:8.1f} | {precision:9.3f} | "
        f"{recall:6.3f} | {f1:.3f}"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"\nBest validation threshold: {best_threshold:.1f}")
print(f"Best validation F1: {best_f1:.3f}")

# Evaluate this chosen threshold on the test set once
test_pred = (test_scores >= best_threshold).astype(int)

print("\nLABEL 2: TEST RESULTS AT SELECTED THRESHOLD")
print("Precision:", round(precision_score(
    y_test[:, 2], test_pred, zero_division=0
), 3))
print("Recall:   ", round(recall_score(
    y_test[:, 2], test_pred, zero_division=0
), 3))
print("F1:       ", round(f1_score(
    y_test[:, 2], test_pred, zero_division=0
), 3))
