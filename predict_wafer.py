
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model and its thresholds
saved = joblib.load("wafer_random_forest.joblib")
model = saved["model"]
thresholds = saved["thresholds"]
labels = saved["labels"]

# Load a wafer map from the dataset for a demonstration
data = np.load("Wafer_Map_Datasets.npz")
wafer = data["arr_0"][0]
actual_labels = data["arr_1"][0]

# Prepare the wafer map for prediction
X = wafer.reshape(1, -1).astype(np.float32) / 2.0

# Predict label probabilities
probabilities = np.column_stack([
    estimator.predict_proba(X)[:, 1]
    for estimator in model.estimators_
])[0]

predictions = (probabilities >= thresholds).astype(int)

print("\n===== WAFER DEFECT PREDICTION =====")
for label, probability, prediction, actual in zip(
    labels, probabilities, predictions, actual_labels
):
    status = "DETECTED" if prediction else "Not detected"
    print(
        f"{label}: {status} | "
        f"Probability: {probability:.3f} | "
        f"Actual label: {actual}"
    )

# Display the wafer map
plt.figure(figsize=(5, 5))
plt.imshow(wafer, cmap="viridis", interpolation="nearest")
plt.title("Wafer Map - Predicted Defect Patterns")
plt.axis("off")
plt.tight_layout()
plt.show()
