# Import tools for evaluation, timing, data processing, and saving models
from sklearn.metrics import classification_report
import time
import numpy as np
import joblib

# Split the dataset into training, validation, and test sets
from sklearn.model_selection import train_test_split

# Random Forest classifier
from sklearn.ensemble import RandomForestClassifier

# Metrics used to measure model performance
from sklearn.metrics import f1_score, hamming_loss

# Allows one classifier to be trained for each output label
from sklearn.multioutput import MultiOutputClassifier


# --------------------------------------------------
# 1. LOAD THE DATASET
# --------------------------------------------------

print("Loading dataset...")

# Load the compressed NumPy dataset
data = np.load("Wafer_Map_Datasets.npz")

# arr_0 contains wafer maps (input data)
wafers = data["arr_0"]

# arr_1 contains the labels for each wafer
# Convert labels to integers, typically 0 or 1
y = data["arr_1"].astype(np.int32)


# --------------------------------------------------
# 2. PREPARE THE INPUT FEATURES
# --------------------------------------------------

# Each wafer map is a 52 x 52 grid.
# Flatten it into a one-dimensional vector of 2704 values.
#
# Example:
# A wafer map with shape (52, 52)
# becomes a feature vector with shape (2704,).
#
# Dividing by 2.0 scales the original numeric values.
# This assumes the wafer-map values are encoded appropriately.
X = wafers.reshape(len(wafers), -1).astype(np.float32) / 2.0

# Release references to the original dataset arrays
# to reduce memory usage.
del wafers, data


# --------------------------------------------------
# 3. SPLIT DATA INTO TRAINING, VALIDATION, AND TEST SETS
# --------------------------------------------------

# Reserve 70% of the data for training.
# Keep 30% temporarily for validation and testing.
#
# random_state=42 makes the split reproducible.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Divide the remaining 30% equally:
# 15% for validation and 15% for testing.
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

# Release arrays that are no longer needed.
del X, X_temp, y_temp


# --------------------------------------------------
# 4. CREATE AND TRAIN THE RANDOM FOREST MODEL
# --------------------------------------------------

print("Training Random Forest...")

# Record the start time so we can measure training duration.
start = time.time()

# Configure the base Random Forest classifier.
base_model = RandomForestClassifier(
    n_estimators=100,       # Build 100 decision trees
    max_depth=20,           # Limit each tree to depth 20
    min_samples_leaf=2,     # Each leaf must contain at least 2 samples
    class_weight="balanced",# Adjust weights for class imbalance
    random_state=42,        # Make training reproducible
    n_jobs=-1               # Use available CPU cores
)

# Wrap the classifier so it learns each output label separately.
#
# If there are 8 labels, this trains 8 binary Random Forests.
# Each forest predicts whether its own defect label is present.
#
# n_jobs=1 means the wrapper runs one label-estimator job
# at a time. The individual forests can still use all CPU cores.
model = MultiOutputClassifier(base_model, n_jobs=1)

# Learn patterns from the training data only.
model.fit(X_train, y_train)

# Display the training time in minutes.
print(f"Training time: {(time.time() - start) / 60:.2f} minutes")


# --------------------------------------------------
# 5. GET VALIDATION PROBABILITIES
# --------------------------------------------------

print("Tuning thresholds on validation data...")

# Predict the probability of the positive class (label = 1)
# for each label, using the validation data.
#
# model.estimators_ contains one fitted classifier per label.
# predict_proba(X_val) typically returns two columns:
#   column 0 = probability of class 0
#   column 1 = probability of class 1
#
# We select column 1 because we want the probability
# that each defect label is present.
#
# np.column_stack combines the probabilities into a matrix:
# rows = wafers, columns = defect labels.
val_probabilities = np.column_stack([
    estimator.predict_proba(X_val)[:, 1]
    for estimator in model.estimators_
])


# --------------------------------------------------
# 6. TUNE A SEPARATE THRESHOLD FOR EACH LABEL
# --------------------------------------------------

thresholds = []

# Loop through every output label.
for j in range(y.shape[1]):

    # Start with the standard probability threshold of 0.5.
    best_threshold = 0.5

    # A value below any valid F1 score, so the first
    # evaluated threshold can become the current best.
    best_f1 = -1.0

    # Try thresholds from 0.10 through 0.90 in steps of 0.05.
    for threshold in np.arange(0.10, 0.91, 0.05):

        # Convert probabilities into binary predictions.
        #
        # Example:
        # probability = 0.72, threshold = 0.60 -> predict 1
        # probability = 0.45, threshold = 0.60 -> predict 0
        val_pred = (
            val_probabilities[:, j] >= threshold
        ).astype(int)

        # Calculate F1 for this label on validation data.
        # zero_division=0 avoids a warning if precision or
        # recall cannot be defined because of zero positives.
        score = f1_score(
            y_val[:, j],
            val_pred,
            zero_division=0
        )

        # Keep the threshold if it produces a better F1 score.
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    # Store the best threshold for this label.
    thresholds.append(best_threshold)

    # Print the chosen threshold and its validation F1.
    print(
        f"Label {j}: threshold={best_threshold:.2f}, "
        f"validation F1={best_f1:.3f}"
    )


# --------------------------------------------------
# 7. EVALUATE ON THE TEST SET
# --------------------------------------------------

# Threshold selection is finished before this evaluation.
# The test set is reserved for final performance measurement.
print("\nEvaluating test set...")

# Calculate the positive-class probabilities for test wafers.
test_probabilities = np.column_stack([
    estimator.predict_proba(X_test)[:, 1]
    for estimator in model.estimators_
])

# Apply each label's own tuned threshold.
#
# NumPy broadcasts the threshold vector across the rows.
# Each probability column is compared with its matching
# label threshold.
y_pred = (
    test_probabilities >= np.array(thresholds)
).astype(int)


# --------------------------------------------------
# 8. PRINT OVERALL PERFORMANCE METRICS
# --------------------------------------------------

print("\n===== RANDOM FOREST RESULTS =====")

# Macro F1: calculate F1 for each label and average equally.
# Useful for seeing how the model performs across defect types.
print(
    f"Macro F1: "
    f"{f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}"
)

# Micro F1: combine true positives, false positives, and
# false negatives across all labels before calculating F1.
print(
    f"Micro F1: "
    f"{f1_score(y_test, y_pred, average='micro', zero_division=0):.4f}"
)

# Hamming loss: fraction of individual label decisions
# that are incorrect. Lower is better.
print(f"Hamming loss: {hamming_loss(y_test, y_pred):.4f}")

# Display the final threshold selected for every label.
print("\nThresholds:", np.round(thresholds, 2))

# Display elapsed time since training began.
# This includes training, threshold tuning, and evaluation.
print(f"Total elapsed time: {(time.time() - start) / 60:.2f} minutes")


# --------------------------------------------------
# 9. PRINT METRICS FOR EACH LABEL
# --------------------------------------------------

print("\n===== PER-LABEL TEST REPORT =====")

# Show precision, recall, F1-score, and support for each label.
#
# precision = of predicted positives, how many were correct?
# recall    = of actual positives, how many did we detect?
# f1-score  = harmonic mean of precision and recall
# support   = number of actual positive examples for that label
#
# The names below are generic. Replace them with verified
# defect names if you know the exact label-column order.
print(classification_report(
    y_test,
    y_pred,
    zero_division=0,
    target_names=[f"Label {i}" for i in range(y_test.shape[1])]
))


# --------------------------------------------------
# 10. SAVE THE MODEL AND ITS CONFIGURATION
# --------------------------------------------------

# Save the fitted classifiers and thresholds together.
# The application can load this file later without retraining.
joblib.dump(
    {
        "model": model,  # Trained multi-output Random Forest
        "thresholds": np.array(thresholds),  # One per label
        "input_shape": (52, 52),  # Expected wafer-map dimensions
        "labels": [f"Label {i}" for i in range(8)]
    },
    "wafer_random_forest.joblib"
)

print("\nSaved model to wafer_random_forest.joblib")