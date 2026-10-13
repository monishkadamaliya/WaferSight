
import time
import warnings
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    classification_report
)
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def main():
    print("Loading dataset...")

    data = np.load("Wafer_Map_Datasets.npz")
    wafers = data["arr_0"]
    labels = data["arr_1"]

    X = wafers.reshape(len(wafers), -1).astype(np.float32) / 2.0
    y = labels.astype(np.int32)

    del wafers, labels, data

    # Reproduce the same train/validation/test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    del X, X_temp, y_temp

    experiments = [
        {"alpha": 0.0001, "max_iter": 100},
        {"alpha": 0.001, "max_iter": 100},
        {"alpha": 0.00001, "max_iter": 100},
    ]

    results = []

    for config in experiments:
        print(f"\nTraining: {config}")
        start = time.time()

        model = OneVsRestClassifier(
            SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=config["alpha"],
                max_iter=config["max_iter"],
                tol=1e-3,
                class_weight="balanced",
                random_state=42
            ),
            n_jobs=1
        )

        model.fit(X_train, y_train)

        # Tune thresholds using validation data only
        val_scores = model.decision_function(X_val)
        thresholds = []
        val_f1_scores = []

        for j in range(y.shape[1]):
            best_threshold = 0.0
            best_f1 = -1.0

            for threshold in np.arange(-2.0, 2.01, 0.1):
                predictions = (
                    val_scores[:, j] >= threshold
                ).astype(int)

                score = f1_score(
                    y_val[:, j],
                    predictions,
                    zero_division=0
                )

                if score > best_f1:
                    best_f1 = score
                    best_threshold = threshold

            thresholds.append(best_threshold)
            val_f1_scores.append(best_f1)

        # Evaluate the held-out test set
        test_scores = model.decision_function(X_test)
        y_pred = (
            test_scores >= np.array(thresholds)
        ).astype(int)

        macro_f1 = f1_score(
            y_test, y_pred,
            average="macro",
            zero_division=0
        )

        micro_f1 = f1_score(
            y_test, y_pred,
            average="micro",
            zero_division=0
        )

        hamming = hamming_loss(y_test, y_pred)
        elapsed = (time.time() - start) / 60

        results.append({
            "config": config,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "hamming_loss": hamming,
            "thresholds": thresholds,
            "val_f1_scores": val_f1_scores,
            "y_pred": y_pred,
            "model": model,
            "time": elapsed
        })

        print(f"Macro F1:     {macro_f1:.4f}")
        print(f"Micro F1:     {micro_f1:.4f}")
        print(f"Hamming loss: {hamming:.4f}")
        print(f"Time:         {elapsed:.2f} minutes")

    # Compare all configurations
    print("\n===== EXPERIMENT COMPARISON =====")

    for result in results:
        print(
            f"alpha={result['config']['alpha']:<8} "
            f"Macro F1={result['macro_f1']:.4f}  "
            f"Micro F1={result['micro_f1']:.4f}  "
            f"Hamming={result['hamming_loss']:.4f}"
        )

    # Select the model with the highest test macro F1
    # for diagnostic purposes only. Do not use this step
    # to claim an unbiased final test score.
    best = max(results, key=lambda r: r["macro_f1"])

    print("\n===== BEST CONFIGURATION =====")
    print("Configuration:", best["config"])
    print(f"Macro F1: {best['macro_f1']:.4f}")
    print(f"Micro F1: {best['micro_f1']:.4f}")
    print(f"Hamming loss: {best['hamming_loss']:.4f}")
    print("Thresholds:", np.round(best["thresholds"], 2))

    print("\n===== PER-LABEL TEST RESULTS =====")
    print(classification_report(
        y_test,
        best["y_pred"],
        zero_division=0,
        target_names=[
            f"Label {i}" for i in range(y_test.shape[1])
        ]
    ))

    print("\n===== VALIDATION F1 BY LABEL =====")
    for i, score in enumerate(best["val_f1_scores"]):
        print(f"Label {i}: {score:.4f}")


if __name__ == "__main__":
    main()
