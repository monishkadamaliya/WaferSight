"""Train and compare candidate models on one common train/test split.

Usage:
    python run_experiments.py
    python run_experiments.py --data Wafer_Map_Datasets.npz --seed 42
    python run_experiments.py --models "Decision Tree" "Logistic Regression"
    python run_experiments.py --quick        (small smoke test, not for reporting)

Outputs (next to this script by default):
    model_results.csv        one row per model, test-set metrics
    per_label_results.csv    precision, recall, F1 for every model and label
    experiment_meta.json     split sizes, seed, library versions, procedure
    models/<name>.joblib     fitted artifact for each model, used by appv3.py
"""

import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from wafer_utils import (
    DEFAULT_LABELS,
    MODEL_NAMES,
    label_scores,
    load_npz,
    preprocess,
    slugify,
)

THRESHOLD_GRID = np.linspace(0.05, 0.95, 91)


# ------------------------------------------------------------
# 1) Candidate models
# ------------------------------------------------------------

def build_models(seed, n_jobs, rf_trees, pca_components, svm_solver="liblinear"):
    """Return {name: (unfitted estimator, settings text)}."""
    return {
        "Random Forest": (
            MultiOutputClassifier(
                RandomForestClassifier(
                    n_estimators=rf_trees,
                    n_jobs=n_jobs,
                    random_state=seed,
                )
            ),
            f"{rf_trees} trees, one forest per label",
        ),
        "Decision Tree": (
            MultiOutputClassifier(
                DecisionTreeClassifier(max_depth=20, random_state=seed)
            ),
            "max_depth=20, one tree per label",
        ),
        "Logistic Regression": (
            Pipeline([
                ("scale", StandardScaler()),
                ("clf", MultiOutputClassifier(
                    LogisticRegression(max_iter=200),
                    n_jobs=n_jobs,
                )),
            ]),
            "StandardScaler, L2, max_iter=200, one model per label",
        ),
        "Linear SVM": (
            # liblinear is single threaded and copies the data into its own
            # format, so it runs one label at a time (n_jobs=1). Running it in
            # parallel workers can exhaust memory and crash on Windows.
            MultiOutputClassifier(
                SGDClassifier(
                    loss="hinge", alpha=1e-4, max_iter=1000,
                    tol=1e-3, random_state=seed,
                ),
                n_jobs=1,
            )
            if svm_solver == "sgd"
            else MultiOutputClassifier(
                LinearSVC(C=0.1, dual=False, random_state=seed),
                n_jobs=1,
            ),
            (
                "Linear SVM (hinge loss, SGD solver, alpha=1e-4), "
                "sigmoid of margin as score"
                if svm_solver == "sgd"
                else "LinearSVC C=0.1 (liblinear) on the 0/0.5/1 inputs, "
                     "sigmoid of margin as score"
            ),
        ),
        "PCA + Logistic Regression": (
            Pipeline([
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=pca_components, random_state=seed)),
                ("clf", MultiOutputClassifier(
                    LogisticRegression(max_iter=300),
                    n_jobs=n_jobs,
                )),
            ]),
            f"StandardScaler, PCA({pca_components}), L2 logistic regression",
        ),
    }


# ------------------------------------------------------------
# 2) Splitting and threshold tuning
# ------------------------------------------------------------

def combo_ids(labels):
    """One integer per distinct label combination, used to stratify."""
    powers = 1 << np.arange(labels.shape[1])
    return labels @ powers


def safe_split(indices, strata, test_size, seed):
    """Stratified split, falling back to a plain split if a class is tiny."""
    try:
        return train_test_split(
            indices, test_size=test_size,
            random_state=seed, stratify=strata[indices],
        )
    except ValueError:
        return train_test_split(
            indices, test_size=test_size, random_state=seed,
        )


def tune_thresholds(y_true, scores):
    """Pick the F1-maximising threshold for each label on validation data."""
    thresholds = np.full(y_true.shape[1], 0.5)

    for k in range(y_true.shape[1]):
        if y_true[:, k].sum() == 0:
            continue

        f1_values = [
            f1_score(y_true[:, k], scores[:, k] >= t, zero_division=0)
            for t in THRESHOLD_GRID
        ]
        thresholds[k] = THRESHOLD_GRID[int(np.argmax(f1_values))]

    return thresholds


# ------------------------------------------------------------
# 3) Main experiment
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="Wafer_Map_Datasets.npz")
    parser.add_argument("--out", default=str(Path(__file__).parent))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2,
                        help="fraction of the training part used to tune thresholds")
    parser.add_argument("--rf-trees", type=int, default=100)
    parser.add_argument("--pca-components", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES,
                        choices=MODEL_NAMES)
    parser.add_argument("--svm-solver", choices=["liblinear", "sgd"],
                        default="liblinear",
                        help="sgd is a lighter linear SVM; use it if liblinear crashes")
    parser.add_argument("--quick", action="store_true",
                        help="use a small subset; for smoke tests only")
    args = parser.parse_args()

    out_dir = Path(args.out)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 3.1) Load data with the same validation the app uses
    print(f"Loading {args.data} ...")
    wafers, labels = load_npz(args.data)

    if args.quick:
        keep = np.random.default_rng(args.seed).choice(
            len(wafers), size=min(4000, len(wafers)), replace=False
        )
        wafers, labels = wafers[keep], labels[keep]
        args.rf_trees = min(args.rf_trees, 20)
        args.pca_components = min(args.pca_components, 30)

    features = preprocess(wafers)
    strata = combo_ids(labels)
    indices = np.arange(len(features))

    # 3.2) One common split for every model: train_fit / val / test
    train_idx, test_idx = safe_split(
        indices, strata, args.test_size, args.seed
    )
    fit_idx, val_idx = safe_split(
        train_idx, strata, args.val_size, args.seed
    )

    X_fit, Y_fit = features[fit_idx], labels[fit_idx]

    # Every label needs positive examples, or linear models cannot be fitted
    empty = [DEFAULT_LABELS[k] for k in np.flatnonzero(labels[fit_idx].sum(0) == 0)]
    if empty:
        raise SystemExit(
            f"No positive training examples for: {', '.join(empty)}. "
            "Use more data or drop --quick."
        )
    X_val, Y_val = features[val_idx], labels[val_idx]
    X_test, Y_test = features[test_idx], labels[test_idx]

    print(
        f"Split sizes: fit={len(fit_idx):,}  val={len(val_idx):,}  "
        f"test={len(test_idx):,}"
    )

    candidates = build_models(
        args.seed, args.n_jobs, args.rf_trees, args.pca_components,
        args.svm_solver,
    )

    # 3.7) Metadata is written first so a crash never leaves stale numbers
    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "dataset_file": str(args.data),
        "n_wafers_used": int(len(features)),
        "quick_mode": bool(args.quick),
        "seed": args.seed,
        "split": {
            "fit": int(len(fit_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(test_idx)),
            "test_fraction": args.test_size,
            "validation_fraction_of_train": args.val_size,
            "stratified_by": "label combination (falls back to random if a class is too small)",
        },
        "preprocessing": "flatten 52x52 map, divide by 2 (values 0, 0.5, 1)",
        "threshold_tuning": "per-label F1-maximising threshold on validation data",
        "evaluation": "single pass on the held-out test set",
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }

    results_path = out_dir / "model_results.csv"
    labels_path = out_dir / "per_label_results.csv"
    meta_path = out_dir / "experiment_meta.json"

    summary_rows, label_rows = [], []

    # Keep earlier results for models that are not being re-run, but only
    # when they came from the identical split (same seed and sizes)
    if results_path.exists() and meta_path.exists() and labels_path.exists():
        with open(meta_path, encoding="utf-8") as handle:
            old_meta = json.load(handle)

        same_split = (
            old_meta.get("seed") == meta["seed"]
            and old_meta.get("split") == meta["split"]
            and old_meta.get("quick_mode") == meta["quick_mode"]
        )

        if same_split:
            keep = [m for m in MODEL_NAMES if m not in args.models]
            old_summary = pd.read_csv(results_path)
            old_labels = pd.read_csv(labels_path)
            summary_rows = old_summary[old_summary["model"].isin(keep)].to_dict("records")
            label_rows = old_labels[old_labels["model"].isin(keep)].to_dict("records")
            if keep:
                print(f"Keeping earlier results for: {', '.join(m for m in keep if m in old_summary['model'].values)}")

    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    def save_tables():
        """Write the CSVs in a fixed model order after every finished model."""
        order = {m: i for i, m in enumerate(MODEL_NAMES)}
        ordered_summary = sorted(summary_rows, key=lambda r: order.get(r["model"], 99))
        ordered_labels = sorted(label_rows, key=lambda r: order.get(r["model"], 99))
        pd.DataFrame(ordered_summary).to_csv(results_path, index=False)
        pd.DataFrame(ordered_labels).to_csv(labels_path, index=False)

    for name in args.models:
        estimator, settings = candidates[name]
        print(f"\n[{name}] fitting ...")

        # 3.3) Fit on the fit portion only; scalers and PCA live inside
        #      the pipeline so they never see validation or test data
        model = clone(estimator)
        started = time.perf_counter()
        model.fit(X_fit, Y_fit)
        train_time = time.perf_counter() - started

        # 3.4) Tune thresholds on validation data, never on test data
        thresholds = tune_thresholds(Y_val, label_scores(model, X_val))

        # 3.5) Evaluate once on the untouched test set
        predictions = (
            label_scores(model, X_test) >= thresholds
        ).astype(int)

        macro = f1_score(Y_test, predictions, average="macro", zero_division=0)
        micro = f1_score(Y_test, predictions, average="micro", zero_division=0)
        ham = hamming_loss(Y_test, predictions)
        subset = accuracy_score(Y_test, predictions)

        print(
            f"  macro F1={macro:.4f}  micro F1={micro:.4f}  "
            f"hamming={ham:.4f}  time={train_time:.1f}s"
        )

        summary_rows.append({
            "model": name,
            "macro_f1": macro,
            "micro_f1": micro,
            "hamming_loss": ham,
            "subset_accuracy": subset,
            "train_time_s": train_time,
            "settings": settings,
        })

        precision, recall, f1, support = precision_recall_fscore_support(
            Y_test, predictions, zero_division=0
        )

        for k, label in enumerate(DEFAULT_LABELS):
            label_rows.append({
                "model": name,
                "label": label,
                "precision": precision[k],
                "recall": recall[k],
                "f1": f1[k],
                "support": int(support[k]),
                "threshold": float(thresholds[k]),
            })

        # 3.6) Save a self-describing artifact for the dashboard
        joblib.dump(
            {
                "name": name,
                "model": model,
                "thresholds": thresholds,
                "labels": DEFAULT_LABELS,
                "settings": settings,
            },
            models_dir / f"{slugify(name)}.joblib",
            compress=3,
        )
        save_tables()
        print(f"  saved results so far ({len(summary_rows)} models)")

    print("\nDone. Wrote model_results.csv, per_label_results.csv, "
          "experiment_meta.json and models/*.joblib")


if __name__ == "__main__":
    main()
