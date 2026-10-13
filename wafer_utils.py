"""Shared helpers used by both appv3.py and run_experiments.py.

Keeping preprocessing and scoring in one place guarantees that training
and prediction treat a wafer map in exactly the same way.
"""

import re

import numpy as np
from sklearn.pipeline import Pipeline

# ------------------------------------------------------------
# Dataset constants
# ------------------------------------------------------------

EXPECTED_SHAPE = (52, 52)
EXPECTED_LABELS = 8

# Cell values in the dataset: 0 blank, 1 pass, 2 fail
CELL_BLANK, CELL_PASS, CELL_FAIL = 0, 1, 2

# Label order of the 8-dimensional arr_1 vector (from Description.pdf)
DEFAULT_LABELS = [
    "Center", "Donut", "Edge_Loc", "Edge_Ring",
    "Loc", "Near_Full", "Scratch", "Random",
]
CODE_ORDER = ["C", "D", "EL", "ER", "L", "NF", "S", "R"]

NAME_TO_CODE = {
    "center": "C", "donut": "D", "edgeloc": "EL", "edgering": "ER",
    "loc": "L", "nearfull": "NF", "scratch": "S", "random": "R",
}

# The 29 mixed types listed in Description.pdf, plus the 8 singles
MIXED_TYPES = [
    "C+EL", "C+ER", "C+L", "C+S", "D+EL", "D+ER", "D+L", "D+S",
    "EL+L", "EL+S", "ER+L", "ER+S", "L+S",
    "C+EL+L", "C+EL+S", "C+ER+L", "C+ER+S", "C+L+S", "D+EL+L",
    "D+EL+S", "D+ER+L", "D+ER+S", "D+L+S", "EL+L+S", "ER+L+S",
    "C+L+EL+S", "C+L+ER+S", "D+L+EL+S", "D+L+ER+S",
]
VALID_COMBOS = {frozenset(c.split("+")) for c in MIXED_TYPES}
VALID_COMBOS |= {frozenset([c]) for c in CODE_ORDER}

# Candidate models compared by run_experiments.py
MODEL_NAMES = [
    "Random Forest",
    "Decision Tree",
    "Logistic Regression",
    "Linear SVM",
    "PCA + Logistic Regression",
]


def slugify(name):
    """'PCA + Logistic Regression' -> 'pca_logistic_regression'."""
    return "_".join(re.sub(r"[^a-z0-9]+", " ", name.lower()).split())


# ------------------------------------------------------------
# Data loading and preprocessing
# ------------------------------------------------------------

def load_npz(path_string):
    """Load and validate wafer maps (arr_0) and labels (arr_1)."""
    with np.load(path_string) as data:
        if "arr_0" not in data or "arr_1" not in data:
            raise ValueError(
                "Dataset must contain arr_0 (wafer maps) "
                "and arr_1 (labels)."
            )
        wafers = data["arr_0"]
        labels = data["arr_1"]

    if wafers.ndim != 3 or tuple(wafers.shape[1:]) != EXPECTED_SHAPE:
        raise ValueError(
            f"Expected wafer maps with shape (N, 52, 52); "
            f"received {wafers.shape}."
        )
    if labels.ndim != 2 or labels.shape[1] != EXPECTED_LABELS:
        raise ValueError(
            f"Expected labels with shape (N, {EXPECTED_LABELS}); "
            f"received {labels.shape}."
        )
    if len(wafers) != len(labels):
        raise ValueError(
            f"Dataset mismatch: {len(wafers)} maps but "
            f"{len(labels)} label rows."
        )
    if len(wafers) == 0:
        raise ValueError("The dataset contains no wafer maps.")

    return wafers, labels.astype(np.int32)


def preprocess(wafers):
    """Flatten and scale wafer maps. Used for training AND prediction."""
    array = np.asarray(wafers)

    if array.ndim == 2:
        array = array[np.newaxis]

    if tuple(array.shape[1:]) != EXPECTED_SHAPE:
        raise ValueError(
            f"Expected 52 x 52 wafer maps; received {array.shape}."
        )

    return array.reshape(len(array), -1).astype(np.float32) / 2.0


# ------------------------------------------------------------
# Scoring
# ------------------------------------------------------------

def label_scores(model, features, expected_labels=EXPECTED_LABELS):
    """Return an (n_samples, n_labels) array of scores in [0, 1].

    Works for a MultiOutputClassifier or a Pipeline that ends in one.
    Models with predict_proba use the class-1 probability. Models that
    only have decision_function (LinearSVC) use a sigmoid of the margin,
    which is a monotonic score and not a calibrated probability.
    """
    if isinstance(model, Pipeline):
        transformed = model[:-1].transform(features)
        head = model[-1]
    else:
        transformed = features
        head = model

    estimators = getattr(head, "estimators_", None)

    if estimators is None:
        raise ValueError("Saved model does not expose fitted estimators_.")

    if len(estimators) != expected_labels:
        raise ValueError(
            f"Model contains {len(estimators)} estimators, "
            f"but {expected_labels} labels are expected."
        )

    columns = []

    for estimator in estimators:
        if hasattr(estimator, "predict_proba"):
            probabilities = estimator.predict_proba(transformed)
            classes = np.asarray(getattr(estimator, "classes_", []))
            positive = np.flatnonzero(classes == 1)

            if len(positive) == 0:
                columns.append(np.zeros(len(transformed)))
            else:
                columns.append(probabilities[:, int(positive[0])])

        elif hasattr(estimator, "decision_function"):
            margin = estimator.decision_function(transformed)
            columns.append(1.0 / (1.0 + np.exp(-margin)))

        else:
            raise TypeError(
                "Each estimator needs predict_proba or decision_function."
            )

    return np.column_stack(columns).astype(float)


# ------------------------------------------------------------
# Pattern naming
# ------------------------------------------------------------

def row_to_name(flags):
    """Name a dataset label row using the fixed dataset label order."""
    codes = [c for c, f in zip(CODE_ORDER, flags) if int(f) == 1]
    return "+".join(codes) if codes else "Normal"


def label_to_code(label):
    """Map a model label such as 'Edge_Loc' or 'Edge_Loc(EL)' to a code."""
    text = str(label)
    key = "".join(ch for ch in text.lower() if ch.isalnum())

    if key in NAME_TO_CODE:
        return NAME_TO_CODE[key]

    upper = text.strip().upper()
    if upper in CODE_ORDER:
        return upper

    if "(" in text and ")" in text:
        inner = text[text.rfind("(") + 1:text.rfind(")")].strip().upper()
        if inner in CODE_ORDER:
            return inner

    return None


def pattern_summary(detected_labels):
    """Return (pattern name, note) for a predicted label set."""
    if not detected_labels:
        return "Normal", None

    codes = [label_to_code(name) for name in detected_labels]

    if any(code is None for code in codes):
        return " + ".join(detected_labels), None

    ordered = [c for c in CODE_ORDER if c in codes]
    name = "+".join(ordered)

    if frozenset(ordered) in VALID_COMBOS:
        return name, "This combination exists among the dataset's 38 types."

    return name, (
        "This combination does not occur among the dataset's 38 types, "
        "so the drawing or input may be unrealistic."
    )
