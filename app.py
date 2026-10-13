
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import BoundaryNorm, ListedColormap

# ============================================================
# WAFERSIGHT — WAFER DEFECT INTELLIGENCE
# ============================================================

st.set_page_config(
    page_title="WaferSight | Defect Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path("wafer_random_forest.joblib")
DATA_PATH = Path("Wafer_Map_Datasets.npz")

EXPECTED_SHAPE = (52, 52)
EXPECTED_LABELS = 8

# ============================================================
# CUSTOM DESIGN
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono&family=Manrope:wght@400;500;600;700;800&display=swap');

    .stApp {
        background: radial-gradient(
            ellipse at top left,
            #152b3c 0%,
            #0a1020 42%
        );
        color: #eef4ff;
        font-family: 'Manrope', sans-serif;
    }

    [data-testid="stSidebar"] {
        background: #0e172a;
        border-right: 1px solid #293957;
    }

    [data-testid="stHeader"] {
        background: rgba(10, 16, 32, 0.75);
    }

    .hero {
        padding: 28px 30px;
        border: 1px solid #30415f;
        border-radius: 20px;
        background: linear-gradient(120deg, #172d42, #111b30);
        margin-bottom: 24px;
    }

    .eyebrow,
    .section-tag {
        color: #78e4ce;
        font-family: 'DM Mono', monospace;
        letter-spacing: 2px;
        font-size: 12px;
    }

    .hero h1 {
        color: #f4f7ff;
        font-size: clamp(28px, 3vw, 40px);
        font-weight: 800;
        margin: 8px 0;
    }

    .hero p {
        color: #a9b9d5;
        font-size: 15px;
        margin-bottom: 0;
        line-height: 1.7;
    }

    div[data-testid="stMetric"] {
        background: #111b2f;
        border: 1px solid #293957;
        border-radius: 14px;
        padding: 16px;
    }

    div[data-testid="stMetricLabel"] {
        color: #9bacc9;
    }

    div[data-testid="stMetricValue"] {
        color: #f4f7ff;
    }

    .stButton > button[kind="primary"] {
        background: #78e4ce;
        color: #07131e;
        border: 0;
        border-radius: 10px;
        font-weight: 800;
        min-height: 45px;
    }

    .stButton > button[kind="primary"]:hover {
        background: #a0f3e2;
        color: #07131e;
    }

    .stButton > button {
        border-radius: 10px;
    }

    div[data-testid="stProgressBar"] > div > div {
        background-color: #78e4ce;
    }

    hr {
        border-color: #293957;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #293957;
        border-radius: 12px;
    }

    .footer {
        text-align: center;
        color: #8295b8;
        font-size: 12px;
        padding: 20px 0;
        letter-spacing: 1px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL AND DATASET
# ============================================================

@st.cache_resource
def load_model(path_string):
    return joblib.load(path_string)


@st.cache_data
def load_dataset(path_string):
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

    if labels.ndim != 2:
        raise ValueError(
            f"Expected a 2D label array; received {labels.shape}."
        )

    if len(wafers) != len(labels):
        raise ValueError(
            f"Dataset mismatch: {len(wafers)} maps but "
            f"{len(labels)} label rows."
        )

    if labels.shape[1] != EXPECTED_LABELS:
        raise ValueError(
            f"Expected {EXPECTED_LABELS} labels per wafer; "
            f"received {labels.shape[1]}."
        )

    if len(wafers) == 0:
        raise ValueError("The dataset contains no wafer maps.")

    return wafers, labels.astype(np.int32)


# ============================================================
# MODEL PREDICTION
# ============================================================

def positive_class_probability(estimator, features):
    """
    Return probability for class 1 without assuming that
    class 1 occupies probability column index 1.
    """
    if not hasattr(estimator, "predict_proba"):
        raise TypeError(
            "A fitted estimator with predict_proba is required."
        )

    probabilities = estimator.predict_proba(features)
    classes = np.asarray(
        getattr(estimator, "classes_", [])
    )

    if probabilities.ndim != 2 or probabilities.shape[0] != 1:
        raise ValueError(
            "Unexpected predict_proba output shape."
        )

    positive_indices = np.flatnonzero(classes == 1)

    if len(positive_indices) == 0:
        if len(classes) == 1 and classes[0] == 0:
            return 0.0

        raise ValueError(
            f"Could not locate positive class 1 in classes {classes}."
        )

    return float(
        probabilities[0, int(positive_indices[0])]
    )


def predict(wafer, saved, expected_label_count):
    if tuple(wafer.shape) != EXPECTED_SHAPE:
        raise ValueError(
            f"Expected a 52 × 52 wafer map; received {wafer.shape}."
        )

    if not isinstance(saved, dict):
        raise ValueError(
            "The saved model artifact must be a dictionary."
        )

    if "model" not in saved or "thresholds" not in saved:
        raise ValueError(
            "Model artifact must contain 'model' and 'thresholds'."
        )

    model = saved["model"]
    estimators = getattr(model, "estimators_", None)

    if estimators is None:
        raise ValueError(
            "Saved model does not expose fitted estimators_."
        )

    if len(estimators) != expected_label_count:
        raise ValueError(
            f"Model contains {len(estimators)} estimators, "
            f"but {expected_label_count} labels are expected."
        )

    thresholds = np.asarray(
        saved["thresholds"], dtype=float
    ).reshape(-1)

    if len(thresholds) != expected_label_count:
        raise ValueError(
            f"Found {len(thresholds)} thresholds; "
            f"expected {expected_label_count}."
        )

    if (
        not np.all(np.isfinite(thresholds))
        or np.any((thresholds < 0) | (thresholds > 1))
    ):
        raise ValueError(
            "Decision thresholds must be between 0 and 1."
        )

    # Preserve the original training preprocessing.
    features = (
        wafer.reshape(1, -1).astype(np.float32) / 2.0
    )

    probabilities = np.array(
        [
            positive_class_probability(estimator, features)
            for estimator in estimators
        ],
        dtype=float,
    )

    predictions = (
        probabilities >= thresholds
    ).astype(np.int32)

    return probabilities, thresholds, predictions


# ============================================================
# WAFER VISUALIZATION
# ============================================================

def render_wafer_map(wafer):
    # Dataset encoding assumed by this application:
    # 0 = blank, 1 = pass, 2 = fail.
    cmap = ListedColormap(
        ["#111b2f", "#78e4ce", "#ff6b7a"]
    )

    norm = BoundaryNorm(
        [-0.5, 0.5, 1.5, 2.5],
        cmap.N,
    )

    fig, ax = plt.subplots(
        figsize=(5.2, 5.2),
        facecolor="#111b2f",
    )

    ax.set_facecolor("#111b2f")

    image = ax.imshow(
        wafer,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    ax.set_title(
        "Selected wafer",
        color="white",
        pad=14,
        fontsize=13,
        fontweight="bold",
    )

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_color("#34445f")

    colorbar = fig.colorbar(
        image,
        ax=ax,
        ticks=[0, 1, 2],
        boundaries=[-0.5, 0.5, 1.5, 2.5],
        fraction=0.046,
        pad=0.04,
    )

    colorbar.ax.set_yticklabels(
        ["Blank", "Pass", "Fail"]
    )

    colorbar.ax.tick_params(
        colors="white",
        labelsize=9,
    )

    colorbar.outline.set_edgecolor("#34445f")

    fig.tight_layout()
    return fig


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not MODEL_PATH.exists():
    st.error(
        f"Model file not found: {MODEL_PATH.resolve()}"
    )
    st.info(
        "Place wafer_random_forest.joblib in the same "
        "folder as app.py."
    )
    st.stop()

if not DATA_PATH.exists():
    st.error(
        f"Dataset file not found: {DATA_PATH.resolve()}"
    )
    st.info(
        "Place Wafer_Map_Datasets.npz in the same "
        "folder as app.py."
    )
    st.stop()

try:
    saved = load_model(str(MODEL_PATH.resolve()))

    wafers, actual_labels_all = load_dataset(
        str(DATA_PATH.resolve())
    )

except Exception as exc:
    st.error(
        f"Could not load or validate the model/dataset: {exc}"
    )
    st.stop()


labels = list(
    saved.get(
        "labels",
        [f"Label {i + 1}" for i in range(EXPECTED_LABELS)],
    )
)

if len(labels) != EXPECTED_LABELS:
    st.error(
        f"Model metadata contains {len(labels)} labels; "
        f"expected {EXPECTED_LABELS}."
    )
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🔬 WaferSight")
    st.caption("DEFECT INTELLIGENCE PLATFORM")
    st.divider()

    st.markdown("### Inspection controls")

    sample_index = int(
        st.number_input(
            "Select wafer sample",
            min_value=0,
            max_value=len(wafers) - 1,
            value=0,
            step=1,
        )
    )

    st.caption(
        f"{len(wafers):,} wafer maps available"
    )

    st.divider()

    st.markdown("### Model information")
    st.write("**Algorithm:** Random Forest")
    st.write("**Task:** Multi-label classification")
    st.write("**Input:** 52 × 52 grid")
    st.write(f"**Outputs:** {len(labels)} labels")

    st.divider()

    st.markdown("### Dataset information")
    st.write(f"**Wafer maps:** {len(wafers):,}")
    st.write(f"**Map size:** {EXPECTED_SHAPE[0]} × {EXPECTED_SHAPE[1]}")
    st.write("**Label format:** Binary multi-label")

    st.divider()

    st.caption(
        "CNN-based recognition is planned future work."
    )


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">
            SEMICONDUCTOR · MACHINE LEARNING
        </div>
        <h1>Wafer Defect Intelligence</h1>
        <p>
            Inspect wafer maps, detect defect patterns, and
            compare model predictions with known labels using
            a trained Random Forest classifier.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TOP METRICS
# ============================================================

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Dataset size",
    f"{len(wafers):,}",
    "wafer maps",
)

m2.metric(
    "Map dimensions",
    "52 × 52",
    "grid input",
)

m3.metric(
    "Output labels",
    str(len(labels)),
    "multi-label",
)

m4.metric(
    "Current model",
    "Random Forest",
    "trained",
)


# ============================================================
# INSPECTION WORKSPACE
# ============================================================

wafer = wafers[sample_index]
actual_labels = actual_labels_all[sample_index]

st.markdown(
    '<div class="section-tag">'
    '01 / INSPECTION WORKSPACE</div>',
    unsafe_allow_html=True,
)

st.header(f"Wafer sample #{sample_index:,}")

map_col, result_col = st.columns(
    [1, 1.25],
    gap="large",
)


# -------------------- WAFER MAP --------------------

with map_col:
    st.subheader("Wafer map")

    st.caption(
        "52 × 52 pixel grid · Discrete dataset states"
    )

    fig = render_wafer_map(wafer)

    st.pyplot(
        fig,
        use_container_width=True,
    )

    plt.close(fig)

    with st.expander("How to read this map"):
        st.markdown(
            """
            - **Blank:** value `0`
            - **Pass:** value `1`
            - **Fail:** value `2`

            Colors represent the dataset's encoded states,
            not a physical wafer photograph.
            """
        )


# -------------------- PREDICTION ANALYSIS --------------------

with result_col:
    st.subheader("Prediction analysis")

    st.caption(
        "Run the trained model on the selected wafer."
    )

    analyze = st.button(
        "▶  Analyze selected wafer",
        type="primary",
        use_container_width=True,
    )

    # Keep each result associated with its sample.
    if analyze:
        try:
            probabilities, thresholds, predictions = predict(
                wafer,
                saved,
                len(labels),
            )

            st.session_state["wafer_result"] = {
                "index": sample_index,
                "probabilities": probabilities,
                "thresholds": thresholds,
                "predictions": predictions,
            }

            st.session_state.pop(
                "wafer_result_error",
                None,
            )

        except Exception as exc:
            st.session_state.pop(
                "wafer_result",
                None,
            )

            st.session_state["wafer_result_error"] = str(exc)

    if "wafer_result_error" in st.session_state:
        st.error(
            "Prediction failed: "
            + st.session_state["wafer_result_error"]
        )

    result = st.session_state.get("wafer_result")

    # Never display results from a different sample.
    if (
        result is not None
        and result.get("index") != sample_index
    ):
        result = None

        st.info(
            "Sample changed. Click Analyze selected wafer "
            "to update results."
        )

    if result is None:
        st.info(
            "Select a wafer and click Analyze selected wafer "
            "to begin."
        )

    else:
        probabilities = result["probabilities"]
        thresholds = result["thresholds"]
        predictions = result["predictions"]

        detected = [
            labels[i]
            for i, value in enumerate(predictions)
            if value == 1
        ]

        if detected:
            st.success(
                "Predicted labels: " + ", ".join(detected)
            )
        else:
            st.warning(
                "No labels crossed their decision thresholds."
            )

        st.markdown("#### Individual label scores")

        for label, probability, prediction, threshold in zip(
            labels,
            probabilities,
            predictions,
            thresholds,
        ):
            first, second, third = st.columns(
                [1.2, 1.1, 0.5]
            )

            with first:
                st.write(f"**{label}**")

            with second:
                st.write(
                    "🟢 Detected"
                    if prediction
                    else "⚪ Not detected"
                )

            with third:
                st.write(f"**{probability:.0%}**")

            st.progress(
                float(np.clip(probability, 0, 1))
            )

            st.caption(
                f"Decision threshold: {threshold:.2f}"
            )

        st.caption(
            "Scores are model-estimated probabilities and "
            "may not be calibrated confidence values."
        )


# ============================================================
# GROUND TRUTH COMPARISON
# ============================================================

st.divider()

st.markdown(
    '<div class="section-tag">'
    '02 / VERIFICATION</div>',
    unsafe_allow_html=True,
)

st.header("Ground truth vs prediction")

st.write(
    "Compare model predictions with the known labels "
    "for this sample."
)

result = st.session_state.get("wafer_result")

if (
    result is None
    or result.get("index") != sample_index
):
    st.info(
        "Run the analysis above to populate the comparison table."
    )

else:
    predictions = np.asarray(
        result["predictions"],
        dtype=np.int32,
    ).reshape(-1)

    probabilities = np.asarray(
        result["probabilities"],
        dtype=float,
    ).reshape(-1)

    actual = np.asarray(
        actual_labels
    ).reshape(-1)

    if (
        len(actual) != len(labels)
        or len(predictions) != len(labels)
    ):
        st.error(
            "Cannot compare labels: actual and predicted "
            "label counts do not match."
        )

    elif not np.isin(actual, [0, 1]).all():
        st.error(
            "Ground-truth labels must contain only 0 and 1 "
            "for this comparison."
        )

    else:
        table = pd.DataFrame(
            {
                "Defect label": labels,
                "Actual": [
                    "Present" if value else "Absent"
                    for value in actual
                ],
                "Predicted": [
                    "Present" if value else "Absent"
                    for value in predictions
                ],
                "Model score": [
                    f"{p:.1%}"
                    for p in probabilities
                ],
                "Match": [
                    "✓ Match"
                    if int(a) == int(p)
                    else "✕ Different"
                    for a, p in zip(actual, predictions)
                ],
            }
        )

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
        )

        correct = int(
            np.sum(actual == predictions)
        )

        total = len(labels)

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Correct label decisions",
            f"{correct}/{total}",
        )

        c2.metric(
            "Label-decision agreement",
            f"{correct / total:.1%}",
        )

        c3.metric(
            "All labels match",
            "Yes"
            if np.array_equal(actual, predictions)
            else "No",
        )

        st.caption(
            "These values describe this selected sample only; "
            "they are not overall test-set metrics."
        )


# ============================================================
# ABOUT THE PROJECT
# ============================================================

st.divider()

with st.expander(
    "About this project and its limitations"
):
    st.markdown(
        """
        **Objective:** Detect wafer defect patterns using
        multi-label machine learning.

        **Current model:** Random Forest with a separate
        decision threshold for each label.

        **Dataset:** Wafer maps represented as 52 × 52 grids.

        **Evaluation:** Report overall performance using held-out
        test data and metrics such as macro F1, micro F1,
        and Hamming loss.

        **Important limitation:** A selected sample may have been
        included in model training. This dashboard demonstrates
        the workflow and does not replace evaluation on unseen wafers.

        **Future scope:** CNN-based recognition and validation
        on independent wafer data.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        WAFERSIGHT · ACADEMIC PROJECT DEMONSTRATION
    </div>
    """,
    unsafe_allow_html=True,
)
