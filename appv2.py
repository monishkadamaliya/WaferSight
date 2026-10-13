
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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

# Cell values used by the dataset and by the painter
CELL_BLANK, CELL_PASS, CELL_FAIL = 0, 1, 2

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
# WAFER PAINTER HELPERS
# ============================================================

# Short codes in the dataset's one-hot order
CODE_ORDER = ["C", "D", "EL", "ER", "L", "NF", "S", "R"]

NAME_TO_CODE = {
    "center": "C", "donut": "D", "edgeloc": "EL", "edgering": "ER",
    "loc": "L", "nearfull": "NF", "scratch": "S", "random": "R",
}

# The 29 mixed types listed in Description.pdf, plus the singles
MIXED_TYPES = [
    "C+EL", "C+ER", "C+L", "C+S", "D+EL", "D+ER", "D+L", "D+S",
    "EL+L", "EL+S", "ER+L", "ER+S", "L+S",
    "C+EL+L", "C+EL+S", "C+ER+L", "C+ER+S", "C+L+S", "D+EL+L",
    "D+EL+S", "D+ER+L", "D+ER+S", "D+L+S", "EL+L+S", "ER+L+S",
    "C+L+EL+S", "C+L+ER+S", "D+L+EL+S", "D+L+ER+S",
]

VALID_COMBOS = {frozenset(c.split("+")) for c in MIXED_TYPES}
VALID_COMBOS |= {frozenset([c]) for c in CODE_ORDER}

# Paint tools: label shown to the user -> cell value
PAINT_TOOLS = {
    "Fail (red)": CELL_FAIL,
    "Pass (green)": CELL_PASS,
    "Blank": CELL_BLANK,
}


@st.cache_data
def wafer_outline(_wafers):
    """True where a die position exists on a real wafer."""
    # 1) take up to ~2000 wafers spread evenly across the dataset
    step = max(1, len(_wafers) // 2000)
    sample = _wafers[::step]

    # 2) a position belongs to the wafer if it is non-blank in most samples
    return (sample != CELL_BLANK).mean(axis=0) > 0.5


@st.cache_data
def normal_scatter_rate(_wafers, _labels):
    """Average fraction of failed dies on defect-free wafers."""
    # 1) find wafers whose label vector is all zeros (Normal)
    normal_idx = np.flatnonzero(_labels.sum(axis=1) == 0)

    if len(normal_idx) == 0:
        return 0.05

    # 2) use up to 500 of them, count failed dies inside the outline
    normal_idx = normal_idx[:500]
    outline = wafer_outline(_wafers)
    subset = _wafers[normal_idx][:, outline]

    return float((subset == CELL_FAIL).mean())


def add_scatter(grid, rate):
    """Randomly turn passing dies into failing dies at the given rate."""
    rng = np.random.default_rng()
    out = grid.copy()
    flip = (out == CELL_PASS) & (rng.random(out.shape) < rate)
    out[flip] = CELL_FAIL
    return out


def commit_painter_grid(new_grid):
    """Save the old grid for Undo, store the new one, reset the chart."""
    history = st.session_state["painter_history"]
    history.append(st.session_state["painter_grid"].copy())
    del history[:-40]

    st.session_state["painter_grid"] = np.asarray(
        new_grid, dtype=np.int32
    )
    # A new chart key clears the old selection box
    st.session_state["painter_chart_n"] += 1


def paint_figure(grid, outline, locked):
    """Heatmap of the grid with an invisible, selectable point layer."""
    # 1) discrete colours: 0 blank, 1 pass, 2 fail
    colorscale = [
        [0.0, "#1b2740"], [1 / 3, "#1b2740"],
        [1 / 3, "#35c46b"], [2 / 3, "#35c46b"],
        [2 / 3, "#ef4444"], [1.0, "#ef4444"],
    ]

    z = grid.astype(float)
    if locked:
        z[~outline] = np.nan  # cells outside the wafer are not drawn

    # 2) selectable points at the centre of each paintable cell
    rows, cols = np.mgrid[0:52, 0:52]
    keep = outline if locked else np.ones_like(outline, dtype=bool)

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=z,
            zmin=0,
            zmax=2,
            colorscale=colorscale,
            showscale=False,
            xgap=1,
            ygap=1,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=cols[keep],
            y=rows[keep],
            mode="markers",
            marker=dict(size=9, opacity=0),
            selected=dict(marker=dict(opacity=0)),
            unselected=dict(marker=dict(opacity=0)),
            hovertemplate="col %{x}, row %{y}<extra></extra>",
            showlegend=False,
        )
    )

    # 3) square plot area, y axis top-down like the dataset arrays
    fig.update_layout(
        dragmode="lasso",
        height=560,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#111b2f",
        plot_bgcolor="#0b1220",
        xaxis=dict(visible=False, range=[-0.5, 51.5], constrain="domain"),
        yaxis=dict(
            visible=False,
            range=[51.5, -0.5],
            scaleanchor="x",
            constrain="domain",
        ),
    )

    return fig


def label_to_code(label):
    """Map a model label such as 'Edge_Loc' or 'Edge_Loc(EL)' to a short code."""
    text = str(label)
    key = "".join(ch for ch in text.lower() if ch.isalnum())

    if key in NAME_TO_CODE:
        return NAME_TO_CODE[key]

    upper = text.strip().upper()
    if upper in CODE_ORDER:
        return upper

    # Fall back to a code in brackets, for example "Edge_Loc(EL)"
    if "(" in text and ")" in text:
        inner = text[text.rfind("(") + 1:text.rfind(")")].strip().upper()
        if inner in CODE_ORDER:
            return inner

    return None


def pattern_summary(detected_labels):
    """Return (pattern name, note) for the predicted label set."""
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
        "so it may be a sign the drawing is unrealistic."
    )


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
# WAFER PAINTER
# ============================================================

st.divider()

st.markdown(
    '<div class="section-tag">'
    '03 / CUSTOM WAFER TEST</div>',
    unsafe_allow_html=True,
)

st.header("Paint your own wafer")

st.write(
    "Mark failed dies in red and passing dies in green, then run "
    "the same model on your drawing. The grid uses the dataset "
    "encoding: 0 blank, 1 pass, 2 fail."
)

# 1) Dataset-derived helpers: wafer outline and typical scatter rate
outline = wafer_outline(wafers)
scatter_default = normal_scatter_rate(wafers, actual_labels_all)

# 2) First-run state
if "painter_grid" not in st.session_state:
    st.session_state["painter_grid"] = outline.astype(np.int32)
    st.session_state["painter_history"] = []
    st.session_state["painter_chart_n"] = 0

# 3) Quick-start controls
scatter_rate = st.slider(
    "Scatter failure rate for random single-die failures",
    min_value=0.0,
    max_value=0.30,
    value=float(np.clip(round(scatter_default, 3), 0.0, 0.30)),
    step=0.005,
    format="%.3f",
    help=(
        "Default is the average failed-die rate measured on "
        "Normal wafers in your dataset."
    ),
)

b1, b2, b3, b4 = st.columns(4)

if b1.button("Clean wafer", use_container_width=True):
    commit_painter_grid(outline.astype(np.int32))
    st.rerun()

if b2.button(
    f"Load wafer #{sample_index:,}",
    use_container_width=True,
):
    commit_painter_grid(wafers[sample_index])
    st.rerun()

if b3.button("Add scatter", use_container_width=True):
    commit_painter_grid(
        add_scatter(st.session_state["painter_grid"], scatter_rate)
    )
    st.rerun()

if b4.button(
    "Undo",
    use_container_width=True,
    disabled=not st.session_state["painter_history"],
):
    st.session_state["painter_grid"] = (
        st.session_state["painter_history"].pop()
    )
    st.session_state["painter_chart_n"] += 1
    st.rerun()

paint_col, paint_result_col = st.columns([1, 1.1], gap="large")

# 4) Painter chart
with paint_col:
    tool_name = st.radio(
        "Paint with",
        list(PAINT_TOOLS),
        horizontal=True,
    )

    locked = st.checkbox(
        "Lock painting to the wafer outline",
        value=True,
    )

    st.caption(
        "Use the lasso or box icon in the chart toolbar, drag over "
        "the dies you want, and they are painted when you release. "
        "Clicking a single die paints that die."
    )

    painted = st.session_state["painter_grid"]

    event = st.plotly_chart(
        paint_figure(painted, outline, locked),
        key=f"painter_chart_{st.session_state['painter_chart_n']}",
        on_select="rerun",
        selection_mode=("points", "box", "lasso"),
        config={"displaylogo": False},
    )

    # 5) Apply the selection to the grid, then reset the chart
    selected = []
    if event and event.get("selection"):
        selected = event["selection"].get("points", [])

    if selected:
        new_grid = painted.copy()
        value = PAINT_TOOLS[tool_name]

        for point in selected:
            if point.get("curve_number", 1) != 1:
                continue

            col = int(round(point["x"]))
            row = int(round(point["y"]))

            if not (0 <= row < 52 and 0 <= col < 52):
                continue

            if locked and not outline[row, col]:
                continue

            new_grid[row, col] = value

        commit_painter_grid(new_grid)
        st.rerun()

    fails = int(np.sum(painted == CELL_FAIL))
    passes = int(np.sum(painted == CELL_PASS))

    k1, k2, k3 = st.columns(3)
    k1.metric("Failed dies", f"{fails:,}")
    k2.metric("Passing dies", f"{passes:,}")
    k3.metric(
        "Fail rate",
        f"{fails / max(1, fails + passes):.1%}",
    )

# 6) Run the model on the painted grid
with paint_result_col:
    st.subheader("Model verdict on your drawing")

    run_paint = st.button(
        "Analyze painted wafer",
        type="primary",
        use_container_width=True,
    )

    if run_paint:
        try:
            probs, thr, preds = predict(
                painted,
                saved,
                len(labels),
            )

            st.session_state["paint_result"] = {
                "grid_bytes": painted.tobytes(),
                "probabilities": probs,
                "thresholds": thr,
                "predictions": preds,
            }
            st.session_state.pop("paint_error", None)

        except Exception as exc:
            st.session_state.pop("paint_result", None)
            st.session_state["paint_error"] = str(exc)

    if "paint_error" in st.session_state:
        st.error(
            "Prediction failed: " + st.session_state["paint_error"]
        )

    paint_result = st.session_state.get("paint_result")

    if paint_result is None:
        st.info(
            "Paint a wafer, then click Analyze painted wafer."
        )

    else:
        if paint_result["grid_bytes"] != painted.tobytes():
            st.warning(
                "The drawing changed since this analysis. "
                "Click Analyze painted wafer to refresh."
            )

        detected = [
            name
            for name, flag in zip(labels, paint_result["predictions"])
            if flag == 1
        ]

        pattern_name, pattern_note = pattern_summary(detected)

        if detected:
            st.success(f"Predicted pattern: {pattern_name}")
        else:
            st.warning(
                "Predicted pattern: Normal "
                "(no label crossed its threshold)"
            )

        if pattern_note:
            st.caption(pattern_note)

        score_table = pd.DataFrame(
            {
                "Defect label": labels,
                "Score": paint_result["probabilities"],
                "Threshold": paint_result["thresholds"],
                "Detected": [
                    "Yes" if v == 1 else "No"
                    for v in paint_result["predictions"]
                ],
            }
        )

        st.dataframe(
            score_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.2f",
                ),
                "Threshold": st.column_config.NumberColumn(
                    "Threshold",
                    format="%.2f",
                ),
            },
        )

    st.caption(
        "A hand-painted map is not a real probe result. The model "
        "learned from dataset wafers, so realistic shapes with "
        "scattered single-die failures give the most meaningful "
        "output."
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
