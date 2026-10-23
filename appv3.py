"""WaferSight: wafer defect intelligence dashboard (Streamlit)."""

import io
import json
from pathlib import Path
from huggingface_hub import hf_hub_download
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from wafer_utils import (
    CELL_BLANK,
    CELL_FAIL,
    CELL_PASS,
    CODE_ORDER,
    DEFAULT_LABELS,
    EXPECTED_LABELS,
    EXPECTED_SHAPE,
    MODEL_NAMES,
    label_scores,
    load_npz,
    pattern_summary,
    preprocess,
    row_to_name,
    slugify,
)

# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="WaferSight | Defect Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).parent
ORIGINAL_MODEL = BASE / "wafer_random_forest.joblib"
MODELS_DIR = BASE / "models"
DATA_PATH = BASE / "Wafer_Map_Datasets.npz"
RESULTS_CSV = BASE / "model_results.csv"
PER_LABEL_CSV = BASE / "per_label_results.csv"
META_JSON = BASE / "experiment_meta.json"

PAGES = [
    "Overview",
    "Wafer prediction",
    "Model comparison",
    "Dataset exploration",
    "Methodology",
]

COLOR_BLANK, COLOR_PASS, COLOR_FAIL = "#1b2740", "#35c46b", "#ef4444"
COLOR_ACCENT, COLOR_MUTED = "#78e4ce", "#3a4d73"

# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono&family=Manrope:wght@400;500;600;700;800&display=swap');

    .stApp {
        background: radial-gradient(ellipse at top left, #152b3c 0%, #0a1020 42%);
        color: #eef4ff;
        font-family: 'Manrope', sans-serif;
    }
    [data-testid="stSidebar"] { background: #0e172a; border-right: 1px solid #293957; }
    [data-testid="stHeader"] { background: rgba(10, 16, 32, 0.75); }

    .hero {
        padding: 26px 30px; border: 1px solid #30415f; border-radius: 12px;
        background: #121f33; margin-bottom: 22px;
    }
    .eyebrow, .section-tag {
        color: #78e4ce; font-family: 'DM Mono', monospace;
        letter-spacing: 2px; font-size: 12px;
    }
    .hero h1 {
        color: #f4f7ff; font-size: clamp(26px, 3vw, 38px);
        font-weight: 800; margin: 8px 0;
    }
    .hero p { color: #a9b9d5; font-size: 15px; margin-bottom: 0; line-height: 1.7; }

    div[data-testid="stMetric"] {
        background: #111b2f; border: 1px solid #293957;
        border-radius: 8px; padding: 14px 16px;
    }
    div[data-testid="stMetricLabel"] { color: #9bacc9; }
    div[data-testid="stMetricValue"] { color: #f4f7ff; }

    .stButton > button { border-radius: 6px; }
    .stButton > button[kind="primary"] {
        background: #78e4ce; color: #07131e; border: 0;
        font-weight: 800; min-height: 45px;
    }
    .stButton > button[kind="primary"]:hover { background: #a0f3e2; color: #07131e; }

    button[data-baseweb="tab"] { font-weight: 700; }
    hr { border-color: #293957; }
    [data-testid="stDataFrame"] { border: 1px solid #293957; border-radius: 8px; }

    .note {
        border-left: 3px solid #78e4ce; background: #101a2d;
        padding: 12px 16px; border-radius: 4px; color: #b9c7e0;
        font-size: 14px; line-height: 1.65;
    }
    .footer {
        text-align: center; color: #8295b8; font-size: 12px;
        padding: 20px 0; letter-spacing: 1px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def page_header(tag, title, text):
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{tag}</div>
            <h1>{title}</h1>
            <p>{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# LOADING (all cached)
# ============================================================

@st.cache_resource
def load_model(path_string):
    return joblib.load(path_string)


@st.cache_data
def load_dataset(path_string):
    return load_npz(path_string)


@st.cache_data
def _read_results(mtime):
    summary = pd.read_csv(RESULTS_CSV)
    per_label = pd.read_csv(PER_LABEL_CSV) if PER_LABEL_CSV.exists() else None

    meta = None
    if META_JSON.exists():
        with open(META_JSON, encoding="utf-8") as handle:
            meta = json.load(handle)

    return summary, per_label, meta


def read_results():
    """Return (summary, per_label, meta) or (None, None, None)."""
    if not RESULTS_CSV.exists():
        return None, None, None
    return _read_results(RESULTS_CSV.stat().st_mtime)


def available_models():
    """Map a display name to a model artifact path."""
    found = {}

    if ORIGINAL_MODEL.exists():
        found["Random Forest (original artifact)"] = ORIGINAL_MODEL

    if MODELS_DIR.exists():
        names = {slugify(n): n for n in MODEL_NAMES}
        order = {slugify(n): i for i, n in enumerate(MODEL_NAMES)}

        for path in sorted(
            MODELS_DIR.glob("*.joblib"),
            key=lambda p: order.get(p.stem, 99),
        ):
            found[f"{names.get(path.stem, path.stem)} (comparison split)"] = path

    return found


# ============================================================
# DATASET STATISTICS (cached, computed in chunks to save memory)
# ============================================================

@st.cache_data
def wafer_outline(_wafers):
    """True where a die position exists on a real wafer."""
    step = max(1, len(_wafers) // 2000)
    return (_wafers[::step] != CELL_BLANK).mean(axis=0) > 0.5


@st.cache_data
def normal_scatter_rate(_wafers, _labels):
    """Average fraction of failed dies on defect-free wafers."""
    normal_idx = np.flatnonzero(_labels.sum(axis=1) == 0)

    if len(normal_idx) == 0:
        return 0.05

    outline = wafer_outline(_wafers)
    subset = _wafers[normal_idx[:500]][:, outline]
    return float((subset == CELL_FAIL).mean())


@st.cache_data
def dataset_stats(_wafers, _labels):
    """Per-wafer fail rate, per-label mean failure maps, pattern names."""
    n = len(_wafers)
    fail_rate = np.zeros(n)
    fail_maps = np.zeros((EXPECTED_LABELS, 52 * 52))

    for start in range(0, n, 4000):
        chunk = _wafers[start:start + 4000]
        failed = (chunk == CELL_FAIL).astype(np.float32)
        present = (chunk != CELL_BLANK).reshape(len(chunk), -1).sum(axis=1)

        fail_rate[start:start + len(chunk)] = (
            failed.reshape(len(chunk), -1).sum(axis=1)
            / np.maximum(present, 1)
        )
        fail_maps += (
            _labels[start:start + 4000].T.astype(np.float32)
            @ failed.reshape(len(chunk), -1)
        )

    counts = np.maximum(_labels.sum(axis=0), 1)
    fail_maps = (fail_maps / counts[:, None]).reshape(EXPECTED_LABELS, 52, 52)

    pattern_names = np.array([row_to_name(row) for row in _labels])

    return fail_rate, fail_maps, pattern_names


# ============================================================
# FIGURE HELPERS
# ============================================================

DISCRETE_SCALE = [
    [0.0, COLOR_BLANK], [1 / 3, COLOR_BLANK],
    [1 / 3, COLOR_PASS], [2 / 3, COLOR_PASS],
    [2 / 3, COLOR_FAIL], [1.0, COLOR_FAIL],
]


def style_fig(fig, height=360):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d6ee", family="Manrope, sans-serif"),
    )
    fig.update_xaxes(gridcolor="#233250", zerolinecolor="#233250")
    fig.update_yaxes(gridcolor="#233250", zerolinecolor="#233250")
    return fig


def wafer_heatmap(grid, height=420, title=None):
    """Discrete green / red wafer map."""
    fig = go.Figure(
        go.Heatmap(
            z=np.asarray(grid, dtype=float),
            zmin=0, zmax=2,
            colorscale=DISCRETE_SCALE,
            showscale=False,
            xgap=1, ygap=1,
            hovertemplate="col %{x}, row %{y}, state %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=30 if title else 0, b=0),
        title=dict(text=title or "", font=dict(size=13)),
        paper_bgcolor="#111b2f",
        plot_bgcolor="#111b2f",
        font=dict(color="#eef4ff"),
        xaxis=dict(visible=False, constrain="domain"),
        yaxis=dict(visible=False, autorange="reversed",
                   scaleanchor="x", constrain="domain"),
    )
    return fig


def map_legend():
    st.caption(
        "Green is a passing die (1), red is a failed die (2), "
        "dark is blank wafer area (0)."
    )


def label_frequency_chart(y_all, names):
    counts = y_all.sum(axis=0)
    fig = go.Figure(
        go.Bar(
            x=names, y=counts, text=[f"{c:,}" for c in counts],
            textposition="outside", marker_color=COLOR_ACCENT,
            hovertemplate="%{x}: %{y:,} wafers<extra></extra>",
        )
    )
    fig.update_yaxes(title="Wafers containing the label")
    return style_fig(fig, 340)


def defect_count_chart(y_all):
    per_wafer = y_all.sum(axis=1)
    values = np.arange(0, int(per_wafer.max()) + 1)
    counts = [int((per_wafer == v).sum()) for v in values]
    names = ["Normal" if v == 0 else f"{v} defect" + ("s" if v > 1 else "")
             for v in values]

    fig = go.Figure(
        go.Bar(
            x=names, y=counts, text=[f"{c:,}" for c in counts],
            textposition="outside", marker_color=COLOR_ACCENT,
            hovertemplate="%{x}: %{y:,} wafers<extra></extra>",
        )
    )
    fig.update_yaxes(title="Wafers")
    return style_fig(fig, 340)


# ============================================================
# PREDICTION HELPERS
# ============================================================

def run_model(grid, saved):
    """Validate the artifact and score one 52 x 52 wafer."""
    grid = np.asarray(grid)

    if tuple(grid.shape) != EXPECTED_SHAPE:
        raise ValueError(f"Expected a 52 x 52 wafer map; received {grid.shape}.")

    if not isinstance(saved, dict):
        raise ValueError("The saved model artifact must be a dictionary.")

    if "model" not in saved or "thresholds" not in saved:
        raise ValueError("Model artifact must contain 'model' and 'thresholds'.")

    thresholds = np.asarray(saved["thresholds"], dtype=float).reshape(-1)

    if len(thresholds) != EXPECTED_LABELS:
        raise ValueError(
            f"Found {len(thresholds)} thresholds; expected {EXPECTED_LABELS}."
        )

    if not np.all(np.isfinite(thresholds)) or np.any(
        (thresholds < 0) | (thresholds > 1)
    ):
        raise ValueError("Decision thresholds must be between 0 and 1.")

    # Same preprocessing as training (shared function in wafer_utils)
    probabilities = label_scores(saved["model"], preprocess(grid))[0]
    predictions = (probabilities >= thresholds).astype(np.int32)

    return probabilities, thresholds, predictions


def prediction_block(tab_key, grid, model_name, saved, labels):
    """Analyze button plus result display. Returns the valid result or None."""
    signature = (model_name, np.asarray(grid, dtype=np.int32).tobytes())
    store = st.session_state.setdefault("predictions", {})

    if st.button(
        "Analyze wafer",
        type="primary",
        key=f"analyze_{tab_key}",
        use_container_width=True,
    ):
        try:
            probs, thr, preds = run_model(grid, saved)
            store[tab_key] = {
                "signature": signature,
                "probabilities": probs,
                "thresholds": thr,
                "predictions": preds,
            }
            store.pop(f"{tab_key}_error", None)
        except Exception as exc:
            store.pop(tab_key, None)
            store[f"{tab_key}_error"] = str(exc)

    if f"{tab_key}_error" in store:
        st.error("Prediction failed: " + store[f"{tab_key}_error"])

    entry = store.get(tab_key)

    if entry is None:
        st.info("Click Analyze wafer to run the selected model.")
        return None

    if entry["signature"] != signature:
        st.info(
            "The wafer or model changed. Click Analyze wafer to update "
            "the result."
        )
        return None

    detected = [n for n, f in zip(labels, entry["predictions"]) if f == 1]
    pattern_name, note = pattern_summary(detected)

    if detected:
        st.success(f"Predicted pattern: {pattern_name}")
    else:
        st.warning("Predicted pattern: Normal (no label crossed its threshold)")

    if note:
        st.caption(note)

    table = pd.DataFrame(
        {
            "Defect label": labels,
            "Score": entry["probabilities"],
            "Threshold": entry["thresholds"],
            "Detected": ["Yes" if v == 1 else "No" for v in entry["predictions"]],
        }
    )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score", min_value=0.0, max_value=1.0, format="%.2f"
            ),
            "Threshold": st.column_config.NumberColumn(
                "Threshold", format="%.2f"
            ),
        },
    )

    st.caption(
        "Scores come from the selected model. Tree and logistic models give "
        "probability estimates that may not be calibrated; the linear SVM "
        "score is a sigmoid of its margin and is not a probability."
    )

    return entry


def comparison_table(entry, actual, labels):
    actual = np.asarray(actual).reshape(-1)
    predicted = np.asarray(entry["predictions"], dtype=np.int32).reshape(-1)

    table = pd.DataFrame(
        {
            "Defect label": labels,
            "Actual": ["Present" if v else "Absent" for v in actual],
            "Predicted": ["Present" if v else "Absent" for v in predicted],
            "Model score": [f"{p:.1%}" for p in entry["probabilities"]],
            "Match": [
                "Match" if int(a) == int(p) else "Different"
                for a, p in zip(actual, predicted)
            ],
        }
    )
    st.dataframe(table, use_container_width=True, hide_index=True)

    correct = int(np.sum(actual == predicted))
    total = len(labels)

    c1, c2, c3 = st.columns(3)
    c1.metric("Correct label decisions", f"{correct}/{total}")
    c2.metric("Label-decision agreement", f"{correct / total:.1%}")
    c3.metric(
        "All labels match",
        "Yes" if np.array_equal(actual, predicted) else "No",
    )
    st.caption(
        "These values describe this selected sample only. They are not "
        "test-set metrics, and the sample may have been used in training."
    )


def parse_upload(uploaded):
    """Read a 52 x 52 wafer map from .npy or .csv and validate it."""
    name = uploaded.name.lower()
    raw = uploaded.getvalue()

    if name.endswith(".npy"):
        array = np.load(io.BytesIO(raw), allow_pickle=False)
    elif name.endswith((".csv", ".txt")):
        array = np.loadtxt(io.StringIO(raw.decode("utf-8")), delimiter=",")
    else:
        raise ValueError("Upload a .npy or .csv file.")

    array = np.asarray(array)

    if tuple(array.shape) != EXPECTED_SHAPE:
        raise ValueError(
            f"Expected a 52 x 52 grid; received shape {array.shape}."
        )

    if not np.isfinite(array).all() or not np.isin(array, [0, 1, 2]).all():
        raise ValueError("Values must be 0 (blank), 1 (pass) or 2 (fail).")

    return array.astype(np.int32)


# ============================================================
# PAINTER HELPERS
# ============================================================

PAINT_TOOLS = {
    "Fail (red)": CELL_FAIL,
    "Pass (green)": CELL_PASS,
    "Blank": CELL_BLANK,
}


def add_scatter(grid, rate):
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

    st.session_state["painter_grid"] = np.asarray(new_grid, dtype=np.int32)
    st.session_state["painter_chart_n"] += 1


def paint_figure(grid, outline, locked):
    """Heatmap plus an invisible selectable point layer for lasso painting."""
    z = grid.astype(float)
    if locked:
        z[~outline] = np.nan

    rows, cols = np.mgrid[0:52, 0:52]
    keep = outline if locked else np.ones_like(outline, dtype=bool)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=z, zmin=0, zmax=2, colorscale=DISCRETE_SCALE,
            showscale=False, xgap=1, ygap=1, hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cols[keep], y=rows[keep], mode="markers",
            marker=dict(size=9, opacity=0),
            selected=dict(marker=dict(opacity=0)),
            unselected=dict(marker=dict(opacity=0)),
            hovertemplate="col %{x}, row %{y}<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        dragmode="lasso", height=560,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#111b2f", plot_bgcolor="#0b1220",
        xaxis=dict(visible=False, range=[-0.5, 51.5], constrain="domain"),
        yaxis=dict(visible=False, range=[51.5, -0.5],
                   scaleanchor="x", constrain="domain"),
    )
    return fig


# ============================================================
# PAGE: OVERVIEW
# ============================================================

def page_overview(wafers, y_all, models):
    page_header(
        "SEMICONDUCTOR · MACHINE LEARNING",
        "Wafer Defect Intelligence",
        "Analyze wafer maps, predict defect patterns with trained "
        "machine-learning models, and compare algorithms on a common "
        "evaluation split. This is a multi-label task: one wafer can carry "
        "several defect labels at once.",
    )

    summary, _, _ = read_results()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Wafer maps", f"{len(wafers):,}")
    m2.metric("Map size", "52 x 52")
    m3.metric("Defect labels", str(EXPECTED_LABELS))

    if summary is None:
        m4.metric("Models compared", "Not run yet")
    else:
        m4.metric("Models compared", str(len(summary)))

    if summary is not None:
        best = summary.loc[summary["macro_f1"].idxmax()]
        st.markdown(
            f'<div class="note">Best macro F1 on the held-out test set: '
            f'<b>{best["model"]}</b> at {best["macro_f1"]:.4f}. '
            f'Values come from <code>model_results.csv</code>; see the '
            f'Model comparison page for the full set of metrics.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="note">No experiment results found yet. Run '
            '<code>python run_experiments.py</code> to train the candidate '
            'models and fill the Model comparison page.</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Defect label distribution")
        st.plotly_chart(label_frequency_chart(y_all, DEFAULT_LABELS))

    with right:
        st.subheader("Simultaneous defects per wafer")
        st.plotly_chart(defect_count_chart(y_all))

    st.divider()
    st.subheader("Project status")

    n_artifacts = len(list(MODELS_DIR.glob("*.joblib"))) if MODELS_DIR.exists() else 0
    status = pd.DataFrame(
        {
            "Component": [
                "Dataset (Wafer_Map_Datasets.npz)",
                "Original Random Forest artifact",
                "Experiment results (model_results.csv)",
                "Comparison model artifacts (models/)",
            ],
            "Status": [
                "Found" if DATA_PATH.exists() else "Missing",
                "Found" if ORIGINAL_MODEL.exists() else "Missing",
                "Found" if RESULTS_CSV.exists() else "Not generated",
                f"{n_artifacts} found" if n_artifacts else "Not generated",
            ],
        }
    )
    st.dataframe(status, use_container_width=True, hide_index=True)


# ============================================================
# PAGE: WAFER PREDICTION
# ============================================================

def painter_tab(wafers, outline, scatter_default, run_prediction):
    st.write(
        "Mark failed dies in red and passing dies in green, then run the "
        "selected model on your drawing. The grid uses the dataset "
        "encoding: 0 blank, 1 pass, 2 fail."
    )

    if "painter_grid" not in st.session_state:
        st.session_state["painter_grid"] = outline.astype(np.int32)
        st.session_state["painter_history"] = []
        st.session_state["painter_chart_n"] = 0

    scatter_rate = st.slider(
        "Scatter failure rate for random single-die failures",
        min_value=0.0, max_value=0.30,
        value=float(np.clip(round(scatter_default, 3), 0.0, 0.30)),
        step=0.005, format="%.3f",
        help="Default is the average failed-die rate measured on Normal "
             "wafers in your dataset.",
    )

    sample_index = int(st.session_state.get("sample_index", 0))
    b1, b2, b3, b4 = st.columns(4)

    if b1.button("Clean wafer", use_container_width=True):
        commit_painter_grid(outline.astype(np.int32))
        st.rerun()

    if b2.button(f"Load wafer #{sample_index:,}", use_container_width=True):
        commit_painter_grid(wafers[sample_index])
        st.rerun()

    if b3.button("Add scatter", use_container_width=True):
        commit_painter_grid(
            add_scatter(st.session_state["painter_grid"], scatter_rate)
        )
        st.rerun()

    if b4.button(
        "Undo", use_container_width=True,
        disabled=not st.session_state["painter_history"],
    ):
        st.session_state["painter_grid"] = st.session_state["painter_history"].pop()
        st.session_state["painter_chart_n"] += 1
        st.rerun()

    paint_col, result_col = st.columns([1, 1.1], gap="large")

    with paint_col:
        tool_name = st.radio("Paint with", list(PAINT_TOOLS), horizontal=True)
        locked = st.checkbox("Lock painting to the wafer outline", value=True)

        st.caption(
            "Use the lasso or box icon in the chart toolbar, drag over the "
            "dies you want, and they are painted when you release. "
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
        k3.metric("Fail rate", f"{fails / max(1, fails + passes):.1%}")

    with result_col:
        st.subheader("Model verdict on your drawing")
        run_prediction("paint", painted)

        st.caption(
            "A hand-painted map is not a real probe result. The models "
            "learned from dataset wafers, so realistic shapes with "
            "scattered single-die failures give the most meaningful output."
        )


def page_prediction(wafers, y_all, models):
    page_header(
        "01 / PREDICTION",
        "Wafer prediction",
        "Choose a model, then inspect a dataset wafer, upload your own "
        "52 x 52 map, or paint a wafer by hand and see which defect "
        "labels the model predicts.",
    )

    if not models:
        st.error(
            "No trained model found. Place wafer_random_forest.joblib next "
            "to appv3.py, or run run_experiments.py to create models/."
        )
        return

    model_name = st.selectbox("Model", list(models))

    try:
        saved = load_model(str(models[model_name].resolve()))
        labels = list(saved.get("labels", DEFAULT_LABELS))
        if len(labels) != EXPECTED_LABELS:
            raise ValueError(
                f"Model metadata contains {len(labels)} labels; "
                f"expected {EXPECTED_LABELS}."
            )
    except Exception as exc:
        st.error(f"Could not load this model: {exc}")
        return

    def run_prediction(tab_key, grid, actual=None):
        return prediction_block(tab_key, grid, model_name, saved, labels)

    outline = wafer_outline(wafers)
    scatter_default = normal_scatter_rate(wafers, y_all)

    tab_ds, tab_up, tab_paint = st.tabs(
        ["Dataset wafer", "Upload a wafer", "Paint a wafer"]
    )

    # ---------------- dataset wafer ----------------
    with tab_ds:
        sample_index = int(
            st.number_input(
                "Select wafer sample",
                min_value=0, max_value=len(wafers) - 1,
                value=0, step=1, key="sample_index",
            )
        )
        st.caption(f"{len(wafers):,} wafer maps available")

        wafer = wafers[sample_index]
        actual = y_all[sample_index]

        map_col, result_col = st.columns([1, 1.25], gap="large")

        with map_col:
            st.subheader(f"Wafer sample #{sample_index:,}")
            st.plotly_chart(wafer_heatmap(wafer))
            map_legend()

        with result_col:
            st.subheader("Prediction analysis")
            entry = run_prediction("dataset", wafer)

        st.subheader("Ground truth vs prediction")

        if entry is None:
            st.info("Run the analysis above to populate the comparison table.")
        else:
            comparison_table(entry, actual, labels)

    # ---------------- upload ----------------
    with tab_up:
        st.write(
            "Upload a 52 x 52 grid as a .npy file or a comma-separated "
            ".csv file. Values must be 0 (blank), 1 (pass) or 2 (fail)."
        )

        uploaded = st.file_uploader(
            "Wafer map file", type=["npy", "csv", "txt"], key="wafer_upload"
        )

        with st.expander("Need a test file? Export a dataset wafer"):
            export_index = int(
                st.number_input(
                    "Dataset wafer to export",
                    min_value=0, max_value=len(wafers) - 1,
                    value=0, step=1, key="export_index",
                )
            )
            buffer = io.BytesIO()
            np.save(buffer, wafers[export_index].astype(np.int8))
            st.download_button(
                f"Download wafer #{export_index:,} as .npy",
                data=buffer.getvalue(),
                file_name=f"wafer_{export_index}.npy",
                mime="application/octet-stream",
            )

        if uploaded is None:
            st.info("Upload a file to begin.")
        else:
            try:
                grid = parse_upload(uploaded)
            except Exception as exc:
                st.error(f"Could not use this file: {exc}")
            else:
                up_map, up_result = st.columns([1, 1.25], gap="large")

                with up_map:
                    st.subheader("Uploaded wafer")
                    st.plotly_chart(wafer_heatmap(grid))
                    map_legend()

                with up_result:
                    st.subheader("Prediction analysis")
                    run_prediction("upload", grid)

    # ---------------- painter ----------------
    with tab_paint:
        painter_tab(wafers, outline, scatter_default, run_prediction)


# ============================================================
# PAGE: MODEL COMPARISON
# ============================================================

METRICS = {
    "Macro F1": ("macro_f1", True),
    "Micro F1": ("micro_f1", True),
    "Hamming loss": ("hamming_loss", False),
    "Subset accuracy": ("subset_accuracy", True),
    "Training time (s)": ("train_time_s", False),
}


def page_comparison():
    page_header(
        "02 / EVALUATION",
        "Model comparison",
        "Every model below was trained on the same split, had its "
        "thresholds tuned on validation data only, and was scored once on "
        "the held-out test set.",
    )

    summary, per_label, meta = read_results()

    if summary is None:
        st.info(
            "No results yet. Run the experiment script from the project "
            "folder, then reload this page:"
        )
        st.code("python run_experiments.py", language="bash")
        return

    if meta and meta.get("quick_mode"):
        st.warning(
            "These results came from quick mode (a small subset). Re-run "
            "without --quick before reporting any numbers."
        )

    # ---- best-per-metric cards
    best_cols = st.columns(4)
    cards = [
        ("Best macro F1", "macro_f1", True, "{:.4f}"),
        ("Best micro F1", "micro_f1", True, "{:.4f}"),
        ("Lowest Hamming loss", "hamming_loss", False, "{:.4f}"),
        ("Fastest training", "train_time_s", False, "{:.1f} s"),
    ]
    for col, (title, key, higher, fmt) in zip(best_cols, cards):
        idx = summary[key].idxmax() if higher else summary[key].idxmin()
        col.metric(title, summary.loc[idx, "model"], fmt.format(summary.loc[idx, key]))

    st.divider()

    # ---- chart for one metric
    left, right = st.columns([1.2, 1], gap="large")

    with left:
        metric_name = st.selectbox("Metric", list(METRICS))
        column, higher = METRICS[metric_name]

        best_idx = summary[column].idxmax() if higher else summary[column].idxmin()
        colors = [
            COLOR_ACCENT if i == best_idx else COLOR_MUTED
            for i in summary.index
        ]

        fig = go.Figure(
            go.Bar(
                x=summary["model"], y=summary[column],
                marker_color=colors,
                text=[f"{v:.4f}" if column != "train_time_s" else f"{v:.1f}"
                      for v in summary[column]],
                textposition="outside",
                hovertemplate="%{x}: %{y:.4f}<extra></extra>",
            )
        )
        fig.update_yaxes(title=metric_name)
        st.plotly_chart(style_fig(fig, 380))
        st.caption(
            "The highlighted bar is the best model for this metric "
            + ("(higher is better)." if higher else "(lower is better).")
        )

    with right:
        st.markdown("##### Training time")
        fig = go.Figure(
            go.Bar(
                y=summary["model"], x=summary["train_time_s"],
                orientation="h", marker_color=COLOR_ACCENT,
                text=[f"{v:.1f} s" for v in summary["train_time_s"]],
                textposition="outside",
            )
        )
        fig.update_xaxes(title="Seconds")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style_fig(fig, 380))

    # ---- full table with a "best on" column
    st.subheader("All metrics")

    table = summary.copy()
    best_on = {name: [] for name in table["model"]}

    for display, (column, higher) in METRICS.items():
        idx = summary[column].idxmax() if higher else summary[column].idxmin()
        best_on[summary.loc[idx, "model"]].append(display)

    table["Best on"] = [", ".join(best_on[m]) or "none" for m in table["model"]]
    table = table.rename(
        columns={
            "model": "Model", "macro_f1": "Macro F1", "micro_f1": "Micro F1",
            "hamming_loss": "Hamming loss", "subset_accuracy": "Subset accuracy",
            "train_time_s": "Train time (s)", "settings": "Settings",
        }
    )
    st.dataframe(
        table, use_container_width=True, hide_index=True,
        column_config={
            "Macro F1": st.column_config.NumberColumn(format="%.4f"),
            "Micro F1": st.column_config.NumberColumn(format="%.4f"),
            "Hamming loss": st.column_config.NumberColumn(format="%.4f"),
            "Subset accuracy": st.column_config.NumberColumn(format="%.4f"),
            "Train time (s)": st.column_config.NumberColumn(format="%.1f"),
        },
    )

    # ---- per-label performance
    if per_label is not None:
        st.subheader("Per-label performance")
        score_name = st.radio(
            "Score", ["f1", "precision", "recall"], horizontal=True
        )

        pivot = per_label.pivot(index="model", columns="label", values=score_name)
        pivot = pivot.reindex(summary["model"])

        fig = go.Figure(
            go.Heatmap(
                z=pivot.values, x=list(pivot.columns), y=list(pivot.index),
                zmin=0, zmax=1,
                colorscale=[[0, "#1b2740"], [1, COLOR_ACCENT]],
                text=np.round(pivot.values, 3), texttemplate="%{text}",
                colorbar=dict(title=score_name),
                hovertemplate="%{y}, %{x}: %{z:.3f}<extra></extra>",
            )
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style_fig(fig, 120 + 55 * len(pivot)))
        st.caption(
            "Lower cells show the defect labels that are harder to identify."
        )

    # ---- setup
    if meta:
        with st.expander("Evaluation setup"):
            split = meta.get("split", {})
            st.markdown(
                f"""
                - **Seed:** {meta.get("seed")}
                - **Split:** {split.get("fit", 0):,} fit, {split.get("validation", 0):,} validation, {split.get("test", 0):,} test
                - **Stratified by:** {split.get("stratified_by")}
                - **Preprocessing:** {meta.get("preprocessing")}
                - **Thresholds:** {meta.get("threshold_tuning")}
                - **Evaluation:** {meta.get("evaluation")}
                - **Generated:** {meta.get("created")}
                """
            )

    st.markdown(
        '<div class="note">The original wafer_random_forest.joblib artifact is '
        'not in this table because its training split is unknown, so it '
        'cannot be compared fairly. The Random Forest row here is retrained '
        'under the common split.</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE: DATASET EXPLORATION
# ============================================================

def page_exploration(wafers, y_all):
    page_header(
        "03 / DATA",
        "Dataset exploration",
        "Size, class balance, and typical patterns in the mixed-type "
        "wafer map dataset.",
    )

    fail_rate, fail_maps, pattern_names = dataset_stats(wafers, y_all)

    counts = y_all.sum(axis=0)
    n_patterns = len(np.unique(pattern_names))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Wafer maps", f"{len(wafers):,}")
    m2.metric("Map size", "52 x 52")
    m3.metric("Distinct patterns", str(n_patterns))
    m4.metric("Label imbalance", f"{counts.max() / max(counts.min(), 1):.1f} : 1")
    st.caption(
        "Label imbalance is the most frequent label count divided by the "
        "least frequent label count. Mixed wafers count toward every label "
        "they carry."
    )

    st.divider()

    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Label frequencies")
        st.plotly_chart(label_frequency_chart(y_all, DEFAULT_LABELS))

    with right:
        st.subheader("Simultaneous defects per wafer")
        st.plotly_chart(defect_count_chart(y_all))

    st.divider()
    st.subheader("Wafers per pattern class")

    pattern_counts = (
        pd.Series(pattern_names).value_counts().rename_axis("Pattern")
        .reset_index(name="Wafers")
    )
    fig = go.Figure(
        go.Bar(
            y=pattern_counts["Pattern"], x=pattern_counts["Wafers"],
            orientation="h", marker_color=COLOR_ACCENT,
            hovertemplate="%{y}: %{x:,} wafers<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(style_fig(fig, max(360, 22 * len(pattern_counts))))

    st.subheader("Average failed-die rate per pattern")
    rate = (
        pd.DataFrame({"Pattern": pattern_names, "Rate": fail_rate})
        .groupby("Pattern")["Rate"].mean().sort_values(ascending=False)
    )
    fig = go.Figure(
        go.Bar(
            y=rate.index, x=rate.values * 100, orientation="h",
            marker_color=COLOR_FAIL,
            hovertemplate="%{y}: %{x:.1f}% of dies failed<extra></extra>",
        )
    )
    fig.update_xaxes(title="Average failed dies (% of dies on the wafer)")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(style_fig(fig, max(360, 22 * len(rate))))

    st.divider()
    st.subheader("Where each defect label fails dies")

    fig = make_subplots(
        rows=2, cols=4, subplot_titles=DEFAULT_LABELS,
        horizontal_spacing=0.02, vertical_spacing=0.12,
    )
    for k in range(EXPECTED_LABELS):
        fig.add_trace(
            go.Heatmap(
                z=fail_maps[k], zmin=0, zmax=1,
                colorscale=[[0, "#0b1220"], [1, COLOR_FAIL]],
                showscale=(k == 0), colorbar=dict(title="Fail rate", len=0.9),
                hovertemplate="col %{x}, row %{y}: %{z:.2f}<extra></extra>",
            ),
            row=k // 4 + 1, col=k % 4 + 1,
        )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, autorange="reversed")
    st.plotly_chart(style_fig(fig, 520))
    st.caption(
        "Each map averages the failed-die indicator over all wafers that "
        "carry that label, so mixed wafers also contribute to every label "
        "they contain."
    )

    st.divider()
    st.subheader("Example wafers")

    options = list(pattern_counts["Pattern"])
    choice = st.selectbox("Pattern", options)

    if "example_seed" not in st.session_state:
        st.session_state["example_seed"] = 0

    if st.button("Show different examples"):
        st.session_state["example_seed"] += 1

    members = np.flatnonzero(pattern_names == choice)
    rng = np.random.default_rng(st.session_state["example_seed"])
    picks = rng.choice(members, size=min(3, len(members)), replace=False)

    cols = st.columns(3)
    for col, idx in zip(cols, picks):
        with col:
            st.plotly_chart(
                wafer_heatmap(wafers[idx], height=300, title=f"{choice} (#{idx:,})")
            )
    map_legend()


# ============================================================
# PAGE: METHODOLOGY
# ============================================================

def page_methodology():
    page_header(
        "04 / METHOD",
        "Model information and methodology",
        "How the task is defined, how the models are trained and "
        "evaluated, and how to read the predictions.",
    )

    st.subheader("Task")
    st.markdown(
        """
        Each wafer map is a 52 x 52 grid. A die is blank (0), passing (1),
        or failed (2). The label is an 8-dimensional binary vector marking
        which basic defect patterns are present: Center, Donut, Edge_Loc,
        Edge_Ring, Loc, Near_Full, Scratch and Random. A wafer can have
        several labels at once (the dataset contains 38 classes: 1 normal,
        8 single and 29 mixed), so this is **multi-label classification**
        and not a single-choice problem. A wafer with no label set is
        reported as Normal.
        """
    )

    st.subheader("Preprocessing")
    st.markdown(
        """
        The grid is flattened to 2,704 values and divided by 2, giving 0,
        0.5 and 1. Training and prediction call the same function
        (`preprocess` in `wafer_utils.py`), so they cannot drift apart.
        Where a model uses a scaler or PCA, it sits inside the model
        pipeline and is fitted on training data only.
        """
    )

    st.subheader("Algorithms compared")
    st.markdown(
        """
        | Model | Idea |
        |---|---|
        | Random Forest | Ensemble of decision trees trained on resampled data; the project baseline |
        | Decision Tree | A single tree of feature splits; easy to explain |
        | Logistic Regression | Linear model with a probability output |
        | Linear SVM | Maximum-margin linear classifier (LinearSVC, or hinge-loss SGD if that option was used; see the Settings column) |
        | PCA + Logistic Regression | Reduces the 2,704 inputs to principal components first |

        Each model is wrapped so that one binary classifier is trained per
        label (the standard one-vs-rest style for multi-label data).
        """
    )

    st.subheader("Evaluation protocol")
    st.markdown(
        """
        1. One stratified train / test split with a fixed seed, shared by all models.
        2. A validation slice is taken from the training part. The decision
           threshold of each label is tuned there to maximise that label's F1.
        3. The held-out test set is scored once with the tuned thresholds.

        **Macro F1** averages F1 over labels with equal weight.
        **Micro F1** pools all label decisions before computing F1.
        **Hamming loss** is the share of individual label decisions that are
        wrong (lower is better). **Subset accuracy** requires every label of
        a wafer to be right at once.
        """
    )

    st.subheader("Limitations and interpretation")
    st.markdown(
        """
        - Part of the dataset was generated with a generative adversarial
          network to balance classes, so results describe this dataset and
          may not transfer to wafers from another fab.
        - Scores are model outputs. Tree and logistic scores may not be
          calibrated, and the linear SVM score is a transformed margin.
        - Predictions on dataset wafers can include samples seen in
          training; only the test-set metrics measure generalisation.
        - Hand-painted and uploaded maps are checked for shape and values
          but may still look unlike real probe data.
        - CNN-based recognition is planned future work.
        """
    )

    st.caption("Dataset reference: J. Wang et al., IEEE Transactions on Semiconductor Manufacturing, doi 10.1109/TSM.2020.3020985.")


# ============================================================
# MAIN
# ============================================================


try:
    dataset_path = hf_hub_download(
        repo_id="monishkadamaliya/wafersight-wafer-maps",
        filename="Wafer_Map_Datasets.npz",
        repo_type="dataset",
    )
    wafers_all, y_all = load_dataset(dataset_path)

except Exception as exc:
    st.error(f"Could not download or load the wafer dataset: {exc}")
    st.info(
        "Check that your Hugging Face dataset is public "
        "and the file exists in its main branch."
    )
    st.stop()


with st.sidebar:
    st.markdown("## WaferSight")
    st.caption("DEFECT INTELLIGENCE PLATFORM")
    st.divider()

    page = st.radio("Navigation", PAGES, label_visibility="collapsed")

    st.divider()
    st.markdown("### Dataset")
    st.write(f"**Wafer maps:** {len(wafers_all):,}")
    st.write("**Map size:** 52 x 52")
    st.write(f"**Labels:** {EXPECTED_LABELS}")
    st.write(f"**Models available:** {len(models)}")

if page == "Overview":
    page_overview(wafers_all, y_all, models)
elif page == "Wafer prediction":
    page_prediction(wafers_all, y_all, models)
elif page == "Model comparison":
    page_comparison()
elif page == "Dataset exploration":
    page_exploration(wafers_all, y_all)
else:
    page_methodology()

st.markdown(
    '<div class="footer">WAFERSIGHT · ACADEMIC PROJECT DEMONSTRATION</div>',
    unsafe_allow_html=True,
)
