"""Shopper Intent Lab - Streamlit front-end.

Scores an uploaded test CSV with the pre-trained pipelines stored in ``model/``.
Nothing is re-fitted here: every model is loaded from disk and only used for
``predict`` / ``predict_proba``.
"""
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from model.shopper_toolkit import (
    CLASS_NAMES,
    METRIC_ORDER,
    RAW_FEATURES,
    TARGET_COLUMN,
    purchase_probability,
    scorecard,
    to_binary_target,
)

HOME = Path(__file__).resolve().parent
MODEL_DIR = HOME / "model"
BUNDLED_TEST_CSV = HOME / "test_data.csv"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}
METRIC_HINTS = {
    "Accuracy": "Share of sessions classified correctly",
    "AUC": "Ranking quality of the purchase probability",
    "Precision": "Of predicted buyers, how many bought",
    "Recall": "Of real buyers, how many were caught",
    "F1": "Harmonic mean of precision & recall",
    "MCC": "Balanced correlation, robust to imbalance",
}
PALETTE = {"ink": "#14213D", "accent": "#E76F51", "teal": "#2A9D8F", "sand": "#F4F1EA"}

st.set_page_config(page_title="Shopper Intent Lab", page_icon="🛍️", layout="wide")

st.markdown(
    f"""
    <style>
      .hero {{background: linear-gradient(120deg, {PALETTE['ink']} 0%, {PALETTE['teal']} 100%);
              color: white; padding: 1.4rem 1.8rem; border-radius: 18px; margin-bottom: 1.2rem;}}
      .hero h1 {{margin: 0; font-size: 2.0rem; color: white;}}
      .hero p {{margin: .35rem 0 0; opacity: .9; font-size: 1.0rem;}}
      .tile {{background: white; border-left: 6px solid {PALETTE['accent']}; border-radius: 14px;
              padding: .8rem 1rem; box-shadow: 0 2px 10px rgba(20,33,61,.08); height: 100%;}}
      .tile .name {{font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; color: #6B7280;}}
      .tile .value {{font-size: 1.85rem; font-weight: 700; color: {PALETTE['ink']};}}
      .tile .note {{font-size: .74rem; color: #6B7280;}}
      .ref {{font-size: .74rem; color: {PALETTE['teal']}; font-weight: 600;}}
    </style>
    <div class="hero">
      <h1>🛍️ Shopper Intent Lab</h1>
      <p>Will an e-commerce browsing session end in a purchase? Upload held-out sessions,
      pick a pre-trained classifier and inspect how well it separates buyers from browsers.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading saved models ...")
def load_model_shelf() -> dict:
    return {name: joblib.load(MODEL_DIR / fname) for name, fname in MODEL_FILES.items()}


@st.cache_data
def load_reference_scores() -> pd.DataFrame:
    return pd.read_csv(MODEL_DIR / "leaderboard.csv").set_index("ML Model Name")


@st.cache_data
def load_training_report() -> dict:
    return json.loads((MODEL_DIR / "training_report.json").read_text())


@st.cache_data
def parse_csv(raw: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(raw))


def inspect_columns(frame: pd.DataFrame) -> tuple[list[str], bool]:
    missing = [c for c in RAW_FEATURES if c not in frame.columns]
    return missing, TARGET_COLUMN in frame.columns


def run_model(model, features: pd.DataFrame, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    buy_score = purchase_probability(model, features)
    return buy_score, (buy_score > cutoff).astype(int)


def metric_tiles(scores: dict, reference: pd.Series | None) -> None:
    for column, metric in zip(st.columns(len(METRIC_ORDER)), METRIC_ORDER):
        ref_line = ""
        if reference is not None:
            ref_line = f'<div class="ref">training hold-out: {reference[metric]:.4f}</div>'
        value = "n/a" if np.isnan(scores[metric]) else f"{scores[metric]:.4f}"
        column.markdown(
            f'<div class="tile"><div class="name">{metric}</div><div class="value">{value}</div>'
            f'<div class="note">{METRIC_HINTS[metric]}</div>{ref_line}</div>',
            unsafe_allow_html=True,
        )


def draw_confusion(y_true, y_pred, title: str):
    grid = confusion_matrix(y_true, y_pred, labels=[0, 1])
    share = grid / grid.sum(axis=1, keepdims=True).clip(min=1)
    captions = np.array(
        [[f"{grid[r, c]}\n({share[r, c]:.1%})" for c in range(2)] for r in range(2)]
    )
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    sns.heatmap(share, annot=captions, fmt="", cmap="YlGnBu", vmin=0, vmax=1, cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, linewidths=2, linecolor="white", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


def draw_roc(curves: dict[str, tuple[np.ndarray, np.ndarray]], focus: str):
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    for name, (y_true, score) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, score)
        emphasised = name == focus
        ax.plot(fpr, tpr, lw=2.6 if emphasised else 1.0, alpha=1 if emphasised else .45,
                color=PALETTE["accent"] if emphasised else None, label=name)
    ax.plot([0, 1], [0, 1], ls=":", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (selected model highlighted)", fontsize=11)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("① Test data")
    upload = st.file_uploader("Upload test CSV", type=["csv"],
                              help=f"Needs the {len(RAW_FEATURES)} feature columns and the '{TARGET_COLUMN}' column.")
    if BUNDLED_TEST_CSV.exists():
        st.download_button("⬇️ Download sample test_data.csv", BUNDLED_TEST_CSV.read_bytes(),
                           file_name="test_data.csv", mime="text/csv")
        fallback = st.checkbox("Use bundled test_data.csv when nothing is uploaded", value=True)
    else:
        fallback = False

    st.header("② Classifier")
    chosen = st.selectbox("Choose a trained model", list(MODEL_FILES))

    with st.expander("Advanced: decision cut-off"):
        cutoff = st.slider("Predict 'Purchase' when probability is above", 0.05, 0.95, 0.50, 0.05,
                           help="0.50 reproduces the metrics reported in the README.")
    st.caption("Models are loaded from `model/*.joblib`; the app never re-trains.")

if upload is not None:
    sessions, source_label = parse_csv(upload.getvalue()), f"uploaded file **{upload.name}**"
elif fallback:
    sessions, source_label = parse_csv(BUNDLED_TEST_CSV.read_bytes()), "bundled **test_data.csv**"
else:
    st.info("Upload a CSV from the sidebar (or tick the bundled-data option) to begin.")
    st.stop()

missing_cols, has_target = inspect_columns(sessions)
if missing_cols:
    st.error(f"The file is missing required feature columns: {', '.join(missing_cols)}")
    st.stop()

if has_target:
    sessions = sessions.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    try:
        truth = to_binary_target(sessions[TARGET_COLUMN]).to_numpy()
    except ValueError as err:
        st.error(str(err))
        st.stop()
else:
    truth = None
    st.warning(f"No '{TARGET_COLUMN}' column found - showing predictions only (metrics need the target).")

shelf = load_model_shelf()
reference = load_reference_scores()
features = sessions[RAW_FEATURES]
outputs = {name: run_model(model, features, cutoff) for name, model in shelf.items()}
buy_score, verdict = outputs[chosen]

summary = st.columns(4)
summary[0].metric("Sessions scored", f"{len(sessions):,}")
summary[1].metric("Source", "upload" if upload is not None else "bundled")
summary[2].metric("Actual buyers", f"{int(truth.sum()):,}" if truth is not None else "—")
summary[3].metric(f"Predicted buyers ({chosen})", f"{int(verdict.sum()):,}")
st.caption(f"Scoring {source_label} with the saved **{chosen}** pipeline at cut-off {cutoff:.2f}.")

tab_eval, tab_board, tab_rows, tab_info = st.tabs(
    ["📊 Evaluation", "🏁 Model comparison", "🔎 Predictions", "🧾 Training notes"]
)

with tab_eval:
    if truth is None:
        st.info("Add the target column to unlock metrics, confusion matrix and classification report.")
    else:
        scores = scorecard(truth, verdict, buy_score)
        st.subheader(f"Evaluation metrics — {chosen}")
        metric_tiles(scores, reference.loc[chosen] if chosen in reference.index else None)
        st.write("")
        left, middle, right = st.columns([1, 1, 1.1])
        with left:
            st.pyplot(draw_confusion(truth, verdict, f"Confusion matrix — {chosen}"))
        with middle:
            st.pyplot(draw_roc({n: (truth, o[0]) for n, o in outputs.items()}, chosen))
        with right:
            st.markdown("**Classification report**")
            report = classification_report(truth, verdict, labels=[0, 1], target_names=CLASS_NAMES,
                                           output_dict=True, zero_division=0)
            report_frame = pd.DataFrame(report).T.rename(index={"accuracy": "overall accuracy"})
            report_frame["support"] = report_frame["support"].astype(int)
            st.dataframe(report_frame.style.format({"precision": "{:.3f}", "recall": "{:.3f}",
                                                    "f1-score": "{:.3f}", "support": "{:,}"}),
                         width="stretch")
            tn, fp, fn, tp = confusion_matrix(truth, verdict, labels=[0, 1]).ravel()
            st.caption(f"Caught {tp} of {tp + fn} buyers; {fp} browsers were wrongly flagged as buyers.")

with tab_board:
    if truth is None:
        st.info("Model comparison needs the target column.")
    else:
        board = pd.DataFrame({name: scorecard(truth, pred, score) for name, (score, pred) in outputs.items()}).T
        board = board[METRIC_ORDER]
        st.subheader("All five saved models on this data")
        st.dataframe(board.style.format("{:.4f}").highlight_max(axis=0, color="#CDEFE9"),
                     width="stretch")
        best = board["MCC"].idxmax()
        st.success(f"Best MCC on this data: **{best}** ({board.loc[best, 'MCC']:.4f}).")

        long_form = board.reset_index(names="Model").melt(id_vars="Model", var_name="Metric", value_name="Score")
        fig, ax = plt.subplots(figsize=(10, 3.6))
        sns.barplot(data=long_form, x="Metric", y="Score", hue="Model", palette="Set2", ax=ax)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.15))
        ax.set_xlabel("")
        fig.tight_layout()
        st.pyplot(fig)

with tab_rows:
    preview = sessions.copy()
    preview.insert(0, "Purchase probability", np.round(buy_score, 4))
    preview.insert(0, "Predicted", np.where(verdict == 1, CLASS_NAMES[1], CLASS_NAMES[0]))
    if truth is not None:
        preview.insert(1, "Correct?", np.where(verdict == truth, "✅", "❌"))
    only_buyers = st.toggle("Show only sessions predicted as Purchase", value=False)
    shown = preview[preview["Predicted"] == CLASS_NAMES[1]] if only_buyers else preview
    st.dataframe(shown, width="stretch", height=420)
    st.download_button("⬇️ Download predictions", shown.to_csv(index=False).encode(),
                       file_name=f"predictions_{MODEL_FILES[chosen].replace('.joblib', '')}.csv", mime="text/csv")

with tab_info:
    notes = load_training_report()
    st.markdown(
        f"""
        **Dataset:** {notes['dataset']} — {notes['rows_after_dedup']:,} unique sessions,
        {len(notes['raw_features'])} input features, purchase rate {notes['positive_rate']:.1%}.
        Split: {notes['train_rows']:,} train / {notes['holdout_rows']:,} hold-out (stratified, seed {notes['split_seed']}).
        Hyper-parameters chosen by {notes['cv_folds']}-fold stratified grid search maximising **{notes['selection_metric']}**.
        """
    )
    tuned = pd.DataFrame(
        [{"Model": m["display_name"], "CV MCC": round(m["cv_mcc"], 4), "Best parameters": json.dumps(m["best_params"])}
         for m in notes["models"].values()]
    )
    st.dataframe(tuned, width="stretch", hide_index=True)
    st.markdown("**Reference hold-out metrics recorded at training time**")
    st.dataframe(reference.style.format("{:.4f}"), width="stretch")
