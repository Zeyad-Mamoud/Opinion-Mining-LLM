"""Streamlit GUI for the opinion mining project."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_METRICS_PATH = ROOT_DIR / "results" / "baseline_metrics.json"
BASE_EVALUATION_PATH = ROOT_DIR / "results" / "baseline_evaluation.csv"
LABELED_DATA_DIR = ROOT_DIR / "data" / "Labeled-data"
LLM_RESPONSES_DIR = ROOT_DIR / "data" / "LLM responses for evaluation"
FINETUNED_RESPONSES_PATH = (
    LLM_RESPONSES_DIR / "finetuned_LLM_responses.json"
)
BASE_RESPONSES_PATH = LLM_RESPONSES_DIR / "qwen2.5-1.5B-instruct_responses.json"

SAMPLE_REVIEWS = [
    "Boot time is super fast, around anywhere from 35 seconds to 1 minute.",
    "Did not enjoy the new Windows 8 and touchscreen functions.",
    "I am pleased with the fast log on, speedy WiFi connection and the long battery life 6 hrs.",
    "Other than not being a fan of click pads industry standard these days and the lousy internal speakers, it's hard for me to find things about this notebook I don't like.",
    "The delivery was fast, the packaging was neat, but the charger stopped working after one week.",
]

FALLBACK_PREDICTION = {
    "domain": "general",
    "aspects": [
        {"term": "delivery", "polarity": "positive"},
        {"term": "packaging", "polarity": "positive"},
        {"term": "charger", "polarity": "negative"},
    ],
}

FINETUNED_SUMMARY = {
    "json_validity_rate": 1.0,
    "average_precision": 0.8667,
    "average_recall": 0.8917,
    "f1_score": 0.8790,
    "polarity_accuracy": 0.9333,
    "matched_aspects": 30,
    "total_ground_truth_aspects": 34,
    "total_predicted_aspects": 35,
}


st.set_page_config(
    page_title="Opinion Mining LLM",
    page_icon="OM",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --primary: #6366f1;
            --primary-light: #818cf8;
            --primary-dark: #4f46e5;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --neutral-50: #f9fafb;
            --neutral-100: #f3f4f6;
            --neutral-200: #e5e7eb;
            --neutral-600: #4b5563;
            --neutral-800: #1f2937;
        }
        
        * {
            transition: all 0.3s ease;
        }
        
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1400px;
            background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
        }
        
        .app-title {
            font-size: 2.8rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: -0.02em;
        }
        
        .app-subtitle {
            color: #6b7280;
            font-size: 1.1rem;
            margin-bottom: 2rem;
            font-weight: 500;
            letter-spacing: 0.01em;
        }
        
        .status-card {
            border: 2px solid #e5e7eb;
            border-radius: 12px;
            padding: 1.5rem;
            background: white;
            height: 100%;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.08);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .status-card:hover {
            border-color: #6366f1;
            box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.15);
            transform: translateY(-2px);
        }
        
        .status-label {
            color: #6b7280;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        
        .status-value {
            color: #1f2937;
            font-size: 1.75rem;
            font-weight: 800;
            margin-top: 0.25rem;
            letter-spacing: -0.01em;
        }
        
        .small-note {
            color: #9ca3af;
            font-size: 0.8rem;
            margin-top: 0.5rem;
            font-weight: 500;
        }
        
        div[data-testid="stMetricValue"] {
            font-size: 1.75rem;
            font-weight: 800;
            color: #6366f1;
        }
        
        div[data-testid="stMetricLabel"] {
            font-weight: 600;
            color: #4b5563;
        }
        
        .input-section {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            border: 2px solid #e5e7eb;
            margin-bottom: 1.5rem;
        }
        
        .input-section:hover {
            border-color: #c7d2fe;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.1);
        }
        
        [data-testid="stTabs"] {
            border-bottom: 3px solid #e5e7eb;
        }
        
        button {
            font-weight: 600;
            border-radius: 8px;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-size: 0.85rem;
        }
        
        button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
            color: white;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
        }
        
        h2, h3, h4 {
            color: #1f2937;
            font-weight: 700;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }
        
        h2 {
            font-size: 1.75rem;
            border-bottom: 3px solid #6366f1;
            padding-bottom: 0.75rem;
        }
        
        h3 {
            font-size: 1.35rem;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
            border-radius: 12px;
            padding: 1.25rem;
            border: 2px solid #e5e7eb;
        }
        
        [data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
            border: 2px solid #e5e7eb;
        }
        
        .chart-container {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            border: 2px solid #e5e7eb;
            margin-bottom: 1.5rem;
        }
        
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_records(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        data = json.loads(text)
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict):
            rows = [data]
        else:
            rows = []
    except json.JSONDecodeError:
        rows = []
        with path.open(encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    rows.append(json.loads(line))

    dict_rows = [row for row in rows if isinstance(row, dict)]
    if limit is not None:
        return dict_rows[:limit]
    return dict_rows


@st.cache_data
def load_labeled_splits(data_dir: Path, limit_per_split: int = 500) -> dict[str, list[dict[str, Any]]]:
    return {
        "Train": load_records(data_dir / "train_labeled_v2.json", limit_per_split),
        "Dev": load_records(data_dir / "dev_labeled_v2.json", limit_per_split),
        "Test": load_records(data_dir / "test_labeled_v2.json", limit_per_split),
    }


def parse_prediction(prediction: Any) -> dict[str, Any]:
    if isinstance(prediction, dict):
        return prediction
    if not isinstance(prediction, str):
        return {"domain": "general", "aspects": []}

    text = prediction.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        matches = re.findall(r"\{.*\}", text, flags=re.DOTALL)
        if matches:
            try:
                return json.loads(matches[-1])
            except json.JSONDecodeError:
                pass

    return {"domain": "general", "aspects": []}


def normalize_prediction(prediction: dict[str, Any], review_text: str) -> dict[str, Any]:
    aspects = prediction.get("aspects", [])
    if not isinstance(aspects, list):
        aspects = []

    normalized_aspects = []
    for aspect in aspects:
        if not isinstance(aspect, dict):
            continue
        term = aspect.get("term") or aspect.get("feature") or ""
        polarity = aspect.get("polarity") or aspect.get("sentiment") or "neutral"
        opinion = aspect.get("opinion") or term
        normalized_aspects.append(
            {
                "term": str(term).strip(),
                "polarity": str(polarity).strip().lower(),
                "opinion": str(opinion).strip(),
            }
        )

    return {
        "review": review_text,
        "domain": prediction.get("domain", "general"),
        "aspects": normalized_aspects,
    }


def lookup_saved_prediction(review_text: str, responses: list[dict[str, Any]]) -> dict[str, Any] | None:
    target = review_text.strip().lower()
    for row in responses:
        if row.get("input", "").strip().lower() == target:
            return parse_prediction(row.get("prediction"))
    return None


def infer_demo_prediction(review_text: str, responses: list[dict[str, Any]]) -> dict[str, Any]:
    saved = lookup_saved_prediction(review_text, responses)
    if saved is not None:
        return saved

    lower_review = review_text.lower()
    aspects: list[dict[str, str]] = []
    keyword_map = {
        "battery": "battery life",
        "screen": "screen",
        "charger": "charger",
        "delivery": "delivery",
        "packaging": "packaging",
        "wifi": "WiFi connection",
        "keyboard": "keyboard",
        "speaker": "internal speakers",
        "support": "tech support",
        "windows": "Windows",
    }
    positive_words = {"good", "great", "fast", "excellent", "easy", "happy", "pleased", "powerful", "works"}
    negative_words = {"bad", "dim", "slow", "problem", "difficult", "not", "lousy", "stopped", "drains", "negative"}

    for keyword, term in keyword_map.items():
        if keyword in lower_review:
            window = lower_review[max(0, lower_review.find(keyword) - 35) : lower_review.find(keyword) + 60]
            polarity = "neutral"
            if any(word in window for word in positive_words):
                polarity = "positive"
            if any(word in window for word in negative_words):
                polarity = "negative"
            aspects.append({"term": term, "polarity": polarity})

    if not aspects:
        return FALLBACK_PREDICTION

    domain = "software" if any(word in lower_review for word in ["windows", "software", "os"]) else "electronics"
    return {"domain": domain, "aspects": aspects}


@st.cache_resource(show_spinner=False)
def load_live_resources(base_model_name: str, adapter_path: str, load_in_4bit: bool):
    from src.model_loader import load_model, load_tokenizer

    tokenizer = load_tokenizer(base_model_name)
    model = load_model(
        base_model_name=base_model_name,
        adapter_path=adapter_path.strip() or None,
        load_in_4bit=load_in_4bit,
    )
    return model, tokenizer


def render_status_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">{label}</div>
            <div class="status-value">{value}</div>
            <div class="small-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_metrics_frame(base_metrics: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Metric": "JSON validity",
                "Base model": float(base_metrics.get("json_validity_rate", 0)),
                "LoRA fine-tuned": FINETUNED_SUMMARY["json_validity_rate"],
            },
            {
                "Metric": "Precision",
                "Base model": float(base_metrics.get("average_precision", 0)),
                "LoRA fine-tuned": FINETUNED_SUMMARY["average_precision"],
            },
            {
                "Metric": "Recall",
                "Base model": float(base_metrics.get("average_recall", 0)),
                "LoRA fine-tuned": FINETUNED_SUMMARY["average_recall"],
            },
            {
                "Metric": "F1 score",
                "Base model": float(base_metrics.get("f1_score", 0)),
                "LoRA fine-tuned": FINETUNED_SUMMARY["f1_score"],
            },
        ]
    )


def render_prediction(prediction: dict[str, Any]) -> None:
    aspects_df = pd.DataFrame(prediction.get("aspects", []))
    if aspects_df.empty:
        st.warning("No aspects were extracted for this review.")
    else:
        st.dataframe(aspects_df, width="stretch", hide_index=True)

    st.code(json.dumps(prediction, indent=2, ensure_ascii=False), language="json")


inject_styles()

base_metrics = load_json(BASE_METRICS_PATH, {})
baseline_df = load_csv(BASE_EVALUATION_PATH)
finetuned_responses = load_records(FINETUNED_RESPONSES_PATH)
base_responses = load_records(BASE_RESPONSES_PATH)
labeled_splits = load_labeled_splits(LABELED_DATA_DIR)

with st.sidebar:
    st.markdown("### Configuration")
    st.divider()
    
    st.markdown("#### Model Settings")
    prediction_mode = st.radio(
        "Prediction source",
        ["Saved/demo predictions", "Live model inference"],
        help="Use saved predictions for fast presentation demos, or load the model for real inference.",
    )
    base_model_name = st.text_input("Base model", value=DEFAULT_BASE_MODEL, placeholder="Qwen/Qwen2.5-1.5B-Instruct")
    adapter_path = st.text_input("LoRA adapter path", value="", placeholder="Optional path to adapter")
    max_new_tokens = st.slider("Max new tokens", min_value=64, max_value=512, value=256, step=32)
    load_in_4bit = st.checkbox("Load in 4-bit", value=True, help="Use 4-bit quantization for faster inference")

    st.divider()
    st.markdown("#### Dataset Statistics")
    labeled_total = sum(len(rows) for rows in labeled_splits.values())
    
    col1, col2 = st.columns(2, gap="small")
    with col1:
        st.metric("Labeled", f"{labeled_total:,}")
    with col2:
        st.metric("Base", f"{len(base_responses):,}")
    
    st.metric("Fine-tuned", f"{len(finetuned_responses):,}")
    
    st.divider()
    st.markdown("#### Split Details")
    split_cols = st.columns(3, gap="small")
    split_cols[0].metric("Train", f"{len(labeled_splits['Train']):,}")
    split_cols[1].metric("Dev", f"{len(labeled_splits['Dev']):,}")
    split_cols[2].metric("Test", f"{len(labeled_splits['Test']):,}")

st.markdown('<div class="app-title">Opinion Mining from Customer Reviews</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Extract aspect-based sentiment using Qwen2.5-1.5B with advanced LoRA fine-tuning</div>',
    unsafe_allow_html=True,
)
st.divider()

summary_cols = st.columns(4, gap="medium")
with summary_cols[0]:
    render_status_card("Base Model", "Qwen 2.5", "1.5B Instruct")
with summary_cols[1]:
    render_status_card("Training", "LoRA", "Parameter-efficient")
with summary_cols[2]:
    render_status_card("Test Split", "1,572", "SemEval ABSA")
with summary_cols[3]:
    render_status_card("Fine-tuned F1", "0.88", "20-sample aligned")

extract_tab, compare_tab, dataset_tab = st.tabs(["Extract Opinion", "Model Comparison", "Datasets"])

with extract_tab:
    left_col, right_col = st.columns([1.05, 1], gap="large")

    with left_col:
        st.markdown("### Review Input")
        selected_sample = st.selectbox("Try a sample review", SAMPLE_REVIEWS, label_visibility="collapsed")
        default_review = st.session_state.get("review_text", selected_sample)
        review_text = st.text_area("Customer review", value=default_review, height=170, placeholder="Paste a customer review here...")

        st.divider()
        action_cols = st.columns(3, gap="small")
        run_clicked = action_cols[0].button("Extract", type="primary", use_container_width=True)
        sample_clicked = action_cols[1].button("Sample", use_container_width=True)
        clear_clicked = action_cols[2].button("Clear", use_container_width=True)

        if sample_clicked:
            st.session_state.review_text = selected_sample
            st.rerun()
        if clear_clicked:
            st.session_state.review_text = ""
            st.rerun()

        st.info("Tip: Use saved/demo mode for fast presentation, or enable live inference for real predictions.")

    if "prediction" not in st.session_state:
        initial = infer_demo_prediction(selected_sample, finetuned_responses)
        st.session_state.prediction = normalize_prediction(initial, selected_sample)

    if run_clicked:
        if not review_text.strip():
            st.warning("Please enter a review before extracting opinions.")
        elif prediction_mode == "Live model inference":
            try:
                from src.inference import generate_structured_output

                with st.spinner("Loading model and generating prediction..."):
                    model, tokenizer = load_live_resources(base_model_name, adapter_path, load_in_4bit)
                    live_prediction = generate_structured_output(
                        model,
                        tokenizer,
                        review_text,
                        max_new_tokens=max_new_tokens,
                    )
                st.session_state.prediction = normalize_prediction(live_prediction, review_text)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Live inference failed: {exc}")
                st.info("Falling back to saved/demo extraction for the GUI preview.")
                demo_prediction = infer_demo_prediction(review_text, finetuned_responses)
                st.session_state.prediction = normalize_prediction(demo_prediction, review_text)
        else:
            demo_prediction = infer_demo_prediction(review_text, finetuned_responses)
            st.session_state.prediction = normalize_prediction(demo_prediction, review_text)

    with right_col:
        st.markdown("### Structured Output")
        current_prediction = st.session_state.prediction
        domain = current_prediction.get("domain", "general")
        aspects = current_prediction.get("aspects", [])

        metric_cols = st.columns(3, gap="small")
        with metric_cols[0]:
            st.metric("Domain", str(domain).title())
        with metric_cols[1]:
            st.metric("Aspects", len(aspects))
        
        polarities = [aspect.get("polarity", "neutral") for aspect in aspects]
        top_polarity = max(set(polarities), key=polarities.count) if polarities else "neutral"
        with metric_cols[2]:
            st.metric("Main Polarity", top_polarity.title())

        st.divider()
        render_prediction(current_prediction)

with compare_tab:
    st.markdown("### Base Model vs LoRA Fine-Tuned Model")
    metrics_df = build_metrics_frame(base_metrics)
    chart_df = metrics_df.melt(id_vars="Metric", var_name="Model", value_name="Score")
    fig = px.bar(
        chart_df,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        color_discrete_sequence=["#9ca3af", "#6366f1"],
        range_y=[0, 1.05],
    )
    fig.update_traces(texttemplate="%{y:.2f}", textposition="outside", marker_line_width=0)
    fig.update_layout(
        height=450, 
        legend_title_text="",
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="white",
        plot_bgcolor="rgba(249, 250, 251, 0.5)",
        font=dict(family="Arial, sans-serif", size=12, color="#4b5563"),
        hovermode="x unified",
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="rgba(0,0,0,0.1)", annotation_text="50%")
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### Key Metrics")
    metric_cols = st.columns(4, gap="medium")
    metric_cols[0].metric("Base JSON Validity", f"{base_metrics.get('json_validity_rate', 0):.0%}")
    metric_cols[1].metric("Fine-tuned JSON Validity", "100%")
    metric_cols[2].metric("Fine-tuned F1", f"{FINETUNED_SUMMARY['f1_score']:.2f}")
    metric_cols[3].metric("Polarity Accuracy", f"{FINETUNED_SUMMARY['polarity_accuracy']:.0%}")

    st.markdown("### Detailed Metrics Comparison")
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.markdown("### Baseline Evaluation Samples")
    if baseline_df.empty:
        st.info("No baseline evaluation CSV was found.")
    else:
        display_cols = ["sample_idx", "review", "json_valid", "ground_truth_aspects", "predicted_aspects"]
        st.dataframe(baseline_df[display_cols], use_container_width=True, hide_index=True)

with dataset_tab:
    st.markdown("### Project Datasets")
    st.caption("Browse `data/Labeled-data` and `data/LLM responses for evaluation` directories.")

    dataset_view = st.radio(
        "Dataset view",
        ["Labeled data", "LLM responses"],
        horizontal=True,
    )

    if dataset_view == "Labeled data":
        split_name = st.selectbox("Split", list(labeled_splits.keys()))
        rows = labeled_splits.get(split_name, [])
        query = st.text_input("Filter labeled reviews", value="", placeholder="Search in reviews...")
        if query.strip():
            rows = [row for row in rows if query.lower() in row.get("input", "").lower()]

        metric_cols = st.columns(3, gap="medium")
        metric_cols[0].metric("Train Loaded", f"{len(labeled_splits['Train']):,}")
        metric_cols[1].metric("Dev Loaded", f"{len(labeled_splits['Dev']):,}")
        metric_cols[2].metric("Test Loaded", f"{len(labeled_splits['Test']):,}")

        st.divider()
        if not rows:
            st.info("No labeled examples match the current filter.")
        else:
            display_rows = []
            for row in rows:
                parsed = parse_prediction(row.get("prediction", "{}"))
                display_rows.append(
                    {
                        "input": row.get("input", ""),
                        "domain": parsed.get("domain", "general"),
                        "aspects_count": len(parsed.get("aspects", []))
                        if isinstance(parsed.get("aspects", []), list)
                        else 0,
                        "prediction": row.get("prediction", ""),
                    }
                )

            st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

            selected_index = st.number_input(
                "Inspect labeled sample number",
                min_value=0,
                max_value=max(len(rows) - 1, 0),
                value=0,
                step=1,
            )
            selected_row = rows[int(selected_index)]
            parsed_prediction = parse_prediction(selected_row.get("prediction", "{}"))
            
            col1, col2 = st.columns(2, gap="large")
            with col1:
                st.write("Selected Review")
                st.code(selected_row.get("input", ""), language="text")
            with col2:
                st.write("Parsed Label")
                st.code(json.dumps(parsed_prediction, indent=2, ensure_ascii=False), language="json")

    else:
        response_source = st.selectbox(
            "Response file",
            ["Base model responses", "Fine-tuned model responses", "Compare side by side"],
        )
        query = st.text_input("Filter response reviews", value="", placeholder="Search in reviews...")

        def filter_response_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            if not query.strip():
                return rows
            return [row for row in rows if query.lower() in row.get("input", "").lower()]

        if response_source == "Base model responses":
            rows = filter_response_rows(base_responses)
            st.metric("Base Response Rows", f"{len(rows):,}")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        elif response_source == "Fine-tuned model responses":
            rows = filter_response_rows(finetuned_responses)
            st.metric("Fine-tuned Response Rows", f"{len(rows):,}")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            base_by_input = {row.get("input", ""): row.get("prediction", "") for row in base_responses}
            compared_rows = []
            for row in finetuned_responses:
                review = row.get("input", "")
                if query.strip() and query.lower() not in review.lower():
                    continue
                compared_rows.append(
                    {
                        "input": review,
                        "base_prediction": base_by_input.get(review, ""),
                        "fine_tuned_prediction": row.get("prediction", ""),
                    }
                )

            st.metric("Compared Rows", f"{len(compared_rows):,}")
            st.dataframe(pd.DataFrame(compared_rows), use_container_width=True, hide_index=True)

            if compared_rows:
                selected_index = st.number_input(
                    "Inspect comparison sample number",
                    min_value=0,
                    max_value=max(len(compared_rows) - 1, 0),
                    value=0,
                    step=1,
                )
                selected_row = compared_rows[int(selected_index)]
                st.write("Review")
                st.code(selected_row["input"], language="text")
                
                compare_cols = st.columns(2, gap="large")
                with compare_cols[0]:
                    st.write("Base Model")
                    st.code(
                        json.dumps(
                            parse_prediction(selected_row["base_prediction"]),
                            indent=2,
                            ensure_ascii=False,
                        ),
                        language="json",
                    )
                with compare_cols[1]:
                    st.write("Fine-tuned Model")
                    st.code(
                        json.dumps(
                            parse_prediction(selected_row["fine_tuned_prediction"]),
                            indent=2,
                            ensure_ascii=False,
                        ),
                        language="json",
                    )
