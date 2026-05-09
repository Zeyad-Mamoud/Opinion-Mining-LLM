import streamlit as st
import pandas as pd
import torch
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.model_loader import load_model, load_tokenizer
from src.inference import generate_structured_output


st.set_page_config(
    page_title="Opinion Mining Live Demo",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Opinion Mining Live Demo")


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "lora_model"


@st.cache_resource
def load_base_model():

    tokenizer = load_tokenizer(BASE_MODEL)

    model = load_model(
        base_model_name=BASE_MODEL,
        adapter_path=None,
    )

    return model, tokenizer


@st.cache_resource
def load_adapter_model():

    tokenizer = load_tokenizer(BASE_MODEL)

    model = load_model(
        base_model_name=BASE_MODEL,
        adapter_path=ADAPTER_PATH,
    )

    return model, tokenizer


sample_reviews = [
    "The battery life is excellent but the screen is dim.",
    "Delivery was fast but the charger stopped working.",
    "WiFi connection is very fast and stable.",
]

review = st.text_area(
    "Customer Review",
    value=sample_reviews[0],
    height=150
)



if st.button("Run Live Comparison"):
    
    with st.spinner("Running base model..."):

        base_model, base_tokenizer = load_base_model()

        base_result = generate_structured_output(
            base_model,
            base_tokenizer,
            review,
            max_new_tokens=256,
        )

    with st.spinner("Running LoRA model..."):

        adapter_model, adapter_tokenizer = load_adapter_model()

        adapter_result = generate_structured_output(
            adapter_model,
            adapter_tokenizer,
            review,
            max_new_tokens=256,
        )


    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Base Model")

        st.json(base_result)

        aspects = pd.DataFrame(
            base_result.get("aspects", [])
        )

        if not aspects.empty:
            st.dataframe(
                aspects,
                use_container_width=True,
                hide_index=True,
            )


    with col2:

        st.subheader("Base + LoRA Adapter")

        st.json(adapter_result)

        aspects = pd.DataFrame(
            adapter_result.get("aspects", [])
        )

        if not aspects.empty:
            st.dataframe(
                aspects,
                use_container_width=True,
                hide_index=True,
            )