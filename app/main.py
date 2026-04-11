"""Streamlit demo for the opinion mining project."""

from __future__ import annotations

import json

import streamlit as st


SAMPLE_OUTPUT = {
    "review": "The camera quality is excellent, but the battery drains quickly.",
    "overall_sentiment": "mixed",
    "aspects": [
        {
            "feature": "camera quality",
            "sentiment": "positive",
            "opinion": "excellent",
        },
        {
            "feature": "battery",
            "sentiment": "negative",
            "opinion": "drains quickly",
        },
    ],
}


st.set_page_config(page_title="Opinion Mining Demo", layout="wide")

st.title("Opinion Mining from Customer Reviews")
st.caption("Extract sentiment and product features from reviews using a base LLM and its LoRA-tuned version.")

st.subheader("Review Input")
review_text = st.text_area(
    "Enter a customer review",
    value="The delivery was fast, the packaging was neat, but the charger stopped working after one week.",
    height=140,
)


st.subheader("Base Model")
st.text_input("Selected base model", value="mistralai/Mistral-7B-Instruct-v0.3")
st.text_input("LoRA adapter path", value="results/lora_adapter")


st.subheader("Expected Structured Output")
st.code(json.dumps(SAMPLE_OUTPUT, indent=2), language="json")

st.info(
    "Next step: connect this UI to `src/model_loader.py` and `src/inference.py` "
    "so the app can run real predictions from the selected base model and its "
    "LoRA-tuned adapter."
)

if st.button("Preview Review Payload"):
    preview = SAMPLE_OUTPUT.copy()
    preview["review"] = review_text
    st.json(preview)
