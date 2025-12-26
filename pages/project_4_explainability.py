# ============================================================
# pages/project_4_explainability.py
# ============================================================

import streamlit as st
from explainability.diagrams import (
    build_step_1_network,
    build_step_2_network,
    build_step_3_network,
    build_step_4_network,
    build_step_5_network,
    build_step_6_network,
    build_step_7_network,
    build_step_8_network,
    build_parliament_combined_network,
    render_pyvis_network,
)

st.set_page_config(layout="wide")
st.title("🏛️ Parliamentary NLP Explainability (Project 4)")

st.markdown(
    """
    This page visualizes the **complete explainable NLP pipeline**
    for parliamentary speech analysis:
    - preprocessing
    - sentiment aggregation
    - similarity & clustering
    - topic evolution
    - NER & emotion prediction
    """
)

STEP_MAP = {
    "Step 1 – Text Preprocessing": build_step_1_network,
    "Step 2 – Exploratory Analysis": build_step_2_network,
    "Step 3 – Sentiment Normalization": build_step_3_network,
    "Step 4 – Word Frequency": build_step_4_network,
    "Step 5 – Similarity Computation": build_step_5_network,
    "Step 6 – Clustering": build_step_6_network,
    "Step 7 – Topic Modeling": build_step_7_network,
    "Step 8 – NER & Emotion Prediction": build_step_8_network,
    "Combined End-to-End Pipeline": build_parliament_combined_network,
}

selected = st.selectbox("Select Pipeline View", list(STEP_MAP.keys()))

net = STEP_MAP[selected]()
render_pyvis_network(net)
