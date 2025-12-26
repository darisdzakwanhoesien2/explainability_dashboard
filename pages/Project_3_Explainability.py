# pages/Project_3_Explainability.py
# ============================================================
# Streamlit Page – Project 3 Explainability
# ============================================================

import streamlit as st
from explainability.project_3.diagrams import build_explainability_dag
from explainability.diagrams import render_pyvis_network


def main():
    st.set_page_config(page_title="Project 3 – Explainability", layout="wide")

    st.title("🩺 Project 3 – Diabetic Retinopathy Explainability")
    st.markdown(
        """
        This page explains the **deep learning pipeline**
        for Diabetic Retinopathy classification, including:

        - Multi-dataset training (APTOS + DeepDRiD)
        - Transfer learning & fine-tuning
        - Attention mechanisms (CBAM)
        - Ensemble learning
        - Grad-CAM visual explainability
        """
    )

    net = build_explainability_dag()
    render_pyvis_network(net)


if __name__ == "__main__":
    main()
