# explainability/project_3/diagrams.py
# ---------------------------------------------------------
# Explainability DAG for Project 3 (DR Classification Pipeline)
# Streamlit + PyVis
# ---------------------------------------------------------

import streamlit as st
from pyvis.network import Network
import tempfile
import os


# =========================================================
# =============== DAG GENERATION FUNCTION =================
# =========================================================
def build_explainability_dag() -> Network:
    """
    Build an explainability DAG for Project 3:
    - Data sources
    - Preprocessing
    - Models
    - Training strategies
    - Ensembles
    - Explainability (Grad-CAM)
    - Outputs / Metrics
    """
    net = Network(
        height="750px",
        width="100%",
        bgcolor="#0e1117",
        font_color="white",
        directed=True,
    )

    # -----------------------
    # Node style presets
    # -----------------------
    def add_node(node_id, label, color, shape="box"):
        net.add_node(
            node_id,
            label=label,
            color=color,
            shape=shape,
            font={"size": 14},
        )

    # -----------------------
    # Data nodes
    # -----------------------
    add_node("aptos", "APTOS 2019 Dataset\n(Fundus Images)", "#1f77b4")
    add_node("deepdrid", "DeepDRiD Dataset\n(Fundus Images)", "#1f77b4")

    # -----------------------
    # Preprocessing
    # -----------------------
    add_node("augmentation", "Data Augmentation\n• Flip\n• Rotate\n• Crop\n• ColorJitter", "#ff7f0e")
    add_node("normalization", "Normalization\n(ImageNet Mean / Std)", "#ff7f0e")

    # -----------------------
    # Models
    # -----------------------
    add_node("resnet18", "ResNet18", "#2ca02c")
    add_node("resnet34", "ResNet34", "#2ca02c")
    add_node("densenet121", "DenseNet121", "#2ca02c")
    add_node("resnet18_cbam", "ResNet18 + CBAM\n(Channel & Spatial Attention)", "#2ca02c")

    # -----------------------
    # Training strategies
    # -----------------------
    add_node("stage1", "Stage 1 Training\nAPTOS Pretraining", "#9467bd")
    add_node("stage2", "Stage 2 Fine-tuning\nDeepDRiD", "#9467bd")
    add_node("freeze", "Freeze Backbone\nTrain FC Only", "#9467bd")

    # -----------------------
    # Ensemble
    # -----------------------
    add_node("ensemble", "Ensemble Learning", "#8c564b")
    add_node("maxvote", "Max Voting", "#8c564b")
    add_node("weighted", "Weighted Average", "#8c564b")
    add_node("stacking", "Stacking\n(Logistic Regression)", "#8c564b")

    # -----------------------
    # Explainability
    # -----------------------
    add_node("gradcam", "Grad-CAM\nVisual Explanations", "#e377c2")

    # -----------------------
    # Metrics & Outputs
    # -----------------------
    add_node("metrics", "Evaluation Metrics\n• Accuracy\n• Cohen Kappa", "#17becf")
    add_node("submission", "Predictions & Submissions", "#17becf")

    # ======================================================
    # ====================== EDGES =========================
    # ======================================================

    # Data → Preprocessing
    net.add_edge("aptos", "augmentation")
    net.add_edge("deepdrid", "augmentation")
    net.add_edge("augmentation", "normalization")

    # Preprocessing → Models
    net.add_edge("normalization", "resnet18")
    net.add_edge("normalization", "resnet34")
    net.add_edge("normalization", "densenet121")
    net.add_edge("normalization", "resnet18_cbam")

    # Models → Training
    net.add_edge("resnet18", "stage1")
    net.add_edge("resnet34", "stage1")
    net.add_edge("densenet121", "stage1")
    net.add_edge("resnet18_cbam", "stage1")

    net.add_edge("stage1", "stage2")
    net.add_edge("stage2", "freeze")

    # Training → Metrics
    net.add_edge("stage2", "metrics")

    # Models → Ensemble
    net.add_edge("resnet18", "ensemble")
    net.add_edge("resnet34", "ensemble")
    net.add_edge("densenet121", "ensemble")

    net.add_edge("ensemble", "maxvote")
    net.add_edge("ensemble", "weighted")
    net.add_edge("ensemble", "stacking")

    # Ensemble → Metrics
    net.add_edge("maxvote", "metrics")
    net.add_edge("weighted", "metrics")
    net.add_edge("stacking", "metrics")

    # Explainability
    net.add_edge("resnet18", "gradcam")
    net.add_edge("resnet18_cbam", "gradcam")

    # Metrics → Output
    net.add_edge("metrics", "submission")

    return net


# =========================================================
# ================= STREAMLIT PAGE ========================
# =========================================================
def render_page():
    """
    Streamlit page for Project 3 Explainability DAG
    """
    st.set_page_config(
        page_title="Project 3 – Explainability DAG",
        layout="wide",
    )

    st.title("📊 Project 3 – Explainability & Training Pipeline")
    st.markdown(
        """
        This diagram explains the **end-to-end deep learning pipeline**
        used for **Diabetic Retinopathy classification**, including:

        - Multi-dataset training (APTOS & DeepDRiD)
        - Transfer learning & fine-tuning
        - Attention-based models (CBAM)
        - Ensemble learning strategies
        - Explainability with **Grad-CAM**
        """
    )

    net = build_explainability_dag()

    # Save PyVis HTML to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        html_path = tmp_file.name

    # Display in Streamlit
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    st.components.v1.html(html_content, height=800, scrolling=True)

    # Cleanup
    os.remove(html_path)


# =========================================================
# ====================== ENTRY =============================
# =========================================================
if __name__ == "__main__":
    render_page()
