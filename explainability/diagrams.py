# ============================================================
# explainability/diagrams.py
# ============================================================
# Global Explainability Registry + Streamlit Page
# ============================================================

import streamlit as st
import tempfile
import os
from typing import Callable, Dict

# ============================================================
# ===================== PROJECT 1 ============================
# ============================================================
from explainability.project_1.diagrams import (
    build_clean_bag_of_phrases_network,
    build_analyze_popularity_network,
    build_find_k_plexes_network,
    build_combined_network as build_news_combined_network,
    render_pyvis as render_pyvis_project_1,
)

# ============================================================
# ===================== PROJECT 2 ============================
# ============================================================
from explainability.project_2.diagrams import (
    build_weakness_network,
    build_deformation_network,
    build_combined_network as build_track_combined_network,
    render_pyvis as render_pyvis_project_2,
)

# ============================================================
# ===================== PROJECT 3 ============================
# ============================================================
from explainability.project_3.diagrams import (
    build_explainability_dag as build_dr_explainability_network,
)

# ============================================================
# ===================== PROJECT 4 ============================
# ============================================================
from explainability.project_4.diagrams import (
    build_step_1_network,
    build_step_2_network,
    build_step_3_network,
    build_step_4_network,
    build_step_5_network,
    build_step_6_network,
    build_step_7_network,
    build_step_8_network,
    build_combined_network as build_parliament_combined_network,
)

# ============================================================
# ===================== PUBLIC EXPORT ========================
# ============================================================
__all__ = [
    # Project 1
    "build_clean_bag_of_phrases_network",
    "build_analyze_popularity_network",
    "build_find_k_plexes_network",
    "build_news_combined_network",

    # Project 2
    "build_weakness_network",
    "build_deformation_network",
    "build_track_combined_network",

    # Project 3
    "build_dr_explainability_network",

    # Project 4
    "build_step_1_network",
    "build_step_2_network",
    "build_step_3_network",
    "build_step_4_network",
    "build_step_5_network",
    "build_step_6_network",
    "build_step_7_network",
    "build_step_8_network",
    "build_parliament_combined_network",

    # Page
    "render_explainability_page",
]

# ============================================================
# ===================== PYVIS RENDERER =======================
# ============================================================
def render_pyvis_network(net, height: int = 800):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        html_path = tmp.name

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    st.components.v1.html(html, height=height, scrolling=True)
    os.remove(html_path)


# ============================================================
# ===================== REGISTRY =============================
# ============================================================
NETWORK_REGISTRY: Dict[str, Dict[str, Callable]] = {
    "Project 1 – News Analytics": {
        "Clean Bag of Phrases": build_clean_bag_of_phrases_network,
        "Analyze Popularity": build_analyze_popularity_network,
        "Find K-Plexes": build_find_k_plexes_network,
        "Combined Pipeline": build_news_combined_network,
    },
    "Project 2 – Track Geometry": {
        "Weakness Detection": build_weakness_network,
        "Deformation Analysis": build_deformation_network,
        "Combined Pipeline": build_track_combined_network,
    },
    "Project 3 – Diabetic Retinopathy": {
        "Training & Explainability Pipeline": build_dr_explainability_network,
    },
    "Project 4 – Parliamentary NLP": {
        "Step 1 – Text Preprocessing": build_step_1_network,
        "Step 2 – Exploratory Analysis": build_step_2_network,
        "Step 3 – Sentiment Normalization": build_step_3_network,
        "Step 4 – Word Frequency": build_step_4_network,
        "Step 5 – Similarity Computation": build_step_5_network,
        "Step 6 – Clustering": build_step_6_network,
        "Step 7 – Topic Modeling": build_step_7_network,
        "Step 8 – NER & Emotion Prediction": build_step_8_network,
        "Combined Pipeline": build_parliament_combined_network,
    },
}


# ============================================================
# ===================== STREAMLIT PAGE =======================
# ============================================================
def render_explainability_page():
    st.set_page_config(page_title="Explainability Diagrams", layout="wide")

    st.title("🧠 Explainability Diagrams")
    st.markdown(
        """
        Interactive **explainability DAGs** for all projects.
        Each diagram represents full data, model, and analysis flow.
        """
    )

    st.sidebar.header("Navigation")

    project_name = st.sidebar.selectbox(
        "Select Project",
        list(NETWORK_REGISTRY.keys()),
    )

    diagram_name = st.sidebar.selectbox(
        "Select Diagram",
        list(NETWORK_REGISTRY[project_name].keys()),
    )

    st.subheader(project_name)
    st.caption(diagram_name)

    build_fn = NETWORK_REGISTRY[project_name][diagram_name]
    network = build_fn()

    render_pyvis_network(network)


if __name__ == "__main__":
    render_explainability_page()


# # explainability/diagrams.py
# # ============================================================
# # Global Explainability Registry + Streamlit Page
# # ============================================================
# # This module aggregates explainability DAGs from:
# #   - project_1
# #   - project_2
# #   - project_3
# #
# # It also exposes a unified Streamlit page for browsing
# # and rendering PyVis-based explainability diagrams.
# # ============================================================

# import streamlit as st
# import tempfile
# import os
# from typing import Callable, Dict

# # ============================================================
# # ===================== PROJECT 1 ============================
# # ============================================================
# from explainability.project_1.diagrams import (
#     build_clean_bag_of_phrases_network,
#     build_analyze_popularity_network,
#     build_find_k_plexes_network,
#     build_combined_network as build_news_combined_network,
#     render_pyvis as render_pyvis_project_1,
# )

# # ============================================================
# # ===================== PROJECT 2 ============================
# # ============================================================
# from explainability.project_2.diagrams import (
#     build_weakness_network,
#     build_deformation_network,
#     build_combined_network as build_track_combined_network,
#     render_pyvis as render_pyvis_project_2,
# )

# # ============================================================
# # ===================== PROJECT 3 ============================
# # ============================================================
# from explainability.project_3.diagrams import (
#     build_explainability_dag as build_dr_explainability_network,
# )

# # ============================================================
# # ===================== PUBLIC EXPORT ========================
# # ============================================================
# __all__ = [
#     # -------- Project 1 --------
#     "build_clean_bag_of_phrases_network",
#     "build_analyze_popularity_network",
#     "build_find_k_plexes_network",
#     "build_news_combined_network",
#     "render_pyvis_project_1",

#     # -------- Project 2 --------
#     "build_weakness_network",
#     "build_deformation_network",
#     "build_track_combined_network",
#     "render_pyvis_project_2",

#     # -------- Project 3 --------
#     "build_dr_explainability_network",

#     # -------- Page --------
#     "render_explainability_page",
# ]

# # ============================================================
# # ===================== PYVIS RENDERER =======================
# # ============================================================
# def render_pyvis_network(net, height: int = 800):
#     """
#     Generic PyVis renderer for Streamlit.
#     """
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
#         net.save_graph(tmp.name)
#         html_path = tmp.name

#     with open(html_path, "r", encoding="utf-8") as f:
#         html = f.read()

#     st.components.v1.html(html, height=height, scrolling=True)
#     os.remove(html_path)


# # ============================================================
# # ===================== REGISTRY =============================
# # ============================================================
# NETWORK_REGISTRY: Dict[str, Dict[str, Callable]] = {
#     "Project 1 – News Analytics": {
#         "Clean Bag of Phrases": build_clean_bag_of_phrases_network,
#         "Analyze Popularity": build_analyze_popularity_network,
#         "Find K-Plexes": build_find_k_plexes_network,
#         "Combined Pipeline": build_news_combined_network,
#     },
#     "Project 2 – Track Geometry": {
#         "Weakness Detection": build_weakness_network,
#         "Deformation Analysis": build_deformation_network,
#         "Combined Pipeline": build_track_combined_network,
#     },
#     "Project 3 – Diabetic Retinopathy": {
#         "Training & Explainability Pipeline": build_dr_explainability_network,
#     },
# }


# # ============================================================
# # ===================== STREAMLIT PAGE =======================
# # ============================================================
# def render_explainability_page():
#     """
#     Unified Explainability Page for all projects.
#     """
#     st.set_page_config(
#         page_title="Explainability Diagrams",
#         layout="wide",
#     )

#     st.title("🧠 Explainability Diagrams")
#     st.markdown(
#         """
#         This page provides **interactive explainability DAGs**
#         for all projects in the course.

#         Each diagram is built using **PyVis** and represents:
#         - Data flow
#         - Model logic
#         - Training strategy
#         - Evaluation and outputs
#         """
#     )

#     # ---------------- Sidebar ----------------
#     st.sidebar.header("Navigation")

#     project_name = st.sidebar.selectbox(
#         "Select Project",
#         list(NETWORK_REGISTRY.keys()),
#     )

#     diagram_name = st.sidebar.selectbox(
#         "Select Diagram",
#         list(NETWORK_REGISTRY[project_name].keys()),
#     )

#     # ---------------- Render ----------------
#     st.subheader(f"{project_name}")
#     st.caption(diagram_name)

#     build_fn = NETWORK_REGISTRY[project_name][diagram_name]
#     network = build_fn()

#     render_pyvis_network(network)


# # ============================================================
# # ===================== ENTRY POINT ==========================
# # ============================================================
# if __name__ == "__main__":
#     render_explainability_page()


# # explainability/diagrams.py

# from explainability.project_1.diagrams import (
#     build_clean_bag_of_phrases_network,
#     build_analyze_popularity_network,
#     build_find_k_plexes_network,
#     build_combined_network as build_news_combined_network,
#     render_pyvis as render_pyvis_project_1,
# )

# from explainability.project_2.diagrams import (
#     build_weakness_network,
#     build_deformation_network,
#     build_combined_network as build_track_combined_network,
#     render_pyvis as render_pyvis_project_2,
# )

# __all__ = [
#     # Project 1
#     "build_clean_bag_of_phrases_network",
#     "build_analyze_popularity_network",
#     "build_find_k_plexes_network",
#     "build_news_combined_network",
#     "render_pyvis_project_1",

#     # Project 2
#     "build_weakness_network",
#     "build_deformation_network",
#     "build_track_combined_network",
#     "render_pyvis_project_2",
# ]



# from pyvis.network import Network
# import streamlit.components.v1 as components

# COLORS = {
#     "input": "#AED6F1",
#     "process": "#F9E79F",
#     "decision": "#F5B7B1",
#     "output": "#ABEBC6",
# }

# # --------------------------------------------------
# # Streamlit-safe PyVis renderer
# # --------------------------------------------------
# def render_pyvis(net, height=600):
#     html = net.generate_html()
#     components.html(html, height=height, scrolling=True)

# # --------------------------------------------------
# # Individual DAG builders
# # --------------------------------------------------
# def build_clean_bag_of_phrases_network(x_offset=0):
#     net = Network(directed=True)

#     net.add_node("clean.input.text", label="text", color=COLORS["input"],
#                  shape="box", x=x_offset, y=0, fixed=True)
#     net.add_node("clean.check.type", label="is list / str?",
#                  color=COLORS["decision"], shape="diamond",
#                  x=x_offset + 200, y=0, fixed=True)
#     net.add_node("clean.join", label="join tokens",
#                  color=COLORS["process"], shape="box",
#                  x=x_offset + 400, y=-80, fixed=True)
#     net.add_node("clean.normalize", label="normalize string",
#                  color=COLORS["process"], shape="box",
#                  x=x_offset + 400, y=80, fixed=True)
#     net.add_node("clean.output", label="cleaned_text",
#                  color=COLORS["output"], shape="box",
#                  x=x_offset + 600, y=0, fixed=True)

#     net.add_edge("clean.input.text", "clean.check.type")
#     net.add_edge("clean.check.type", "clean.join")
#     net.add_edge("clean.check.type", "clean.normalize")
#     net.add_edge("clean.join", "clean.output")
#     net.add_edge("clean.normalize", "clean.output")

#     return net


# def build_analyze_popularity_network(x_offset=0):
#     net = Network(directed=True)

#     net.add_node("pop.input.df", label="df",
#                  color=COLORS["input"], shape="box",
#                  x=x_offset, y=0, fixed=True)
#     net.add_node("pop.input.metric", label="metric",
#                  color=COLORS["input"], shape="box",
#                  x=x_offset, y=120, fixed=True)
#     net.add_node("pop.extract.tokens", label="extract tokens/entities",
#                  color=COLORS["process"], shape="box",
#                  x=x_offset + 250, y=0, fixed=True)
#     net.add_node("pop.count.freq", label="count frequencies",
#                  color=COLORS["process"], shape="box",
#                  x=x_offset + 500, y=0, fixed=True)
#     net.add_node("pop.rank.top", label="top/bottom tweets",
#                  color=COLORS["process"], shape="box",
#                  x=x_offset + 500, y=120, fixed=True)
#     net.add_node("pop.output", label="tokens + tweet sets",
#                  color=COLORS["output"], shape="box",
#                  x=x_offset + 750, y=60, fixed=True)

#     net.add_edge("pop.input.df", "pop.extract.tokens")
#     net.add_edge("pop.extract.tokens", "pop.count.freq")
#     net.add_edge("pop.count.freq", "pop.output")
#     net.add_edge("pop.input.metric", "pop.rank.top")
#     net.add_edge("pop.rank.top", "pop.output")

#     return net


# def build_find_k_plexes_network(x_offset=0):
#     net = Network(directed=True)

#     net.add_node("kplex.input.graph", label="graph",
#                  color=COLORS["input"], shape="box",
#                  x=x_offset, y=0, fixed=True)
#     net.add_node("kplex.input.k", label="k",
#                  color=COLORS["input"], shape="box",
#                  x=x_offset, y=120, fixed=True)
#     net.add_node("kplex.search", label="Bron–Kerbosch search",
#                  color=COLORS["process"], shape="box",
#                  x=x_offset + 250, y=60, fixed=True)
#     net.add_node("kplex.check", label="is_k_plex",
#                  color=COLORS["decision"], shape="diamond",
#                  x=x_offset + 500, y=60, fixed=True)
#     net.add_node("kplex.output", label="k-plexes",
#                  color=COLORS["output"], shape="box",
#                  x=x_offset + 750, y=60, fixed=True)

#     net.add_edge("kplex.input.graph", "kplex.search")
#     net.add_edge("kplex.input.k", "kplex.search")
#     net.add_edge("kplex.search", "kplex.check")
#     net.add_edge("kplex.check", "kplex.output")

#     return net

# # --------------------------------------------------
# # ✅ TRUE COMBINED DAG (FIXED)
# # --------------------------------------------------
# def build_combined_network():
#     net = Network(directed=True)

#     sub_nets = [
#         build_clean_bag_of_phrases_network(0),
#         build_analyze_popularity_network(900),
#         build_find_k_plexes_network(1800),
#     ]

#     for sub in sub_nets:
#         for node in sub.nodes:
#             net.add_node(
#                 node["id"],
#                 **{k: v for k, v in node.items() if k != "id"}
#             )
#         for edge in sub.edges:
#             net.add_edge(edge["from"], edge["to"], label=edge.get("label"))

#     # Semantic links
#     net.add_edge("clean.output", "pop.input.df",
#                  label="cleaned text")
#     net.add_edge("pop.output", "kplex.input.graph",
#                  label="entity graph")

#     return net
