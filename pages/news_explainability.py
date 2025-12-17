# pages/news_explainability.py

import streamlit as st
from explainability.code_snippets import (
    CLEAN_BAG_OF_PHRASES_CODE,
    ANALYZE_POPULARITY_CODE,
    FIND_K_PLEXES_CODE,
)
from explainability.diagrams import (
    build_clean_bag_of_phrases_network,
    build_analyze_popularity_network,
    build_find_k_plexes_network,
    render_pyvis,
)

st.set_page_config(layout="wide")
st.title("News Popularity Explainability Pipeline")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Pipeline Diagram")
    net = build_clean_bag_of_phrases_network()
    render_pyvis(net)

with col2:
    st.subheader("Source Code")
    st.code(CLEAN_BAG_OF_PHRASES_CODE, language="python")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Popularity Analysis Diagram")
    net = build_analyze_popularity_network()
    render_pyvis(net)

with col2:
    st.subheader("Source Code")
    st.code(ANALYZE_POPULARITY_CODE, language="python")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("k-Plex Detection Diagram")
    net = build_find_k_plexes_network()
    render_pyvis(net)

with col2:
    st.subheader("Source Code")
    st.code(FIND_K_PLEXES_CODE, language="python")
