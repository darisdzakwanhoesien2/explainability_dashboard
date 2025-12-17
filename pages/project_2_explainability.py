import streamlit as st
from explainability.diagrams import (
    build_track_combined_network,
    render_pyvis_project_2,
)
from explainability.code_snippets import ANALYZE_CODE, DEFORM_CODE

st.set_page_config(layout="wide")
st.title("Track Weakness & Deformation Explainability")

st.header("End-to-End Pipeline")
html = render_pyvis_project_2(build_track_combined_network())
st.components.v1.html(html, height=800, scrolling=True)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Weakness Analysis Code")
    st.code(ANALYZE_CODE, language="python")

with col2:
    st.subheader("Track Deformation Code")
    st.code(DEFORM_CODE, language="python")
