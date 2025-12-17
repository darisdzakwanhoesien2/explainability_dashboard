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
    build_combined_network,
    render_pyvis,
)

st.set_page_config(layout="wide")
st.title("News Popularity Explainability Dashboard")

st.header("End-to-End Explainability (Combined DAG)")
render_pyvis(build_combined_network(), height=700)

st.divider()

sections = [
    ("Text Cleaning", build_clean_bag_of_phrases_network, CLEAN_BAG_OF_PHRASES_CODE),
    ("Popularity Analysis", build_analyze_popularity_network, ANALYZE_POPULARITY_CODE),
    ("k-Plex Detection", build_find_k_plexes_network, FIND_K_PLEXES_CODE),
]

for title, builder, code in sections:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"{title} Diagram")
        render_pyvis(builder())

    with col2:
        st.subheader("Source Code")
        st.code(code, language="python")

    st.divider()


# import streamlit as st
# from explainability.code_snippets import (
#     CLEAN_BAG_OF_PHRASES_CODE,
#     ANALYZE_POPULARITY_CODE,
#     FIND_K_PLEXES_CODE,
# )
# from explainability.diagrams import (
#     build_clean_bag_of_phrases_network,
#     build_analyze_popularity_network,
#     build_find_k_plexes_network,
#     build_combined_network,
#     render_pyvis,
# )

# st.set_page_config(layout="wide")
# st.title("News Popularity Explainability Pipeline")

# # ======================================================
# # 🔷 End-to-End Explainability DAG
# # ======================================================
# st.header("End-to-End Pipeline (True Combined DAG)")
# render_pyvis(build_combined_network(), height=700)

# st.divider()

# # ======================================================
# # 🔹 Individual Explainability Views
# # ======================================================
# sections = [
#     ("Text Cleaning", build_clean_bag_of_phrases_network, CLEAN_BAG_OF_PHRASES_CODE),
#     ("Popularity Analysis", build_analyze_popularity_network, ANALYZE_POPULARITY_CODE),
#     ("k-Plex Detection", build_find_k_plexes_network, FIND_K_PLEXES_CODE),
# ]

# for title, builder, code in sections:
#     col1, col2 = st.columns(2)

#     with col1:
#         st.subheader(f"{title} Diagram")
#         render_pyvis(builder())

#     with col2:
#         st.subheader("Source Code")
#         st.code(code, language="python")

#     st.divider()
