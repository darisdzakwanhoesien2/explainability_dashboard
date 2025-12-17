import streamlit as st
from explainability.project_1.logic.text_cleaning import clean_bag_of_phrases

st.title("News Popularity App")

text = st.text_area("Enter bag_of_phrases")
if st.button("Clean"):
    st.write(clean_bag_of_phrases(text))
