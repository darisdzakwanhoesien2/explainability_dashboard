import streamlit as st
from explainability.logic.text_cleaning import clean_bag_of_phrases
from explainability.logic.popularity import analyze_popularity

st.title("News Popularity App")

text = st.text_area("Bag of phrases")
if st.button("Clean"):
    st.write(clean_bag_of_phrases(text))
