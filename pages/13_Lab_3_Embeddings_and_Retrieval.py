import streamlit as st

from labs.lab3_embeddings_retrieval.embeddings_playground import render_embeddings_lab
from utils.ui import display_footer

# Page Config
st.set_page_config(
    page_title="Lab 3: Embeddings & Retrieval",
    page_icon="🔮",
    layout="wide",
)

# Render Lab
render_embeddings_lab()

# Footer
display_footer()
