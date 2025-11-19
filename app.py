import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer

# Configure Streamlit layout to use full width
st.set_page_config(layout="wide")

# Load SentenceTransformer model only once from local download using Streamlit's caching mechanism
@st.cache_resource
def load_model():
    """Load and cache the SentenceTransformer model from local path."""
    #return SentenceTransformer("./local_model")
    return SentenceTransformer("all-MiniLM-L6-v2")

# Load model into session state to avoid reloading
if "model" not in st.session_state:
    st.session_state.model = load_model()

# Import internal app helper modules
import app_helpers.helpers as f
import app_helpers.similarity as s
import app_helpers.theming as t
import app_helpers.segment as seg
import app_helpers.pdf_extract as p

# Set app title
st.title("Quote Picker")

# Load the data (inputted by user)
df = f.load_data()


if df is not None:
    # Cache the original unfiltered dataset
    if "original_df" not in st.session_state:
        st.session_state.original_df = df.copy()

    # Filter data on first load
    if df not in st.session_state:
        st.session_state.df = f.filters(df).reset_index(drop=True)

    # Apply theme-based filters
    st.session_state.df = t.theme_filter().reset_index(drop=True)

    # Initialize set to track selected quote IDs
    if "selected_ids" not in st.session_state:
        st.session_state.selected_ids = set()

    # Identify selected rows by matching their IDs
    selected_rows = [
        i for i, row_id in enumerate(st.session_state.df["id"]) if row_id in st.session_state.selected_ids
    ]

    # Sort by similarity score (descending)
    st.session_state.df = st.session_state.df.sort_values(by="score", ascending=False).reset_index(drop=True)

    # Display interactive dataframe with single-row selection
    event = st.dataframe(
        st.session_state.df,
        hide_index=True,
        column_config=f.column_display(),
        column_order=f.get_column_order(),
        key="data",
        on_select="rerun",
        selection_mode="single-row"
    )

    # Display selected quote in main view
    f.display_quote(event, st.session_state.df)

    # Expanders for context and similar sentences
    with st.expander("See context."):
        f.get_context(event)

    with st.expander("See 5 similar sentences:"):
        s.display_similar(event)

    # Initialize list to draft selected quotes
    if "draft" not in st.session_state:
        st.session_state.draft = []

    # Enable sorting of selected quotes for export or review
    f.order_quotes()

else:
    # Prompt user to upload data
    st.info("Please upload PDF(s) or CSV file to continue.")
    st.stop()  # Prevent execution until valid data is uploaded

