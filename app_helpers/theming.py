import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Use the model loaded in Streamlit session state
model = st.session_state.model


@st.cache_data(show_spinner=False)
def compute_embeddings(quotes):
    """
    Compute sentence embeddings for a list of quotes using the cached model.

    Args:
        quotes (list of str): List of text quotes to embed.

    Returns:
        np.ndarray: Array of embedding vectors corresponding to the input quotes.
    """
    return model.encode(quotes, show_progress_bar=False)


def classify_quotes_by_theme(theme_text, quotes, quote_embeddings, threshold=0.25):
    """
    Classify quotes by computing cosine similarity against a theme text embedding.
    Returns matched quotes whose similarity score exceeds the threshold.

    Args:
        theme_text (str): The theme or topic text to classify quotes against.
        quotes (list of str): List of quotes to classify.
        quote_embeddings (np.ndarray): Precomputed embeddings for the quotes.
        threshold (float): Similarity threshold for including quotes as matching.

    Returns:
        list of tuples: Each tuple contains (index, quote_text, similarity_score),
                        sorted descending by similarity_score.
    """
    # Encode the theme text to get its embedding
    theme_embedding = model.encode([theme_text])[0]

    # Compute cosine similarity between the theme embedding and each quote embedding
    similarities = cosine_similarity([theme_embedding], quote_embeddings)[0]

    # Collect quotes that meet or exceed the similarity threshold
    results = [
        (i, quote, score)
        for i, (quote, score) in enumerate(zip(quotes, similarities))
        if score >= threshold
    ]

    # Sort results by similarity score in descending order
    results.sort(key=lambda x: x[2], reverse=True)
    return results


def theme_filter():
    """
    Streamlit sidebar widget for filtering quotes by a user-inputted theme/topic.
    Uses NLP similarity to classify quotes relative to the theme text.

    Returns:
        pd.DataFrame: Filtered dataframe of quotes matching the theme,
                      or the original dataframe if no filter is applied.
    """
    with st.sidebar:
        # Text input to specify the theme/topic filter
        st.session_state.filters["topic"] = st.text_input(
            "Filter by topic",
            value=st.session_state.filters.get("topic", ""),
            help=(
                "This feature uses NLP to classify quotes into the inputted topic. "
                "It is an experimental feature."
            ),
            key="filter_topics",
        )
        # Optional: Add confidence threshold slider if needed
        # threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.25, step=0.05, key="theme_threshold")

    # Get the current dataframe from session state
    df = st.session_state.df
    quotes = df["quotes"].tolist()

    # Retrieve user input for topic filter
    topic_text = st.session_state.filters.get("topic", "")
    threshold = 0.25  # You can make this dynamic if adding slider above

    if topic_text:
        # Compute embeddings for all quotes (cached)
        quote_embeddings = compute_embeddings(quotes)

        # Get matched quotes based on similarity threshold
        matched = classify_quotes_by_theme(topic_text, quotes, quote_embeddings, threshold)

        if matched:
            # Extract indices of matched quotes and filter dataframe accordingly
            matched_indices = [i for i, _, _ in matched]
            filtered_df = df.iloc[matched_indices].copy()
            return filtered_df
        else:
            # Show warning if no matches found
            st.warning("No quotes matched that theme with the current threshold.")
            # Return empty dataframe to indicate no results
            return df.iloc[0:0]

    # Return full dataframe if no filter applied
    return df
