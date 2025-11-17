import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data(show_spinner=False)
def get_tfidf_matrix(quotes):
    """
    Compute the similarity distance matrix for a list of quotes.

    Args:
        quotes (list of str): List of text quotes.

    Returns:
        tuple: (tfidf_matrix, vectorizer)
            - tfidf_matrix (sparse matrix): similarity distance matrix.
            - vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer instance.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    matrix = vectorizer.fit_transform(quotes)
    return matrix, vectorizer


def find_similar_quotes(selected_quote, quotes, tfidf_matrix, k=5):
    """
    Find the top-k quotes most similar to the selected quote based on cosine similarity.

    Args:
        selected_quote (str): The quote to find similarities for.
        quotes (list of str): List of all quotes.
        tfidf_matrix (sparse matrix): similarity distance matrix for all quotes.
        k (int): Number of similar quotes to return (default is 5).

    Returns:
        list of tuples: Each tuple contains (index, quote_text, similarity_score)
                        for the top-k similar quotes, excluding the selected quote itself.
                        Returns empty list if selected_quote not found.
    """
    try:
        selected_index = quotes.index(selected_quote)
    except ValueError:
        # selected_quote not found in list, return empty
        return []

    # Compute cosine similarities between the selected quote and all quotes
    cosine_similarities = cosine_similarity(tfidf_matrix[selected_index], tfidf_matrix).flatten()

    # Get indices of top-k similar quotes excluding the selected quote itself
    similar_indices = cosine_similarities.argsort()[::-1][1:k+1]

    # Prepare result as list of (index, quote_text, similarity_score)
    return [(idx, quotes[idx], cosine_similarities[idx]) for idx in similar_indices]


def is_in_draft(index):
    """
    Check if a quote with a given index is already in the draft list stored in session state.

    Args:
        index (int): Index of the quote.

    Returns:
        bool: True if the quote is in draft, False otherwise.
    """
    return any(item.get("index") == index for item in st.session_state.draft)


def remove_from_draft(index):
    """
    Remove a quote from the draft list by its index.

    Args:
        index (int): Index of the quote to remove.
    """
    st.session_state.draft = [item for item in st.session_state.draft if item.get("index") != index]


def add_to_draft(quote_row, index):
    """
    Add a quote to the draft list in session state.

    Args:
        quote_row (pd.Series): The full row of the quote from the DataFrame.
        index (int): Index of the quote.
    """
    quote_dict = quote_row.to_dict()
    quote_dict["index"] = index
    st.session_state.draft.append(quote_dict)


def display_similar(event, k=5):
    """
    Display similar quotes to the one selected by the user.

    Args:
        event: Streamlit table selection event object containing selection info.
        k (int): Number of similar quotes to display (default is 5).

    Behaviour:
        - If no quote is selected, shows an info message.
        - Computes similarity distance matrix for all quotes.
        - Finds similar quotes to the selected one.
        - Displays each similar quote with organisation and similarity score.
        - Provides buttons to add/remove quotes from the draft list.
    """
    if not event or not event.selection or not event.selection.rows:
        st.info("Select a quote to see similar sentences.")
        return

    df = st.session_state.original_df
    quotes = df["quotes"].tolist()
    selected_index = event.selection.rows[-1]  # Get last selected row index
    selected_quote = quotes[selected_index]

    # Compute TF-IDF matrix (cached)
    tfidf_matrix, _ = get_tfidf_matrix(quotes)

    # Find similar quotes based on cosine similarity
    similar_quotes = find_similar_quotes(selected_quote, quotes, tfidf_matrix, k=k)

    if not similar_quotes:
        st.info("No similar quotes found.")
        return

    # Display each similar quote with buttons to add/remove from draft
    for idx, quote_text, score in similar_quotes:
        quote_row = df.iloc[idx]
        org = quote_row.get("organisation", "Unknown")
        in_draft = is_in_draft(idx)

        col1, col2 = st.columns([8, 2])
        with col1:
            st.markdown(f"\"{quote_text}\"  _(Source: {org}, Similarity: {score:.2f})_")

        with col2:
            key = f"sim_toggle_{idx}"
            if in_draft:
                st.button(
                    "Remove from Draft",
                    key=key,
                    on_click=lambda i=idx: remove_from_draft(i)
                )
            else:
                st.button(
                    "Add to Draft",
                    key=key,
                    on_click=lambda row=quote_row, i=idx: add_to_draft(row, i)
                )
