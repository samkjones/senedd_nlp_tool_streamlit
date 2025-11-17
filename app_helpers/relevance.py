from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import numpy as np
import re
from sentence_transformers import SentenceTransformer


def generate_summary(text, model, num_sentences=3):
    """
    Generate a summary by selecting the most central sentences in the document.

    Args:
        text (str): The full text to summarise.
        model (SentenceTransformer): Your embedding model.
        num_sentences (int): Number of summary sentences to return.

    Returns:
        str: A pseudo-summary of the text.
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    # Get embeddings
    embeddings = model.encode(sentences)
    avg_embedding = np.mean(embeddings, axis=0)

    # Compute cosine similarity to average embedding
    similarity_scores = cosine_similarity(embeddings, [avg_embedding]).flatten()

    # Select top N most representative sentences
    top_indices = np.argsort(similarity_scores)[-num_sentences:]
    top_indices = sorted(top_indices)  # maintain original order

    summary_sentences = [sentences[i] for i in top_indices]
    return " ".join(summary_sentences)


def compute_relevance_scores(quotes, full_text):
    """
    Compute cosine similarity between each quote and a summary of the full text.

    Args:
        quotes (list[str]): List of individual quotes.
        full_text (str): Full document text.

    Returns:
        list[float]: Relevance scores for each quote.
    """
    model = st.session_state.model
    summary = generate_summary(full_text, model)

    summary_embedding = model.encode([summary])[0]
    quote_embeddings = model.encode(quotes)

    scores = cosine_similarity(quote_embeddings, [summary_embedding]).flatten()
    return scores.tolist()
