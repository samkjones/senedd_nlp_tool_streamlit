import pdfplumber
import re
import os
import pandas as pd
import streamlit as st

import app_helpers.segment as seg
import app_helpers.relevance as rel


@st.cache_resource
def load_model():
    """
    Retrieve and cache the SentenceTransformer model from session state.

    Returns:
        SentenceTransformer: Preloaded model from session state.
    """
    return st.session_state.model


@st.cache_data
def extract_pdf_info(filename):
    """
    Extract organisation ID and name from a structured filename.

    Args:
        filename (str): Filename in format '[org_id] [org_name].pdf'

    Returns:
        tuple[str, str]: (organisation ID, organisation name)

    Raises:
        ValueError: If file is not a PDF or format is invalid.
    """
    if not filename.lower().endswith(".pdf"):
        raise ValueError("File is not a PDF.")

    filename_no_ext = filename.rsplit(".", 1)[0]
    parts = filename_no_ext.split(" ", 1)

    if len(parts) != 2:
        raise ValueError("Filename does not match expected format: '[org_id] [org_name].pdf'")

    org_id = parts[0].strip()
    org_name = parts[1].strip()

    return org_id, org_name


def detect_language_from_page1(page):
    """
    Detect the language of the form based on keywords on the first page.
    This depends on what key words are used, or if the language is specified at all.
    No way to tell in the 2025-2026 consultation responses so "?"

    Args:
        page (pdfplumber.page.Page): The first page of the PDF.

    Returns:
        str: Detected language string ("English") or if it doesn't specify then ?
    """
    text = page.extract_text()
    return "English" if "English" in text else "?"


def clean_page_text(page, header_margin=70, footer_margin=70):
    """
    Remove headers, footers, and contact info from a PDF page.

    Args:
        page (pdfplumber.page.Page): Page to clean.
        header_margin (int): Vertical margin to exclude header.
        footer_margin (int): Vertical margin to exclude footer.

    Returns:
        str: Cleaned text from the page.
    """
    words = page.extract_words()
    height = page.height

    # Filter out words in header/footer margins
    valid_words = [
        w['text'] for w in words
        if header_margin < float(w['top']) < height - footer_margin
    ]

    text = "\n".join(valid_words)
    cleaned = remove_contact_and_address_lines(text)
    return cleaned


@st.cache_data
def extract_sentences_from_pdf(pdf_path):
    """
    Extract quotes from a given PDF, skipping metadata and contact info.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        pd.DataFrame: Structured DataFrame of extracted quotes.
    """
    filename = os.path.basename(pdf_path)
    org_id, org_name = extract_pdf_info(filename)

    all_text = []

    with pdfplumber.open(pdf_path) as pdf:
        if len(pdf.pages) == 0:
            return pd.DataFrame(columns=["quotes", "language", "organisation", "score"])

        # Detect language from first page
        language = detect_language_from_page1(pdf.pages[0])

        # Extract and clean text from each page (except the first)
        for page in pdf.pages[1:]:
            cleaned_text = clean_page_text(page)
            if cleaned_text.strip():
                all_text.append(cleaned_text)

    full_text = " ".join(all_text)

    # Split into quotes
    chunks = seg.split_text(full_text)
    df = pd.DataFrame(chunks, columns=["quotes"])
    df["language"] = language
    df["organisation"] = org_name

    # Compute relevance scores
    df["score"] = rel.compute_relevance_scores(df["quotes"].tolist(), full_text)
    df["score"] = df["score"].round(2)

    # Save extracted data to CSV
    save_data(df)
    return df


def save_data(df):
    """
    Save the extracted DataFrame to CSV, formatting columns.

    Args:
        df (pd.DataFrame): DataFrame to save.
    """
    # Add required metadata
    df['id'] = range(len(df))  # Use row index as ID
    df['question'] = None  # Placeholder for future use

    # Ensure consistent column order
    df = df[['id', 'organisation', 'question', 'score', 'quotes', 'language']]

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/pdf_data.csv", index=False)


def remove_contact_and_address_lines(text):
    """
    Remove lines likely to contain personal contact or address information.

    Args:
        text (str): Input text to clean.

    Returns:
        str: Cleaned text with sensitive lines removed.
    """
    lines = text.splitlines()
    cleaned_lines = []

    # Regex patterns for various contact details
    plus44_pattern = re.compile(r"\+44[()\d\s]*")
    email_pattern = re.compile(r"\b\w+@\w+\.\w+\b")
    postcode_pattern = re.compile(r"\b[A-Z]{1,2}\d{1,2}\s?\d[A-Z]{2}\b")
    phone_pattern = re.compile(r"\(?0\d{2,4}\)?[\s\-]?\d{2,4}")

    for line in lines:
        line = line.strip()
        line = " ".join(line.split())  # Normalize whitespace

        if (
            plus44_pattern.search(line) or
            email_pattern.search(line) or
            postcode_pattern.search(line) or
            phone_pattern.search(line) or
            re.search(r"\b(?:tel|telephone|email|fax|contact|t/ffon)\b", line, re.IGNORECASE)
        ):
            continue  # Skip line if it contains contact info

        cleaned_lines.append(line)

    return " ".join(cleaned_lines)
