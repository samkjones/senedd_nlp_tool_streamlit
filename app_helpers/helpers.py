import streamlit as st
import pandas as pd
import numpy as np
from streamlit_sortables import sort_items
from streamlit.runtime.scriptrunner import RerunException, RerunData

from app_helpers.pdf_extract import extract_sentences_from_pdf
from app_helpers.forms_extract import process_forms_csv

def rerun():
    """Force a rerun of the Streamlit script."""
    raise RerunException(RerunData())

@st.cache_data
def get_data(filename):
    """
    Load CSV data from the given filename and cache it.

    Args:
        filename (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(filename)


def load_data():
    """
    Upload PDF(s) and/or CSV(s), extract sentences from each,
    and return a combined DataFrame.

    Currently does not allow for more data to be uploaded after initial upload.

    Returns:
        pd.DataFrame | None: Combined DataFrame of extracted quotes,
                             or None if no valid files uploaded.
    """
    st.header("Upload your data (PDFs and/or CSVs)")

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs or CSV files",
        type=["pdf", "csv"],
        accept_multiple_files=True,
        key="combined_uploader"
    )
    st.write('The CSVs should correspond to Microsoft Forms outputs and include ONLY the column headers "organisation", "language" '
             ' and a column header for each consultation question.')

    dataframes = []
    total_pdf_sentences = 0
    total_csv_sentences = 0

    if uploaded_files:
        for uploaded_file in uploaded_files:
            filename = uploaded_file.name.lower()

            # --- Handle PDFs ---
            if filename.endswith(".pdf"):
                with open(uploaded_file.name, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                try:
                    df = extract_sentences_from_pdf(uploaded_file.name)
                    df["source_file"] = uploaded_file.name
                    df["source_type"] = "pdf"
                    dataframes.append(df)
                    total_pdf_sentences += len(df)
                except Exception as e:
                    st.error(f"Failed to process PDF '{uploaded_file.name}': {e}")

            # --- Handle CSVs ---
            elif filename.endswith(".csv"):
                try:
                    df = pd.read_csv(uploaded_file)
                    required_columns = {"organisation", "language"}
                    csv_columns = set(col.lower() for col in df.columns)

                    if not required_columns.issubset(csv_columns):
                        missing = required_columns - csv_columns
                        st.error(f"CSV '{uploaded_file.name}' is missing required column(s): {', '.join(missing)}")
                        continue
                    elif len(csv_columns) <= len(required_columns):
                        st.error(f"CSV '{uploaded_file.name}' must have at least one question column.")
                        continue

                    processed_df = process_forms_csv(df)
                    processed_df["source_file"] = uploaded_file.name
                    processed_df["source_type"] = "csv"
                    dataframes.append(processed_df)
                    total_csv_sentences += len(processed_df)
                except Exception as e:
                    st.error(f"Failed to process CSV '{uploaded_file.name}': {e}")

        # Single success messages
        if total_pdf_sentences > 0:
            st.success(f"Extracted {total_pdf_sentences} sentences from {len([f for f in uploaded_files if f.name.lower().endswith('.pdf')])} PDF(s).")
        if total_csv_sentences > 0:
            st.success(f"Processed {total_csv_sentences} sentences from {len([f for f in uploaded_files if f.name.lower().endswith('.csv')])} CSV file(s).")

    if dataframes:
        return pd.concat(dataframes, ignore_index=True)

    return None


def column_display():
    """
    Returns a dictionary defining custom column headers and visibility.

    Returns:
        dict: Column configuration for Streamlit table display.
    """
    return {
        "quotes": "Quotes",
        "score": st.column_config.NumberColumn(
            "Relevance Score",
            help="This score is calculated as...",
        ),
        "organisation": "Source",
        "language": "Language",
        "question": None,
        "id": None
    }


def get_column_order():
    """Returns the preferred column order for display."""
    return ["quotes", "score", "organisation", "language"]

def is_in_draft(index):
    """
    Check if the given quote index is in the draft list.

    Args:
        index (int): Index of the quote.

    Returns:
        bool: True if present in draft.
    """
    return any(item.get("index") == index for item in st.session_state.draft)

def remove_from_draft(index):
    """Remove a quote from the draft based on its index."""
    st.session_state.draft = [item for item in st.session_state.draft if item.get("index") != index]

def add_to_draft(quote_row, index):
    """Add a quote to the draft with its original index."""
    quote_dict = quote_row.to_dict()
    quote_dict["index"] = index
    st.session_state.draft.append(quote_dict)


def display_quote(event, df):
    """
    Display selected quote and buttons to add/remove it from draft.

    Args:
        event: Streamlit event from dataframe selection (corresponding to row selected)
        df (pd.DataFrame): The dataframe being displayed.
    """
    selected = event.selection.rows if event and event.selection else []
    if not selected:
        st.info("No quote selected")
        return

    selected_id = selected[-1]
    quote = df.iloc[selected_id]

    st.subheader(f"\"{quote['quotes']}\"")

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"*- {quote['organisation']}*")
    with col2:
        st.write(f"Original Language: {quote.get('language', 'Unknown')}")

    key = f"quote_toggle_{selected_id}"
    in_draft = is_in_draft(selected_id)

    if in_draft:
        st.button("Remove from Draft", key=key, on_click=lambda idx=selected_id: remove_from_draft(idx))
    else:
        st.button("Add to Draft", key=key, on_click=lambda row=quote, idx=selected_id: add_to_draft(row, idx))



def filters(df):
    """
    Display sidebar UI for filtering quotes and return the filtered DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame to filter.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    default_filters = {
        "organisation": "All",
        "search": "",
        "predefined": "All",
        "question": "All",
        "topic": ""
    }

    if "filters" not in st.session_state:
        st.session_state.filters = default_filters.copy()

    with st.sidebar:
        if st.button("Clear filters 🧹"):
            st.session_state.filters = default_filters.copy()
            for key, val in default_filters.items():
                st.session_state[f"filter_{key}"] = val
            rerun()

        # Organisation dropdown
        org_options = ["All"] + sorted(df["organisation"].dropna().unique().tolist())
        st.session_state.filters["organisation"] = st.selectbox(
            "Filter by Source",
            options=org_options,
            index=org_options.index(st.session_state.filters["organisation"]),
            key="filter_organisation"
        )

        # Free text search
        st.session_state.filters["search"] = st.text_input(
            "Search Quotes", value=st.session_state.filters["search"], key="filter_search"
        )

        # Predefined filters
        predefined_options = ["All", "Available in Welsh", "Quotes with stats"]
        st.session_state.filters["predefined"] = st.selectbox(
            "Predefined Filters",
            options=predefined_options,
            index=predefined_options.index(st.session_state.filters["predefined"]),
            key="filter_predefined"
        )

        # Question filter
        question_options = ["All"] + sorted(df["question"].dropna().unique().tolist())
        st.session_state.filters["question"] = st.selectbox(
            "Filter by Question",
            options=question_options,
            index=question_options.index(st.session_state.filters["question"]),
            key="filter_question"
        )

    # Filtering logic
    filtered_df = df.copy()

    org = st.session_state.filters["organisation"]
    if org != "All":
        filtered_df = filtered_df[filtered_df["organisation"] == org]

    search_text = st.session_state.filters["search"]
    if search_text:
        filtered_df = filtered_df[filtered_df["quotes"].str.contains(search_text, case=False, na=False)]

    predefined = st.session_state.filters["predefined"]
    if predefined == "Available in Welsh":
        filtered_df = filtered_df[filtered_df["language"].isin(["Welsh", "Both"])]
    elif predefined == "Quotes with stats":
        stat_pattern = (
            r"(?i)(?:\d+[\.,]?\d*\s*%|\bpercent(?:age)?\b|\bproportion\b|\brate\b|"
            r"\bratio\b|\bstatistic(?:s)?\b|[\+\-\*/=])"
        )
        filtered_df = filtered_df[filtered_df["quotes"].str.contains(stat_pattern, na=False, regex=True)]

    question = st.session_state.filters["question"]
    if question != "All":
        filtered_df = filtered_df[filtered_df["question"] == question]

    return filtered_df.reset_index(drop=True)



def get_draggable_style():
    return """
    .sortable-component {
    background-color:rgb(255, 255, 255);
    font-size: 16px;
    counter-reset: item;
    }
    .sortable-item {
    background-color: white;
    color: black;
    }
    """


def get_draggable_style():
    """Return custom CSS for drag-and-drop styling."""
    return """
    .sortable-component {
        background-color: rgb(255, 255, 255);
        font-size: 16px;
        counter-reset: item;
    }
    .sortable-item {
        background-color: white;
        color: black;
    }
    """

def display_draft(sorted_display_list, display_map):
    """
    Render the current draft in the chosen quote order.

    Args:
        sorted_display_list (list): Ordered list of quote texts.
        display_map (dict): Map from quote text to original quote data.
    """
    st.subheader("Draft document preview:")

    if not sorted_display_list:
        st.info("No quotes in draft.")
        return

    draft_hash = hash(tuple(sorted_display_list))
    if st.session_state.get("last_draft_hash") == draft_hash:
        return

    st.session_state["last_draft_hash"] = draft_hash

    draft_lines = []
    for quote_text in sorted_display_list:
        item = display_map[quote_text]
        source = item.get("organisation", "Someone")
        language = item.get("language", "English")

        if language in ("Welsh", "Both"):
            line = f'{source} [Available in Welsh] said "{quote_text}"'
        else:
            line = f'{source} said "{quote_text}"'
        draft_lines.append(line)

    for line in draft_lines:
        st.write(line)

    st.download_button(
        label="Download draft as .txt file",
        data="\n\n".join(draft_lines),
        file_name="draft_document.txt",
        mime="text/plain"
    )

def order_quotes():
    """
    Show a drag-and-drop UI to reorder selected quotes and update the draft preview.
    """
    display_map = {item["quotes"]: item for item in st.session_state.draft}
    display_list = list(display_map.keys())

    sorted_display_list = sort_items(
        display_list,
        direction="vertical",
        custom_style=get_draggable_style()
    )

    if (
        "last_sorted_list" not in st.session_state
        or st.session_state.last_sorted_list != sorted_display_list
    ):
        st.session_state.last_sorted_list = sorted_display_list
        display_draft(sorted_display_list, display_map)


def get_context(event):
    """
    Show the sentence before and after the selected quote for context.

    Args:
        event: Selection event from Streamlit dataframe.
    """
    df = st.session_state.df
    original_df = st.session_state.original_df
    selected = event.selection.rows if event and event.selection else []

    if not selected:
        st.info("No quote selected.")
        return

    selected_row = df.iloc[selected[-1]]
    selected_quote = selected_row["quotes"]
    matches = original_df[original_df["quotes"] == selected_quote]
    if matches.empty:
        st.warning("Selected quote not found in original data.")
        return

    index = matches.index[0]
    max_index = len(original_df) - 1

    prev_quote = original_df.loc[index - 1, "quotes"] if index > 0 else ""
    curr_quote = original_df.loc[index, "quotes"]
    next_quote = original_df.loc[index + 1, "quotes"] if index < max_index else ""

    combined_quote = f"{prev_quote} {curr_quote} {next_quote}".strip()
    organisation = original_df.loc[index, "organisation"]

    quote_dict = {"organisation": organisation, "quotes": combined_quote}
    in_draft = quote_dict in st.session_state.draft

    col1, col2 = st.columns([8, 2])
    with col1:
        st.markdown(combined_quote)
    with col2:
        if in_draft:
            st.button("Remove from Draft", key="context_remove", on_click=lambda: st.session_state.draft.remove(quote_dict))
        else:
            st.button("Add to Draft", key="context_add", on_click=lambda: st.session_state.draft.append(quote_dict))


