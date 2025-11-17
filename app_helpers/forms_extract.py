import os
import pandas as pd
import app_helpers.segment as seg
import app_helpers.relevance as rel


def process_forms_csv(df):
    """
    Process a form-based CSV into a flat quote-level DataFrame with relevance scores.

    Args:
        df (pd.DataFrame): The uploaded CSV with 'organisation', 'language', and question columns.

    Returns:
        pd.DataFrame: Flattened and scored quotes.
    """
    rows = []
    id_counter = 0

    # Iterate over each row in the original df
    for _, row in df.iterrows():
        org = row["organisation"]
        lang = row["language"]

        for col in df.columns:
            if col.lower() in ["organisation", "language"]:
                continue

            # Full response to this question
            question_response = str(row[col])

            # Skip empty responses
            if not question_response.strip():
                continue

            # Split into quotes
            segments_list = seg.split_text(question_response)

            # Compute relevance scores relative to the full question response only
            scores = rel.compute_relevance_scores(segments_list, question_response)

            for seg_text, score in zip(segments_list, scores):
                id_counter += 1
                rows.append({
                    "id": id_counter,
                    "organisation": org,
                    "language": lang,
                    "question": col,
                    "quotes": seg_text,
                    "score": round(score, 2),
                })

    processed_df = pd.DataFrame(rows)

    # Ensure 'data' folder exists
    os.makedirs("data", exist_ok=True)
    processed_df.to_csv("data/forms_data.csv", index=False)

    return processed_df
