import re

def split_text(text):
    """
    Split text into sentences and filter out sentences that are just numbers or numbering.

    Args:
        text (str): Input text to split.

    Returns:
        list[str]: List of cleaned sentences.
    """
    # Split text by punctuation marks typically ending sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    # Regex to detect sentences that are just numbering (like '1.', '2.2.', '3.1.4.')
    number_pattern = re.compile(r'^(\d+\.)+(\d+)?\.?$')

    filtered = [s for s in sentences if not number_pattern.match(s.strip()) and s.strip() != '']

    return filtered




'''
old code used to make does meaningful splits, instead of just sentences

# Lightweight spaCy pipeline with only rule-based sentence tokenizer
nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")

model = SentenceTransformer("all-MiniLM-L6-v2")

def spacy_sent_tokenize(text):
    doc = nlp(text.strip())
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def should_attach_to_previous(sent, prev_sent, similarity, threshold):
    sent_lower = sent.lower()
    prev_lower = prev_sent.lower()

    starts_with = ("is", "was", "are", "can", "should", "will", "that", "this", "it", "so")
    short = len(sent.split()) < 6
    is_question = prev_sent.strip().endswith("?")
    is_response = sent_lower.startswith("response") or "response:" in sent_lower

    return (
            similarity >= threshold
            or short
            or sent_lower.startswith(starts_with)
            or (is_question and is_response)
    )

def split_text(text, min_sents=1, max_sents=5, similarity_threshold=0.9):
    sentences = spacy_sent_tokenize(text)
    embeddings = model.encode(sentences)

    segments = []
    current_segment = [sentences[0]]
    current_embed_sum = embeddings[0]
    current_count = 1

    for i in range(1, len(sentences)):
        sent = sentences[i]
        sent_embed = embeddings[i]
        prev_sent = current_segment[-1]
        current_mean_embed = current_embed_sum / current_count

        similarity = cosine_similarity(
            current_mean_embed.reshape(1, -1), sent_embed.reshape(1, -1)
        )[0][0]

        if (
            current_count < max_sents
            and should_attach_to_previous(sent, prev_sent, similarity, similarity_threshold)
        ):
            current_segment.append(sent)
            current_embed_sum += sent_embed
            current_count += 1
        else:
            segments.append(" ".join(current_segment))
            current_segment = [sent]
            current_embed_sum = sent_embed
            current_count = 1

    if current_segment:
        segments.append(" ".join(current_segment))

    return segments
'''