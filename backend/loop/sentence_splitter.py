def split_sentences(text: str) -> list[str]:
    """
    Splits text into sentences using NLTK punkt tokenizer.
    Falls back to period-splitting if NLTK fails.
    Filters out empty strings and very short fragments (< 3 words).
    """
    try:
        import nltk
        nltk.download("punkt_tab", quiet=True)
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
    except Exception:
        sentences = [s.strip() for s in text.split(".") if s.strip()]

    return [s for s in sentences if len(s.split()) >= 3]
