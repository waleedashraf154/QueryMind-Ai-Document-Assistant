from .common import _embed_model, BATCH_SIZE

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of text strings using a local sentence-transformers model
    (all-MiniLM-L6-v2). Runs entirely on-device — no API calls, no quota,
    no network dependency.

    Parameters
    ----------
    texts : list[str]
        The strings to embed. Must be non-empty.

    Returns
    -------
    list[list[float]]
        One embedding vector per input string (384-dim each).
    """
    if not texts:
        raise ValueError("embed_texts: texts list is empty.")
    if _embed_model is None:
        raise RuntimeError(
            "sentence-transformers is not installed. "
            "Run: pip install sentence-transformers"
        )

    vectors = _embed_model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return [vec.tolist() for vec in vectors]

