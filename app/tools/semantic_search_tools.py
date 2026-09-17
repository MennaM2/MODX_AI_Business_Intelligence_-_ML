"""
Semantic search over a dataset's text columns, backed by a real
vector database (FAISS).

Why this exists: most of the agent's other tools answer *structured*
questions (stats, correlations, ML metrics). This tool answers
*unstructured* questions like "find rows about late deliveries" or
"which reviews mention a refund" by embedding each row's text and
retrieving the nearest neighbours to the query embedding - the same
retrieval step used in a RAG pipeline.

The index is built once per file and cached on disk under
`.vector_cache/`, keyed by an md5 hash of the file path + text
column, so repeated queries against the same dataset don't re-embed
every row from scratch.
"""

import hashlib
import os
import pickle

import numpy as np
import pandas as pd

# faiss and sentence-transformers are heavy, optional dependencies -
# imported lazily inside semantic_search() rather than at module load,
# so the rest of the application (API, other tools, UI) still starts
# and works even in an environment where they aren't installed. If
# they're missing, semantic_search() returns a clear error instead of
# crashing the whole process at import time.


CACHE_DIR = ".vector_cache"

# Small, fast, good-enough embedding model (~80MB) - fine for a CPU
# box and for demo-sized CSVs. Loaded lazily so importing this module
# doesn't pay the model-load cost unless the tool is actually used.
_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. Install it "
                "with 'pip install sentence-transformers' to enable "
                "semantic_search."
            ) from exc
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def _cache_key(file_path: str, text_column: str) -> str:
    raw = f"{os.path.abspath(file_path)}::{text_column}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _build_or_load_index(file_path: str, text_column: str):
    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            "faiss-cpu is not installed. Install it with "
            "'pip install faiss-cpu' to enable semantic_search."
        ) from exc

    os.makedirs(CACHE_DIR, exist_ok=True)
    key = _cache_key(file_path, text_column)
    index_path = os.path.join(CACHE_DIR, f"{key}.faiss")
    meta_path = os.path.join(CACHE_DIR, f"{key}.pkl")

    if os.path.exists(index_path) and os.path.exists(meta_path):
        index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            texts = pickle.load(f)
        return index, texts

    df = pd.read_csv(file_path)

    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found in dataset."
        )

    texts = df[text_column].fillna("").astype(str).tolist()

    model = _get_model()
    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        normalize_embeddings=True
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    # Inner product on normalized vectors == cosine similarity.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(texts, f)

    return index, texts


def semantic_search(
    file_path: str,
    text_column: str,
    query: str,
    top_k: int = 5
) -> dict:
    """
    Find the rows in `text_column` whose meaning is closest to
    `query`, using vector similarity search instead of exact keyword
    matching (so it can match paraphrases and related wording, not
    just literal substrings).
    """
    try:
        index, texts = _build_or_load_index(file_path, text_column)
    except Exception as exc:
        return {"error": str(exc)}

    if index.ntotal == 0:
        return {"error": "No text found to search."}

    try:
        model = _get_model()
        query_vector = model.encode(
            [query],
            normalize_embeddings=True
        )
    except Exception as exc:
        return {"error": str(exc)}
    query_vector = np.asarray(query_vector, dtype="float32")

    top_k = min(top_k, index.ntotal)
    scores, indices = index.search(query_vector, top_k)

    results = []
    for score, row_index in zip(scores[0], indices[0]):
        if row_index == -1:
            continue
        results.append({
            "row_index": int(row_index),
            "text": texts[row_index],
            "similarity": round(float(score), 4)
        })

    return {
        "query": query,
        "matches": results
    }
