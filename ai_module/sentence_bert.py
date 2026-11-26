"""
Sentence-BERT Embedding Module.
Uses pretrained sentence-transformers/all-MiniLM-L6-v2 to generate fresh, dense,
384-dimensional semantic embeddings for resumes and job descriptions.
Implements disk caching (.npy) and batch processing to optimize inference.
"""

import os
from typing import List, Optional, Union
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_DIR = "outputs/embeddings"

_MODEL_INSTANCE: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    """
    Singleton loader for SentenceTransformer to avoid repeated disk reads.
    """
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        _MODEL_INSTANCE = SentenceTransformer(MODEL_NAME)
    return _MODEL_INSTANCE


def encode_texts(
    texts: List[str],
    batch_size: int = 64,
    show_progress: bool = False
) -> np.ndarray:
    """
    Generates fresh 384-dimensional embeddings for a list of texts using batching.
    Returns normalized float32 numpy array.
    """
    model = get_model()
    # Normalize embeddings for direct dot-product cosine similarity
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    return embeddings.astype(np.float32)


def get_or_create_job_embeddings(
    jobs_df: pd.DataFrame,
    cache_path: str = os.path.join(EMBEDDINGS_DIR, "job_embeddings_minilm.npy"),
    force_recompute: bool = False
) -> np.ndarray:
    """
    Retrieves cached job embeddings or generates fresh embeddings from combined job text.
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path) and not force_recompute:
        return np.load(cache_path)
    
    texts = jobs_df["combined_job_text"].fillna("").tolist()
    embeddings = encode_texts(texts, batch_size=64, show_progress=False)
    np.save(cache_path, embeddings)
    return embeddings


def get_or_create_cv_embeddings(
    cv_df: pd.DataFrame,
    cache_path: str = os.path.join(EMBEDDINGS_DIR, "cv_embeddings_minilm.npy"),
    force_recompute: bool = False,
    max_records: Optional[int] = None,
    batch_size: int = 64
) -> np.ndarray:
    """
    Retrieves cached CV embeddings or generates fresh embeddings from resume text.
    Preserves exact row-for-row alignment with candidate_id in cv_df.
    Reuses existing valid cached embeddings when expanding dataset to avoid unnecessary recomputation.
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    target_df = cv_df.iloc[:max_records] if max_records is not None else cv_df
    target_count = len(target_df)
    
    if os.path.exists(cache_path) and not force_recompute:
        cached = np.load(cache_path)
        if max_records is not None and len(cached) >= max_records:
            return cached[:max_records]
        elif len(cached) == target_count:
            return cached
        elif 0 < len(cached) < target_count:
            # Reusing already verified cached embeddings to avoid unnecessary recomputation
            existing_count = len(cached)
            remaining_texts = target_df["clean_resume_text"].iloc[existing_count:].fillna("").tolist()
            new_embeddings = encode_texts(remaining_texts, batch_size=batch_size, show_progress=False)
            full_embeddings = np.vstack([cached, new_embeddings])
            np.save(cache_path, full_embeddings)
            return full_embeddings
            
    texts = target_df["clean_resume_text"].fillna("").tolist()
    embeddings = encode_texts(texts, batch_size=batch_size, show_progress=False)
    np.save(cache_path, embeddings)
    return embeddings


def encode_single_resume(text: str) -> np.ndarray:
    """
    Encodes a single resume text (e.g. from newly uploaded resume).
    Returns a 1D float32 array of shape (384,).
    """
    emb = encode_texts([text], batch_size=1, show_progress=False)
    return emb[0]
