"""
Text Similarity and AI Matching Module.
Implements TF-IDF vectorization as a traditional baseline,
dense Sentence-BERT cosine similarity, rank comparison metrics,
and transparent evaluation without fabricated labels.
"""

from typing import List, Dict, Any, Tuple, Optional, Set
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import spearmanr, pearsonr


def compute_cosine_similarity(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Computes cosine similarity between a single vector (D,) and a matrix (N, D).
    If inputs are already L2-normalized, this is equivalent to dot product.
    Returns 1D array of similarity scores bounded in [-1.0, 1.0].
    """
    vec = np.asarray(vec, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)
    
    vec_norm = np.linalg.norm(vec)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    
    # Avoid division by zero
    vec_norm = vec_norm if vec_norm > 1e-9 else 1.0
    matrix_norms = np.where(matrix_norms > 1e-9, matrix_norms, 1.0)
    
    dot_products = np.dot(matrix, vec)
    similarities = dot_products / (matrix_norms * vec_norm)
    return np.clip(similarities, -1.0, 1.0)


class TfidfBaselineMatcher:
    """
    Traditional TF-IDF Vectorizer matching baseline.
    Relies on lexical term frequency-inverse document frequency overlap.
    """
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english"
        )
        self.is_fitted = False
        self.corpus_matrix: Optional[np.ndarray] = None
        
    def fit_corpus(self, texts: List[str]):
        """
        Fits the TF-IDF vocabulary on candidate resume texts.
        """
        self.corpus_matrix = self.vectorizer.fit_transform(texts).toarray()
        self.is_fitted = True
        
    def match(self, query_text: str) -> np.ndarray:
        """
        Transforms query job description text and computes cosine similarity against corpus.
        """
        if not self.is_fitted or self.corpus_matrix is None:
            raise ValueError("TfidfBaselineMatcher must be fitted before matching.")
            
        query_vec = self.vectorizer.transform([query_text]).toarray()[0]
        return compute_cosine_similarity(query_vec, self.corpus_matrix)


def compare_matching_models(
    tfidf_scores: np.ndarray,
    sbert_scores: np.ndarray,
    candidate_ids: List[str],
    top_k: int = 10
) -> Dict[str, Any]:
    """
    Compares TF-IDF baseline and Sentence-BERT matching behavior:
    - Rank correlation (Spearman rho)
    - Score correlation (Pearson r)
    - Top-K candidate overlap
    - Methodological difference explanation
    """
    # Pearson and Spearman correlations across all candidates
    spearman_corr, _ = spearmanr(tfidf_scores, sbert_scores)
    pearson_corr, _ = pearsonr(tfidf_scores, sbert_scores)
    
    # Top-K candidate sets
    top_k_tfidf_indices = np.argsort(tfidf_scores)[::-1][:top_k]
    top_k_sbert_indices = np.argsort(sbert_scores)[::-1][:top_k]
    
    top_k_tfidf_ids = set([candidate_ids[i] for i in top_k_tfidf_indices])
    top_k_sbert_ids = set([candidate_ids[i] for i in top_k_sbert_indices])
    
    overlap_count = len(top_k_tfidf_ids.intersection(top_k_sbert_ids))
    overlap_pct = round((overlap_count / top_k * 100.0), 1)
    
    return {
        "spearman_rank_correlation": round(float(spearman_corr), 3),
        "pearson_score_correlation": round(float(pearson_corr), 3),
        "top_k": top_k,
        "top_k_overlap_count": overlap_count,
        "top_k_overlap_percentage": overlap_pct,
        "tfidf_mean_score": round(float(np.mean(tfidf_scores)), 4),
        "sbert_mean_score": round(float(np.mean(sbert_scores)), 4),
        "sbert_std_score": round(float(np.std(sbert_scores)), 4),
        "tfidf_std_score": round(float(np.std(tfidf_scores)), 4),
        "observations": [
            "TF-IDF relies purely on explicit keyword co-occurrence; it yields near-zero similarity when vocabulary differs even if meaning is identical.",
            "Sentence-BERT captures deep contextual semantics, synonymy, and implied competencies even when exact phrasing diverges.",
            "A candidate matching score represents semantic text relevance, not a probability or guarantee of hiring suitability."
        ]
    }


def compute_precision_recall_at_k(
    ranked_candidate_ids: List[str],
    relevant_candidate_ids: Set[str],
    k: int = 10
) -> Dict[str, float]:
    """
    Calculates Precision@K and Recall@K based on a set of known relevant candidate IDs.
    Returns 0.0 if relevant set is empty.
    """
    if not relevant_candidate_ids or k <= 0:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "k": k}
        
    top_k_candidates = ranked_candidate_ids[:k]
    hits = len(set(top_k_candidates).intersection(relevant_candidate_ids))
    
    precision = hits / k
    recall = hits / len(relevant_candidate_ids)
    
    return {
        "k": k,
        "hits": hits,
        "relevant_total": len(relevant_candidate_ids),
        "precision_at_k": round(precision, 3),
        "recall_at_k": round(recall, 3)
    }
