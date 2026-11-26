"""
Singleton Application State Loader for FastAPI.
Loads datasets, precomputed SBERT embeddings, TF-IDF matcher, and IDA model evaluation
results once at startup (replacing Streamlit's @st.cache_data / @st.cache_resource).
"""

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from fastapi import Request

from ai_module.resume_processing import load_and_preprocess_cvs, load_and_preprocess_jobs
from ai_module.sentence_bert import get_or_create_cv_embeddings, get_or_create_job_embeddings
from ai_module.similarity import TfidfBaselineMatcher
from ida_module.evaluation import evaluate_all_models
from ida_module.model_training import train_offer_models
from ida_module.preprocessing import preprocess_recruitment_data


@dataclass
class AppState:
    cv_df: pd.DataFrame
    job_df: pd.DataFrame
    matches_df: pd.DataFrame
    ida_df: pd.DataFrame
    audit_info: Dict[str, Any]
    job_embeddings: np.ndarray
    cv_embeddings: np.ndarray
    tfidf_matcher: TfidfBaselineMatcher
    evaluation_results: Dict[str, Any]
    rank_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    feedback_lock: threading.Lock = field(default_factory=threading.Lock)


_STATE_LOCK = threading.Lock()
_GLOBAL_STATE: Optional[AppState] = None


def load_app_state() -> AppState:
    """
    Loads all datasets, embeddings, TF-IDF baseline, and evaluation artifacts once.
    Falls back to generating cleaned data / trained models if cached outputs are missing.
    """
    cv_df = load_and_preprocess_cvs("data/cv.csv")
    job_df = load_and_preprocess_jobs("data/job.csv")
    matches_df = pd.read_csv("data/matches.csv")

    cleaned_ida_path = "outputs/cleaned_data/ida_cleaned_recruitment.csv"
    if os.path.exists(cleaned_ida_path):
        ida_df = pd.read_csv(cleaned_ida_path)
        audit_info = {
            "total_offers": int(ida_df["offered_at"].notnull().sum()),
            "offers_accepted_count": int((ida_df["accepted"] == True).sum()),
            "offers_rejected_count": int((ida_df["accepted"] == False).sum()),
            "excluded_anomaly_records": int((ida_df["offer_date_anomaly"] == True).sum()),
        }
    else:
        ida_df, audit_info = preprocess_recruitment_data()

    job_embeddings = get_or_create_job_embeddings(job_df)
    cv_embeddings = get_or_create_cv_embeddings(cv_df)

    tfidf_matcher = TfidfBaselineMatcher(max_features=5000)
    corpus_texts = cv_df["clean_resume_text"].tolist()
    tfidf_matcher.fit_corpus(corpus_texts)

    eval_json_path = "outputs/models/evaluation_results.json"
    if os.path.exists(eval_json_path):
        with open(eval_json_path, "r", encoding="utf-8") as f:
            evaluation_results = json.load(f)
    else:
        train_out = train_offer_models(ida_df)
        evaluation_results = evaluate_all_models(
            models_dict=train_out["models"],
            X_test=train_out["data"]["X_test"],
            y_test=train_out["data"]["y_test"],
            feature_names=train_out["all_feature_names"],
        )

    return AppState(
        cv_df=cv_df,
        job_df=job_df,
        matches_df=matches_df,
        ida_df=ida_df,
        audit_info=audit_info,
        job_embeddings=job_embeddings,
        cv_embeddings=cv_embeddings,
        tfidf_matcher=tfidf_matcher,
        evaluation_results=evaluation_results,
    )


def get_or_init_state() -> AppState:
    global _GLOBAL_STATE
    if _GLOBAL_STATE is None:
        with _STATE_LOCK:
            if _GLOBAL_STATE is None:
                _GLOBAL_STATE = load_app_state()
    return _GLOBAL_STATE


def get_app_state(request: Request) -> AppState:
    state = getattr(request.app.state, "app_state", None)
    if state is None:
        state = get_or_init_state()
        request.app.state.app_state = state
    return state
