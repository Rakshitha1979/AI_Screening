"""Jobs, AI Semantic Screening, Active Learning Queue, and Feedback Audit Log routers."""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from ai_module.candidate_ranking import (
    FEEDBACK_FILE,
    identify_uncertain_candidates,
    load_recruiter_feedback,
    record_recruiter_feedback,
)
from ai_module.similarity import compare_matching_models, compute_cosine_similarity
from ai_module.skill_extraction import compute_skill_gap, parse_skills_field
from backend.schemas import (
    CandidateDetailResponse,
    FeedbackEntry,
    FeedbackRequest,
    JobDetail,
    JobSummary,
    ModelCompareResponse,
    RankedCandidate,
    ScatterSamplePoint,
    ScreeningRankResponse,
    UncertainCandidate,
    UncertainQueueResponse,
)
from backend.state import AppState, get_app_state

router = APIRouter(prefix="/api", tags=["Screening"])


def _resolve_job(state: AppState, job_id: str) -> tuple[int, pd.Series]:
    matches = state.job_df[state.job_df["job_id"].astype(str) == str(job_id)]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    orig_idx = int(matches.index[0])
    return orig_idx, matches.iloc[0]


def _get_or_compute_scores(state: AppState, job_id: str) -> Dict[str, Any]:
    key = str(job_id)
    if key in state.rank_cache:
        return state.rank_cache[key]

    orig_idx, job_row = _resolve_job(state, key)
    job_emb = state.job_embeddings[orig_idx]
    active_cv_count = len(state.cv_embeddings)

    sbert_sims = compute_cosine_similarity(job_emb, state.cv_embeddings)
    tfidf_scores: Optional[np.ndarray] = None
    if state.tfidf_matcher is not None and state.tfidf_matcher.is_fitted:
        job_query_text = str(job_row.get("combined_job_text", "") or "")
        tfidf_scores = state.tfidf_matcher.match(job_query_text)[:active_cv_count]

    sorted_indices = np.argsort(sbert_sims)[::-1]
    cached = {
        "orig_idx": orig_idx,
        "job_row": job_row,
        "sbert_sims": sbert_sims,
        "tfidf_scores": tfidf_scores,
        "sorted_indices": sorted_indices,
    }
    state.rank_cache[key] = cached
    return cached


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None or pd.isna(val):
        return default
    return float(val)


def _safe_str(val: Any, default: str = "") -> str:
    if val is None or pd.isna(val):
        return default
    return str(val)


@router.get("/jobs/categories", response_model=List[str])
def get_job_categories(state: AppState = Depends(get_app_state)) -> List[str]:
    cats = sorted([str(c) for c in state.job_df["category"].dropna().unique().tolist() if str(c).strip()])
    return cats


@router.get("/jobs", response_model=List[JobSummary])
def list_jobs(
    category: Optional[str] = Query(default=None),
    state: AppState = Depends(get_app_state),
) -> List[JobSummary]:
    df = state.job_df
    if category and category != "All Categories":
        df = df[df["category"].astype(str) == category]

    return [
        JobSummary(
            job_id=str(row["job_id"]),
            job_title=_safe_str(row.get("job_title"), "Untitled Role"),
            category=_safe_str(row.get("category"), "General"),
            years_required=int(_safe_float(row.get("years_required"), 0)),
        )
        for _, row in df.iterrows()
    ]


@router.get("/jobs/{job_id}", response_model=JobDetail)
def get_job_detail(job_id: str, state: AppState = Depends(get_app_state)) -> JobDetail:
    _, row = _resolve_job(state, job_id)
    skills = [str(s) for s in parse_skills_field(row.get("clean_skills", []))]
    return JobDetail(
        job_id=str(row["job_id"]),
        job_title=_safe_str(row.get("job_title"), "Untitled Role"),
        category=_safe_str(row.get("category"), "General"),
        description=_safe_str(row.get("job_description"), ""),
        years_required=int(_safe_float(row.get("years_required"), 0)),
        skills=skills,
    )


@router.get("/screening/rank", response_model=ScreeningRankResponse)
def rank_candidates(
    job_id: str = Query(...),
    top_n: int = Query(default=15, ge=1, le=100),
    state: AppState = Depends(get_app_state),
) -> ScreeningRankResponse:
    cached = _get_or_compute_scores(state, job_id)
    job_row = cached["job_row"]
    sbert_sims = cached["sbert_sims"]
    tfidf_scores = cached["tfidf_scores"]
    sorted_indices = cached["sorted_indices"]

    job_skills = parse_skills_field(job_row.get("clean_skills", []))
    active_cv_count = len(state.cv_embeddings)
    top_indices = sorted_indices[:top_n]

    candidates: List[RankedCandidate] = []
    for rank_num, idx in enumerate(top_indices, start=1):
        cand = state.cv_df.iloc[int(idx)]
        cand_skills = cand.get("parsed_skills", [])
        if not cand_skills:
            cand_skills = parse_skills_field(cand.get("clean_skills", []))
        gap = compute_skill_gap(cand_skills, job_skills)
        candidates.append(
            RankedCandidate(
                rank=rank_num,
                candidate_id=str(cand["candidate_id"]),
                similarity_score=round(float(sbert_sims[idx]), 4),
                tfidf_score=round(float(tfidf_scores[idx]), 4) if tfidf_scores is not None else None,
                skill_coverage_pct=float(gap["coverage_percentage"]),
                years_experience=round(_safe_float(cand.get("years_experience"), 0.0), 1),
                target_position=_safe_str(cand.get("target_position"), "Applicant"),
                matching_skills=[str(s) for s in gap["matching_skills"]],
                missing_skills=[str(s) for s in gap["missing_skills"]],
                education_text=_safe_str(cand.get("education_text"), "Not specified"),
            )
        )

    return ScreeningRankResponse(
        job_id=str(job_id),
        total_pool_size=active_cv_count,
        top_n=top_n,
        candidates=candidates,
    )


@router.get("/screening/candidate/{candidate_id}", response_model=CandidateDetailResponse)
def get_candidate_detail(
    candidate_id: str,
    job_id: str = Query(...),
    state: AppState = Depends(get_app_state),
) -> CandidateDetailResponse:
    cached = _get_or_compute_scores(state, job_id)
    job_row = cached["job_row"]
    sbert_sims = cached["sbert_sims"]
    tfidf_scores = cached["tfidf_scores"]

    active_cv_count = len(state.cv_embeddings)
    active_df = state.cv_df.iloc[:active_cv_count]
    cand_matches = active_df[active_df["candidate_id"].astype(str) == str(candidate_id)]
    if cand_matches.empty:
        raise HTTPException(status_code=404, detail=f"Candidate '{candidate_id}' not found.")

    idx = int(cand_matches.index[0])
    cand = cand_matches.iloc[0]
    job_skills = parse_skills_field(job_row.get("clean_skills", []))
    cand_skills = cand.get("parsed_skills", [])
    if not cand_skills:
        cand_skills = parse_skills_field(cand.get("clean_skills", []))
    gap = compute_skill_gap(cand_skills, job_skills)

    return CandidateDetailResponse(
        candidate_id=str(cand["candidate_id"]),
        job_id=str(job_id),
        target_position=_safe_str(cand.get("target_position"), "Applicant"),
        years_experience=round(_safe_float(cand.get("years_experience"), 0.0), 1),
        education_text=_safe_str(cand.get("education_text"), "Not specified"),
        similarity_score=round(float(sbert_sims[idx]), 4),
        tfidf_score=round(float(tfidf_scores[idx]), 4) if tfidf_scores is not None else None,
        skill_coverage_pct=float(gap["coverage_percentage"]),
        matching_skills=[str(s) for s in gap["matching_skills"]],
        missing_skills=[str(s) for s in gap["missing_skills"]],
        resume_text=_safe_str(cand.get("resume_text"), ""),
    )


@router.get("/screening/uncertain", response_model=UncertainQueueResponse)
def get_uncertain_queue(
    job_id: str = Query(...),
    state: AppState = Depends(get_app_state),
) -> UncertainQueueResponse:
    cached = _get_or_compute_scores(state, job_id)
    active_cv_count = len(state.cv_embeddings)
    active_cv_df = state.cv_df.iloc[:active_cv_count]

    res = identify_uncertain_candidates(
        candidates_df=active_cv_df,
        sbert_similarities=cached["sbert_sims"],
        job_id=str(job_id),
        tfidf_scores=cached["tfidf_scores"],
        decision_threshold=0.45,
        uncertainty_margin=0.05,
        disagreement_threshold=0.20,
        top_n=30,
    )

    def _to_models(df: pd.DataFrame) -> List[UncertainCandidate]:
        items: List[UncertainCandidate] = []
        for _, r in df.iterrows():
            tfidf_val = r.get("tfidf_score")
            items.append(
                UncertainCandidate(
                    candidate_id=str(r["candidate_id"]),
                    candidate_name=_safe_str(r.get("candidate_name"), f"Candidate #{r['candidate_id']}"),
                    similarity_score=float(r["similarity_score"]),
                    tfidf_score=float(tfidf_val) if tfidf_val is not None and not pd.isna(tfidf_val) else None,
                    uncertainty_score=float(r["uncertainty_score"]),
                    uncertainty_reason=_safe_str(r.get("uncertainty_reason"), ""),
                    review_status=_safe_str(r.get("review_status"), "Awaiting Review"),
                    target_position=_safe_str(r.get("target_position"), "Applicant"),
                    years_experience=round(_safe_float(r.get("years_experience"), 0.0), 1),
                    education_text=_safe_str(r.get("education_text"), "Not specified"),
                    review_timestamp=_safe_str(r.get("review_timestamp"), ""),
                    review_notes=_safe_str(r.get("review_notes"), ""),
                )
            )
        return items

    return UncertainQueueResponse(
        job_id=str(job_id),
        pending=_to_models(res["pending_candidates"]),
        reviewed=_to_models(res["reviewed_candidates"]),
        total_awaiting_review=int(res["total_awaiting_review"]),
        total_reviewed=int(res["total_reviewed"]),
        total_flagged=int(res["total_flagged"]),
        decision_threshold=float(res["decision_threshold"]),
        uncertainty_margin=float(res["uncertainty_margin"]),
    )


@router.get("/screening/compare", response_model=ModelCompareResponse)
def compare_screening_models(
    job_id: str = Query(...),
    top_k: int = Query(default=15, ge=1, le=50),
    state: AppState = Depends(get_app_state),
) -> ModelCompareResponse:
    cached = _get_or_compute_scores(state, job_id)
    sbert_sims = cached["sbert_sims"]
    tfidf_scores = cached["tfidf_scores"]
    if tfidf_scores is None:
        raise HTTPException(status_code=500, detail="TF-IDF matcher is not fitted.")

    active_cv_count = len(state.cv_embeddings)
    active_df = state.cv_df.iloc[:active_cv_count]
    candidate_ids = [str(cid) for cid in active_df["candidate_id"].tolist()]

    comp = compare_matching_models(
        tfidf_scores=tfidf_scores,
        sbert_scores=sbert_sims,
        candidate_ids=candidate_ids,
        top_k=top_k,
    )

    rng = np.random.default_rng(42)
    sample_size = min(300, active_cv_count)
    sample_idx = rng.choice(active_cv_count, size=sample_size, replace=False)

    scatter_sample = [
        ScatterSamplePoint(
            candidate_id=str(active_df.iloc[int(i)]["candidate_id"]),
            tfidf_score=round(float(tfidf_scores[int(i)]), 4),
            similarity_score=round(float(sbert_sims[int(i)]), 4),
            years_experience=round(_safe_float(active_df.iloc[int(i)].get("years_experience"), 0.0), 1),
        )
        for i in sample_idx
    ]

    return ModelCompareResponse(
        spearman_rank_correlation=float(comp["spearman_rank_correlation"]),
        pearson_score_correlation=float(comp["pearson_score_correlation"]),
        top_k=int(comp["top_k"]),
        top_k_overlap_count=int(comp["top_k_overlap_count"]),
        top_k_overlap_percentage=float(comp["top_k_overlap_percentage"]),
        tfidf_mean_score=float(comp["tfidf_mean_score"]),
        sbert_mean_score=float(comp["sbert_mean_score"]),
        sbert_std_score=float(comp["sbert_std_score"]),
        tfidf_std_score=float(comp["tfidf_std_score"]),
        observations=[str(o) for o in comp["observations"]],
        scatter_sample=scatter_sample,
    )


@router.post("/feedback", response_model=FeedbackEntry)
def submit_feedback(
    payload: FeedbackRequest,
    state: AppState = Depends(get_app_state),
) -> FeedbackEntry:
    with state.feedback_lock:
        record_recruiter_feedback(
            candidate_id=str(payload.candidate_id),
            job_id=str(payload.job_id),
            job_title=payload.job_title,
            decision=payload.decision,
            relevance_score=float(payload.relevance_score),
            notes=payload.notes,
            feedback_path=FEEDBACK_FILE,
        )
        df = load_recruiter_feedback(FEEDBACK_FILE)

    last_row = df.iloc[-1]
    return FeedbackEntry(
        timestamp=str(last_row["timestamp"]),
        candidate_id=str(last_row["candidate_id"]),
        job_id=str(last_row["job_id"]),
        job_title=_safe_str(last_row["job_title"]),
        recruiter_decision=_safe_str(last_row["recruiter_decision"]),
        ai_relevance_score=float(last_row["ai_relevance_score"]),
        notes=_safe_str(last_row.get("notes"), ""),
    )


@router.get("/feedback", response_model=List[FeedbackEntry])
def list_feedback(
    job_id: Optional[str] = Query(default=None),
    state: AppState = Depends(get_app_state),
) -> List[FeedbackEntry]:
    with state.feedback_lock:
        df = load_recruiter_feedback(FEEDBACK_FILE)

    if df.empty:
        return []

    if job_id is not None:
        df = df[df["job_id"].astype(str) == str(job_id)]

    df = df.iloc[::-1]
    return [
        FeedbackEntry(
            timestamp=_safe_str(r["timestamp"]),
            candidate_id=str(r["candidate_id"]),
            job_id=str(r["job_id"]),
            job_title=_safe_str(r.get("job_title"), ""),
            recruiter_decision=_safe_str(r.get("recruiter_decision"), ""),
            ai_relevance_score=_safe_float(r.get("ai_relevance_score"), 0.0),
            notes=_safe_str(r.get("notes"), ""),
        )
        for _, r in df.iterrows()
    ]
