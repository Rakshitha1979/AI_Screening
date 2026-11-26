"""
Candidate Ranking and Responsible AI Screening Module.
Ranks candidates against job requirements using fresh semantic similarity scores,
performs transparent skill-gap analysis, maintains audit logs,
and records human recruiter feedback separately without automatic unverified retraining.
"""

import os
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from ai_module.similarity import compute_cosine_similarity
from ai_module.skill_extraction import compute_skill_gap
from ai_module.resume_processing import parse_skills_field

FEEDBACK_FILE = "outputs/recruiter_feedback.csv"


def rank_candidates_for_job(
    job_row: pd.Series,
    candidates_df: pd.DataFrame,
    job_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    top_n: int = 20,
    tfidf_scores: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Ranks candidates against a selected job description using fresh Sentence-BERT
    embeddings and cosine similarity.
    
    Responsible AI Guarantees:
    - Never uses age, gender, race, nationality, or demographic attributes in ranking.
    - Focuses strictly on job-relevant skills, stated experience, and contextual text relevance.
    - Reminds recruiters that similarity is an advisory relevance metric, not a hiring decision.
    """
    # Calculate fresh cosine similarities
    sbert_similarities = compute_cosine_similarity(job_embedding, candidate_embeddings)
    
    # Create ranking dataframe
    ranking_df = candidates_df.copy()
    ranking_df["similarity_score"] = np.round(sbert_similarities, 4)
    
    if tfidf_scores is not None and len(tfidf_scores) == len(ranking_df):
        ranking_df["tfidf_score"] = np.round(tfidf_scores, 4)
        
    # Sort by Sentence-BERT similarity score descending
    ranking_df = ranking_df.sort_values("similarity_score", ascending=False).reset_index(drop=True)
    top_candidates = ranking_df.head(top_n).copy()
    
    job_skills = parse_skills_field(job_row.get("clean_skills", []))
    
    # Compute skill gap analysis for top candidates
    gap_data = []
    for _, cand in top_candidates.iterrows():
        cand_skills = cand.get("parsed_skills", [])
        if not cand_skills:
            cand_skills = parse_skills_field(cand.get("clean_skills", []))
            
        gap = compute_skill_gap(cand_skills, job_skills)
        gap_data.append({
            "matching_skills": ", ".join(gap["matching_skills"]) if gap["matching_skills"] else "None identified",
            "missing_skills": ", ".join(gap["missing_skills"]) if gap["missing_skills"] else "None",
            "skill_coverage_pct": gap["coverage_percentage"],
            "matched_skill_count": gap["match_count"],
            "required_skill_count": gap["required_count"]
        })
        
    gap_df = pd.DataFrame(gap_data)
    for col in gap_df.columns:
        top_candidates[col] = gap_df[col].values
        
    top_candidates["rank"] = range(1, len(top_candidates) + 1)
    
    # Keep recruiter-relevant columns
    display_cols = [
        "rank", "candidate_id", "similarity_score", "skill_coverage_pct",
        "matching_skills", "missing_skills", "years_experience",
        "target_position", "education_text", "resume_text"
    ]
    if "tfidf_score" in top_candidates.columns:
        display_cols.insert(3, "tfidf_score")
        
    existing_cols = [c for c in display_cols if c in top_candidates.columns]
    return top_candidates[existing_cols]


def record_recruiter_feedback(
    candidate_id: str,
    job_id: Any,
    job_title: str,
    decision: str,
    relevance_score: float,
    notes: str = "",
    feedback_path: str = FEEDBACK_FILE
) -> bool:
    """
    Records recruiter review (e.g. 'Relevant', 'Not Relevant', 'Needs Review').
    Saves feedback in a dedicated audit log without unverified automatic model retraining.
    """
    os.makedirs(os.path.dirname(feedback_path), exist_ok=True)
    
    new_entry = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "candidate_id": str(candidate_id),
        "job_id": str(job_id),
        "job_title": str(job_title),
        "recruiter_decision": str(decision),
        "ai_relevance_score": float(relevance_score),
        "notes": str(notes).strip()
    }
    
    new_df = pd.DataFrame([new_entry])
    if os.path.exists(feedback_path):
        new_df.to_csv(feedback_path, mode="a", header=False, index=False)
    else:
        new_df.to_csv(feedback_path, index=False)
        
    return True


def load_recruiter_feedback(feedback_path: str = FEEDBACK_FILE) -> pd.DataFrame:
    """
    Loads historical recruiter feedback if available.
    """
    if os.path.exists(feedback_path):
        return pd.read_csv(feedback_path)
    return pd.DataFrame(columns=[
        "timestamp", "candidate_id", "job_id", "job_title",
        "recruiter_decision", "ai_relevance_score", "notes"
    ])


def identify_uncertain_candidates(
    candidates_df: pd.DataFrame,
    sbert_similarities: np.ndarray,
    job_id: Any,
    tfidf_scores: Optional[np.ndarray] = None,
    decision_threshold: float = 0.45,
    uncertainty_margin: float = 0.05,
    disagreement_threshold: float = 0.20,
    feedback_path: str = FEEDBACK_FILE,
    top_n: int = 30
) -> Dict[str, Any]:
    """
    Identifies candidates with uncertain screening results for Active Learning review.
    
    Uncertainty Detection Methodology:
    1. Decision Boundary Proximity:
       Measures closeness to a defined screening threshold (default: 0.45, margin: ±0.05).
       Candidates closest to the threshold have the highest classification ambiguity.
    2. Model Disagreement (Neural vs. Lexical):
       Identifies candidates where Sentence-BERT semantic similarity and TF-IDF keyword
       matching conflict by >= disagreement_threshold (0.20), highlighting potential
       keyword stuffing or non-standard phrasing requiring human validation.
       
    Active Learning Characteristics:
    - Ranks candidates by uncertainty score descending (most ambiguous cases first).
    - Checks the feedback audit log to separate candidates awaiting review from those already reviewed.
    - Prevents duplicates and maintains exact candidate ID alignment.
    """
    feedback_df = load_recruiter_feedback(feedback_path)
    job_feedback = feedback_df[feedback_df["job_id"].astype(str) == str(job_id)]
    
    reviewed_map = {}
    if not job_feedback.empty:
        # Retain latest review decision per candidate for this requisition
        for _, r in job_feedback.iterrows():
            reviewed_map[str(r["candidate_id"])] = {
                "decision": str(r["recruiter_decision"]),
                "timestamp": str(r.get("timestamp", "")),
                "notes": str(r.get("notes", ""))
            }
            
    records = []
    has_tfidf = tfidf_scores is not None and len(tfidf_scores) == len(candidates_df)
    
    for idx in range(len(candidates_df)):
        cand = candidates_df.iloc[idx]
        cand_id = str(cand["candidate_id"])
        sbert_val = float(sbert_similarities[idx])
        dist = abs(sbert_val - decision_threshold)
        is_borderline = dist <= uncertainty_margin
        
        tfidf_val = float(tfidf_scores[idx]) if has_tfidf else None
        diff = abs(sbert_val - tfidf_val) if tfidf_val is not None else 0.0
        # Disagreement flagged among plausible candidates (sbert >= 0.32)
        is_disagreement = has_tfidf and (diff >= disagreement_threshold) and (sbert_val >= 0.32)
        
        if not (is_borderline or is_disagreement):
            continue
            
        prox_score = max(0.0, 1.0 - (dist / uncertainty_margin)) if is_borderline else 0.0
        div_score = min(1.0, diff / 0.35) if is_disagreement else 0.0
        
        if is_borderline and is_disagreement:
            unc_score = round(0.5 * prox_score + 0.5 * div_score + 0.15, 4)
            reason = f"Borderline score ({sbert_val:.3f} near {decision_threshold}) & Model divergence (|SBERT-TFIDF| = {diff:.3f})"
        elif is_borderline:
            unc_score = round(prox_score, 4)
            reason = f"Borderline Fit: Semantic score ({sbert_val:.3f}) within ±{uncertainty_margin:.2f} of {decision_threshold} decision boundary"
        else:
            unc_score = round(div_score * 0.85, 4)
            if sbert_val > tfidf_val:
                reason = f"Model Divergence: Semantic relevance ({sbert_val:.3f}) vs keyword match ({tfidf_val:.3f}) suggests contextual phrasing"
            else:
                reason = f"Model Divergence: Keyword match ({tfidf_val:.3f}) vs semantic relevance ({sbert_val:.3f}) suggests potential keyword stuffing"
                
        is_reviewed = cand_id in reviewed_map
        review_status = reviewed_map[cand_id]["decision"] if is_reviewed else "Awaiting Review"
        
        cand_name = cand.get("candidate_name")
        if not cand_name or pd.isna(cand_name):
            cand_name = f"Candidate #{cand_id}"
            
        record = {
            "candidate_id": cand_id,
            "candidate_name": cand_name,
            "similarity_score": round(sbert_val, 4),
            "tfidf_score": round(tfidf_val, 4) if tfidf_val is not None else None,
            "uncertainty_score": unc_score,
            "uncertainty_reason": reason,
            "review_status": review_status,
            "target_position": cand.get("target_position", "Applicant"),
            "years_experience": cand.get("years_experience", 0.0),
            "education_text": cand.get("education_text", ""),
            "clean_skills": cand.get("clean_skills", "[]"),
            "parsed_skills": cand.get("parsed_skills", []),
            "resume_text": cand.get("resume_text", ""),
            "review_timestamp": reviewed_map[cand_id]["timestamp"] if is_reviewed else "",
            "review_notes": reviewed_map[cand_id]["notes"] if is_reviewed else ""
        }
        records.append(record)
        
    all_uncertain_df = pd.DataFrame(records)
    if all_uncertain_df.empty:
        empty_cols = [
            "candidate_id", "candidate_name", "similarity_score", "tfidf_score",
            "uncertainty_score", "uncertainty_reason", "review_status"
        ]
        empty_df = pd.DataFrame(columns=empty_cols)
        return {
            "pending_candidates": empty_df,
            "reviewed_candidates": empty_df,
            "total_awaiting_review": 0,
            "total_reviewed": 0,
            "total_flagged": 0,
            "decision_threshold": decision_threshold,
            "uncertainty_margin": uncertainty_margin
        }
        
    all_uncertain_df = all_uncertain_df.sort_values("uncertainty_score", ascending=False).reset_index(drop=True)
    
    pending_df = all_uncertain_df[all_uncertain_df["review_status"] == "Awaiting Review"].reset_index(drop=True)
    reviewed_df = all_uncertain_df[all_uncertain_df["review_status"] != "Awaiting Review"].reset_index(drop=True)
    
    if top_n is not None and top_n > 0:
        pending_df = pending_df.head(top_n)
        
    return {
        "pending_candidates": pending_df,
        "reviewed_candidates": reviewed_df,
        "total_awaiting_review": len(all_uncertain_df[all_uncertain_df["review_status"] == "Awaiting Review"]),
        "total_reviewed": len(reviewed_df),
        "total_flagged": len(all_uncertain_df),
        "decision_threshold": decision_threshold,
        "uncertainty_margin": uncertainty_margin
    }
