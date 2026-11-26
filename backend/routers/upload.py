"""Resume Upload & Demo Screening router."""

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ai_module.resume_upload import screen_uploaded_resume
from ai_module.sentence_bert import encode_single_resume
from ai_module.similarity import compute_cosine_similarity
from ai_module.skill_extraction import (
    compute_skill_gap,
    extract_education,
    extract_skills_from_text,
    extract_years_experience,
    parse_skills_field,
)
from backend.schemas import (
    HistogramBin,
    UploadBenchmark,
    UploadDemoRequest,
    UploadScreenResponse,
    UploadSkillGap,
)
from backend.state import AppState, get_app_state

router = APIRouter(prefix="/api/upload", tags=["Upload"])

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

SAMPLE_RESUME_TEXT = (
    "SENIOR SOFTWARE ENGINEER & ML SPECIALIST\n"
    "Summary: Over 6 years of experience building high-throughput distributed systems and machine learning workflows.\n"
    "Core Competencies: Python, Machine Learning, Deep Learning, PyTorch, Docker, Kubernetes, AWS, PostgreSQL, REST API, Agile.\n"
    "Education: Master of Science (M.S.) in Computer Science, B.Tech in Information Technology.\n"
    "Experience: Led development of real-time recommendation engines on AWS using Python and PyTorch. "
    "Containerized microservices using Docker and Kubernetes."
)


def _resolve_job(state: AppState, job_id: str):
    matches = state.job_df[state.job_df["job_id"].astype(str) == str(job_id)]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    orig_idx = int(matches.index[0])
    return matches.iloc[0], state.job_embeddings[orig_idx]


def _build_benchmark(job_embedding: np.ndarray, cv_embeddings: np.ndarray, score: float) -> UploadBenchmark:
    existing_sims = compute_cosine_similarity(job_embedding, cv_embeddings)
    total = int(len(existing_sims))
    percentile = round(float((existing_sims < score).mean() * 100.0), 1)
    placement_rank = int((existing_sims >= score).sum()) + 1
    pool_median = round(float(np.median(existing_sims)), 4)

    counts, edges = np.histogram(existing_sims, bins=30)
    histogram = [
        HistogramBin(
            bin_start=round(float(edges[i]), 3),
            bin_end=round(float(edges[i + 1]), 3),
            label=f"{edges[i]:.2f}",
            count=int(counts[i]),
        )
        for i in range(len(counts))
    ]
    return UploadBenchmark(
        placement_rank=placement_rank,
        total_candidates=total,
        percentile=percentile,
        pool_median=pool_median,
        histogram=histogram,
    )


@router.post("/screen", response_model=UploadScreenResponse)
async def upload_and_screen(
    file: UploadFile = File(...),
    job_id: str = Form(...),
    state: AppState = Depends(get_app_state),
):
    job_row, job_emb = _resolve_job(state, job_id)
    file_bytes = await file.read()

    if not file_bytes:
        return JSONResponse(status_code=422, content={"error": "Uploaded file is empty."})
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        return JSONResponse(status_code=422, content={"error": "File exceeds the 10 MB size limit."})

    filename = file.filename or "resume"
    result = screen_uploaded_resume(
        file_bytes=file_bytes,
        filename=filename,
        job_row=job_row,
        job_embedding=job_emb,
    )

    if not result.get("success"):
        return JSONResponse(status_code=422, content={"error": result.get("error", "Unable to parse resume.")})

    benchmark = _build_benchmark(job_emb, state.cv_embeddings, float(result["similarity_score"]))
    gap = result["skill_gap"]

    return UploadScreenResponse(
        success=True,
        filename=str(result["filename"]),
        cleaned_text=str(result["cleaned_text"]),
        extracted_skills=[str(s) for s in result["extracted_skills"]],
        extracted_education=[str(e) for e in result["extracted_education"]],
        extracted_years_experience=float(result["extracted_years_experience"]),
        similarity_score=float(result["similarity_score"]),
        skill_gap=UploadSkillGap(
            matching_skills=[str(s) for s in gap["matching_skills"]],
            missing_skills=[str(s) for s in gap["missing_skills"]],
            coverage_percentage=float(gap["coverage_percentage"]),
            match_count=int(gap["match_count"]),
            required_count=int(gap["required_count"]),
        ),
        benchmark=benchmark,
    )


@router.post("/demo", response_model=UploadScreenResponse)
def demo_screen(
    payload: UploadDemoRequest,
    state: AppState = Depends(get_app_state),
) -> UploadScreenResponse:
    job_row, job_emb = _resolve_job(state, payload.job_id)
    job_skills = parse_skills_field(job_row.get("clean_skills", []))

    cand_skills = extract_skills_from_text(SAMPLE_RESUME_TEXT)
    cand_edu = extract_education(SAMPLE_RESUME_TEXT)
    cand_exp = extract_years_experience(SAMPLE_RESUME_TEXT)
    gap = compute_skill_gap(cand_skills, job_skills)

    cand_emb = encode_single_resume(SAMPLE_RESUME_TEXT)
    sim = round(float(compute_cosine_similarity(job_emb, cand_emb.reshape(1, -1))[0]), 4)
    benchmark = _build_benchmark(job_emb, state.cv_embeddings, sim)

    return UploadScreenResponse(
        success=True,
        filename="Sample_Candidate_Profile.pdf",
        cleaned_text=SAMPLE_RESUME_TEXT,
        extracted_skills=[str(s) for s in cand_skills],
        extracted_education=[str(e) for e in cand_edu],
        extracted_years_experience=float(cand_exp),
        similarity_score=sim,
        skill_gap=UploadSkillGap(
            matching_skills=[str(s) for s in gap["matching_skills"]],
            missing_skills=[str(s) for s in gap["missing_skills"]],
            coverage_percentage=float(gap["coverage_percentage"]),
            match_count=int(gap["match_count"]),
            required_count=int(gap["required_count"]),
        ),
        benchmark=benchmark,
    )
