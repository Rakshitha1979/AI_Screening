"""
Pydantic v2 Response and Request Schemas for the FastAPI Backend.
"""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    resumes_loaded: int
    jobs_loaded: int
    applications_loaded: int
    cv_embeddings_shape: List[int]
    job_embeddings_shape: List[int]


class OfferSummary(BaseModel):
    accepted: int
    declined: int
    total: int
    accepted_pct: float
    declined_pct: float


class OverviewCounts(BaseModel):
    resumes: int
    jobs: int
    applications: int
    offers: int
    anomalies: int


class OverviewResponse(BaseModel):
    counts: OverviewCounts
    offers: OfferSummary


class JobSummary(BaseModel):
    job_id: str
    job_title: str
    category: str
    years_required: int


class JobDetail(BaseModel):
    job_id: str
    job_title: str
    category: str
    description: str
    years_required: int
    skills: List[str]


class RankedCandidate(BaseModel):
    rank: int
    candidate_id: str
    similarity_score: float
    tfidf_score: Optional[float] = None
    skill_coverage_pct: float
    years_experience: float
    target_position: str
    matching_skills: List[str]
    missing_skills: List[str]
    education_text: str


class ScreeningRankResponse(BaseModel):
    job_id: str
    total_pool_size: int
    top_n: int
    candidates: List[RankedCandidate]


class CandidateDetailResponse(BaseModel):
    candidate_id: str
    job_id: str
    target_position: str
    years_experience: float
    education_text: str
    similarity_score: float
    tfidf_score: Optional[float] = None
    skill_coverage_pct: float
    matching_skills: List[str]
    missing_skills: List[str]
    resume_text: str


class UncertainCandidate(BaseModel):
    candidate_id: str
    candidate_name: str
    similarity_score: float
    tfidf_score: Optional[float] = None
    uncertainty_score: float
    uncertainty_reason: str
    review_status: str
    target_position: str
    years_experience: float
    education_text: str
    review_timestamp: str = ""
    review_notes: str = ""


class UncertainQueueResponse(BaseModel):
    job_id: str
    pending: List[UncertainCandidate]
    reviewed: List[UncertainCandidate]
    total_awaiting_review: int
    total_reviewed: int
    total_flagged: int
    decision_threshold: float
    uncertainty_margin: float


class ScatterSamplePoint(BaseModel):
    candidate_id: str
    tfidf_score: float
    similarity_score: float
    years_experience: float


class ModelCompareResponse(BaseModel):
    spearman_rank_correlation: float
    pearson_score_correlation: float
    top_k: int
    top_k_overlap_count: int
    top_k_overlap_percentage: float
    tfidf_mean_score: float
    sbert_mean_score: float
    sbert_std_score: float
    tfidf_std_score: float
    observations: List[str]
    scatter_sample: List[ScatterSamplePoint]


CanonicalDecision = Literal["Suitable", "Not Suitable", "Needs Further Review"]


class FeedbackRequest(BaseModel):
    candidate_id: str = Field(..., min_length=1)
    job_id: str = Field(..., min_length=1)
    job_title: str = Field(..., min_length=1)
    decision: CanonicalDecision
    relevance_score: float
    notes: str = ""


class FeedbackEntry(BaseModel):
    timestamp: str
    candidate_id: str
    job_id: str
    job_title: str
    recruiter_decision: str
    ai_relevance_score: float
    notes: str = ""


class HistogramBin(BaseModel):
    bin_start: float
    bin_end: float
    label: str
    count: int


class UploadBenchmark(BaseModel):
    placement_rank: int
    total_candidates: int
    percentile: float
    pool_median: float
    histogram: List[HistogramBin]


class UploadSkillGap(BaseModel):
    matching_skills: List[str]
    missing_skills: List[str]
    coverage_percentage: float
    match_count: int
    required_count: int


class UploadScreenResponse(BaseModel):
    success: bool
    filename: str
    cleaned_text: str
    extracted_skills: List[str]
    extracted_education: List[str]
    extracted_years_experience: float
    similarity_score: float
    skill_gap: UploadSkillGap
    benchmark: UploadBenchmark


class UploadDemoRequest(BaseModel):
    job_id: str


class AnalyticsSummaryResponse(BaseModel):
    total_applications: int
    total_offers: int
    accepted_offers: int
    rejected_offers: int
    acceptance_rate_pct: float
    declined_rate_pct: float
    avg_time_to_acceptance_days: float
    median_time_to_acceptance_days: float
    valid_tta_samples: int
    excluded_anomalies: int
    avg_full_hire_duration_days: float
    median_full_hire_duration_days: float


class FunnelStage(BaseModel):
    stage: str
    label: str
    candidate_count: int
    percentage: float


class MonthlyTrend(BaseModel):
    year_month: str
    application_count: int


class SourceStat(BaseModel):
    source: str
    total_applications: int
    offers_received: int
    offers_accepted: int
    offers_rejected: int
    offer_rate_pct: float
    acceptance_rate_pct: float


class InterviewOutcome(BaseModel):
    outcome: str
    count: int


class OfferOutcome(BaseModel):
    accepted: bool
    label: str
    count: int


class RoleFamilyStat(BaseModel):
    role_family: str
    count: int


class TimeAnalysisResponse(BaseModel):
    tta_count: int
    tta_mean: float
    tta_median: float
    full_count: int
    full_mean: float
    full_median: float
    anomalies_excluded: int
    tta_histogram: List[HistogramBin]
    full_histogram: List[HistogramBin]


class ModelMetricRow(BaseModel):
    Model: str
    Accuracy: float
    Precision: float
    Recall: float
    F1_Score: float = Field(..., alias="F1-Score")
    ROC_AUC: float = Field(..., alias="ROC-AUC")

    model_config = {"populate_by_name": True}


class RocCurveData(BaseModel):
    fpr: List[float]
    tpr: List[float]
    auc: float


class FeatureImportanceItem(BaseModel):
    feature: str
    display_feature: str
    importance: Optional[float] = None
    coefficient: Optional[float] = None
    abs_impact: Optional[float] = None


class ModelComparisonResponse(BaseModel):
    metrics_table: List[ModelMetricRow]
    confusion_matrices: Dict[str, List[List[int]]]
    roc_data: Dict[str, RocCurveData]
    feature_importances: Dict[str, List[FeatureImportanceItem]]
    test_sample_count: int
    test_accepted_count: int
    test_rejected_count: int
