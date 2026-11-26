export interface HealthResponse {
  status: string;
  resumes_loaded: number;
  jobs_loaded: number;
  applications_loaded: number;
  cv_embeddings_shape: number[];
  job_embeddings_shape: number[];
}

export interface OverviewCounts {
  resumes: number;
  jobs: number;
  applications: number;
  offers: number;
  anomalies: number;
}

export interface OfferSummary {
  accepted: number;
  declined: number;
  total: number;
  accepted_pct: number;
  declined_pct: number;
}

export interface OverviewResponse {
  counts: OverviewCounts;
  offers: OfferSummary;
}

export interface JobSummary {
  job_id: string;
  job_title: string;
  category: string;
  years_required: number;
}

export interface JobDetail {
  job_id: string;
  job_title: string;
  category: string;
  description: string;
  years_required: number;
  skills: string[];
}

export interface RankedCandidate {
  rank: number;
  candidate_id: string;
  similarity_score: number;
  tfidf_score: number | null;
  skill_coverage_pct: number;
  years_experience: number;
  target_position: string;
  matching_skills: string[];
  missing_skills: string[];
  education_text: string;
}

export interface ScreeningRankResponse {
  job_id: string;
  total_pool_size: number;
  top_n: number;
  candidates: RankedCandidate[];
}

export interface CandidateDetailResponse {
  candidate_id: string;
  job_id: string;
  target_position: string;
  years_experience: number;
  education_text: string;
  similarity_score: number;
  tfidf_score: number | null;
  skill_coverage_pct: number;
  matching_skills: string[];
  missing_skills: string[];
  resume_text: string;
}

export interface UncertainCandidate {
  candidate_id: string;
  candidate_name: string;
  similarity_score: number;
  tfidf_score: number | null;
  uncertainty_score: number;
  uncertainty_reason: string;
  review_status: string;
  target_position: string;
  years_experience: number;
  education_text: string;
  review_timestamp: string;
  review_notes: string;
}

export interface UncertainQueueResponse {
  job_id: string;
  pending: UncertainCandidate[];
  reviewed: UncertainCandidate[];
  total_awaiting_review: number;
  total_reviewed: number;
  total_flagged: number;
  decision_threshold: number;
  uncertainty_margin: number;
}

export interface ScatterSamplePoint {
  candidate_id: string;
  tfidf_score: number;
  similarity_score: number;
  years_experience: number;
}

export interface ModelCompareResponse {
  spearman_rank_correlation: number;
  pearson_score_correlation: number;
  top_k: number;
  top_k_overlap_count: number;
  top_k_overlap_percentage: number;
  tfidf_mean_score: number;
  sbert_mean_score: number;
  sbert_std_score: number;
  tfidf_std_score: number;
  observations: string[];
  scatter_sample: ScatterSamplePoint[];
}

export type CanonicalDecision = "Suitable" | "Not Suitable" | "Needs Further Review";

export interface FeedbackRequest {
  candidate_id: string;
  job_id: string;
  job_title: string;
  decision: CanonicalDecision;
  relevance_score: number;
  notes: string;
}

export interface FeedbackEntry {
  timestamp: string;
  candidate_id: string;
  job_id: string;
  job_title: string;
  recruiter_decision: string;
  ai_relevance_score: number;
  notes: string;
}

export interface HistogramBin {
  bin_start: number;
  bin_end: number;
  label: string;
  count: number;
}

export interface UploadBenchmark {
  placement_rank: number;
  total_candidates: number;
  percentile: number;
  pool_median: number;
  histogram: HistogramBin[];
}

export interface UploadSkillGap {
  matching_skills: string[];
  missing_skills: string[];
  coverage_percentage: number;
  match_count: number;
  required_count: number;
}

export interface UploadScreenResponse {
  success: boolean;
  filename: string;
  cleaned_text: string;
  extracted_skills: string[];
  extracted_education: string[];
  extracted_years_experience: number;
  similarity_score: number;
  skill_gap: UploadSkillGap;
  benchmark: UploadBenchmark;
}

export interface AnalyticsSummaryResponse {
  total_applications: number;
  total_offers: number;
  accepted_offers: number;
  rejected_offers: number;
  acceptance_rate_pct: number;
  declined_rate_pct: number;
  avg_time_to_acceptance_days: number;
  median_time_to_acceptance_days: number;
  valid_tta_samples: number;
  excluded_anomalies: number;
  avg_full_hire_duration_days: number;
  median_full_hire_duration_days: number;
}

export interface FunnelStage {
  stage: string;
  label: string;
  candidate_count: number;
  percentage: number;
}

export interface MonthlyTrend {
  year_month: string;
  application_count: number;
}

export interface SourceStat {
  source: string;
  total_applications: number;
  offers_received: number;
  offers_accepted: number;
  offers_rejected: number;
  offer_rate_pct: number;
  acceptance_rate_pct: number;
}

export interface InterviewOutcome {
  outcome: string;
  count: number;
}

export interface OfferOutcome {
  accepted: boolean;
  label: string;
  count: number;
}

export interface RoleFamilyStat {
  role_family: string;
  count: number;
}

export interface TimeAnalysisResponse {
  tta_count: number;
  tta_mean: number;
  tta_median: number;
  full_count: number;
  full_mean: number;
  full_median: number;
  anomalies_excluded: number;
  tta_histogram: HistogramBin[];
  full_histogram: HistogramBin[];
}

export interface ModelMetricRow {
  Model: string;
  Accuracy: number;
  Precision: number;
  Recall: number;
  "F1-Score": number;
  "ROC-AUC": number;
}

export interface RocCurveData {
  fpr: number[];
  tpr: number[];
  auc: number;
}

export interface FeatureImportanceItem {
  feature: string;
  display_feature: string;
  importance: number | null;
  coefficient: number | null;
  abs_impact: number | null;
}

export interface ModelComparisonResponse {
  metrics_table: ModelMetricRow[];
  confusion_matrices: Record<string, number[][]>;
  roc_data: Record<string, RocCurveData>;
  feature_importances: Record<string, FeatureImportanceItem[]>;
  test_sample_count: number;
  test_accepted_count: number;
  test_rejected_count: number;
}
