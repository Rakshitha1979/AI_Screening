import type {
  AnalyticsSummaryResponse,
  CandidateDetailResponse,
  FeedbackEntry,
  FeedbackRequest,
  FunnelStage,
  HealthResponse,
  InterviewOutcome,
  JobDetail,
  JobSummary,
  ModelCompareResponse,
  ModelComparisonResponse,
  MonthlyTrend,
  OfferOutcome,
  OverviewResponse,
  RoleFamilyStat,
  ScreeningRankResponse,
  SourceStat,
  TimeAnalysisResponse,
  UncertainQueueResponse,
  UploadScreenResponse,
} from "./types";

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let msg = `Request failed (${res.status})`;
    try {
      const body = (await res.json()) as { error?: string; detail?: string };
      if (body.error) msg = body.error;
      else if (body.detail) msg = body.detail;
    } catch {
      // keep default status error
    }
    throw new Error(msg);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => fetchJson<HealthResponse>("/api/health"),
  overview: () => fetchJson<OverviewResponse>("/api/overview"),

  jobCategories: () => fetchJson<string[]>("/api/jobs/categories"),
  jobs: (category?: string) => {
    const q = category && category !== "All Categories" ? `?category=${encodeURIComponent(category)}` : "";
    return fetchJson<JobSummary[]>(`/api/jobs${q}`);
  },
  jobDetail: (jobId: string) => fetchJson<JobDetail>(`/api/jobs/${encodeURIComponent(jobId)}`),

  screeningRank: (jobId: string, topN: number) =>
    fetchJson<ScreeningRankResponse>(
      `/api/screening/rank?job_id=${encodeURIComponent(jobId)}&top_n=${topN}`
    ),
  candidateDetail: (candidateId: string, jobId: string) =>
    fetchJson<CandidateDetailResponse>(
      `/api/screening/candidate/${encodeURIComponent(candidateId)}?job_id=${encodeURIComponent(jobId)}`
    ),
  uncertainQueue: (jobId: string) =>
    fetchJson<UncertainQueueResponse>(`/api/screening/uncertain?job_id=${encodeURIComponent(jobId)}`),
  compareModels: (jobId: string, topK: number) =>
    fetchJson<ModelCompareResponse>(
      `/api/screening/compare?job_id=${encodeURIComponent(jobId)}&top_k=${topK}`
    ),

  submitFeedback: (payload: FeedbackRequest) =>
    fetchJson<FeedbackEntry>("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  listFeedback: (jobId?: string) => {
    const q = jobId ? `?job_id=${encodeURIComponent(jobId)}` : "";
    return fetchJson<FeedbackEntry[]>(`/api/feedback${q}`);
  },

  uploadScreen: (file: File, jobId: string) => {
    const form = new FormData();
    form.append("file", file);
    form.append("job_id", jobId);
    return fetchJson<UploadScreenResponse>("/api/upload/screen", {
      method: "POST",
      body: form,
    });
  },
  uploadDemo: (jobId: string) =>
    fetchJson<UploadScreenResponse>("/api/upload/demo", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: jobId }),
    }),

  analyticsSummary: () => fetchJson<AnalyticsSummaryResponse>("/api/analytics/summary"),
  analyticsFunnel: () => fetchJson<FunnelStage[]>("/api/analytics/funnel"),
  analyticsMonthly: () => fetchJson<MonthlyTrend[]>("/api/analytics/monthly"),
  analyticsSources: () => fetchJson<SourceStat[]>("/api/analytics/sources"),
  analyticsInterviews: () => fetchJson<InterviewOutcome[]>("/api/analytics/interviews"),
  analyticsOffers: () => fetchJson<OfferOutcome[]>("/api/analytics/offers"),
  analyticsRoles: () => fetchJson<RoleFamilyStat[]>("/api/analytics/roles"),
  analyticsTime: () => fetchJson<TimeAnalysisResponse>("/api/analytics/time"),

  modelsComparison: () => fetchJson<ModelComparisonResponse>("/api/models"),
};
