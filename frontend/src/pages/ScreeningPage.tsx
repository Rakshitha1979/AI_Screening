import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ChevronDown, ChevronUp } from "lucide-react";
import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api/client";
import type { CanonicalDecision, RankedCandidate, UncertainCandidate } from "../api/types";
import { InfoCallout } from "../components/Callout";
import { DataTable, type ColumnDef } from "../components/DataTable";
import { PageSkeleton } from "../components/ErrorBoundary";
import { MetricCard } from "../components/MetricCard";
import { SectionHeader } from "../components/SectionHeader";
import { SkillBadge } from "../components/SkillBadge";

export function ScreeningPage() {
  const queryClient = useQueryClient();
  const [category, setCategory] = useState<string>("All Categories");
  const [jobId, setJobId] = useState<string>("");
  const [topN, setTopN] = useState<number>(15);
  const [detailsOpen, setDetailsOpen] = useState<boolean>(true);
  const [selectedCandId, setSelectedCandId] = useState<string>("");
  const [resumeOpen, setResumeOpen] = useState<boolean>(false);
  const [obsOpen, setObsOpen] = useState<boolean>(false);

  // Recruiter feedback form state
  const [decision, setDecision] = useState<CanonicalDecision>("Suitable");
  const [notes, setNotes] = useState<string>("");
  const [toastMsg, setToastMsg] = useState<string>("");

  // Active learning queue state
  const [queueTab, setQueueTab] = useState<"pending" | "reviewed">("pending");
  const [selectedUncId, setSelectedUncId] = useState<string>("");
  const [uncDecision, setUncDecision] = useState<CanonicalDecision>("Suitable");
  const [uncNotes, setUncNotes] = useState<string>("");
  const [uncToast, setUncToast] = useState<string>("");

  const categoriesQ = useQuery({ queryKey: ["jobCategories"], queryFn: api.jobCategories });
  const jobsQ = useQuery({
    queryKey: ["jobs", category],
    queryFn: () => api.jobs(category),
  });

  useEffect(() => {
    if (jobsQ.data && jobsQ.data.length > 0) {
      if (!jobId || !jobsQ.data.some((j) => j.job_id === jobId)) {
        setJobId(jobsQ.data[0].job_id);
      }
    }
  }, [jobsQ.data, jobId]);

  const jobDetailQ = useQuery({
    queryKey: ["jobDetail", jobId],
    queryFn: () => api.jobDetail(jobId),
    enabled: Boolean(jobId),
  });

  const rankQ = useQuery({
    queryKey: ["screeningRank", jobId, topN],
    queryFn: () => api.screeningRank(jobId, topN),
    enabled: Boolean(jobId),
  });

  useEffect(() => {
    if (rankQ.data && rankQ.data.candidates.length > 0) {
      if (!selectedCandId || !rankQ.data.candidates.some((c) => c.candidate_id === selectedCandId)) {
        setSelectedCandId(rankQ.data.candidates[0].candidate_id);
      }
    }
  }, [rankQ.data, selectedCandId]);

  const candDetailQ = useQuery({
    queryKey: ["candidateDetail", selectedCandId, jobId],
    queryFn: () => api.candidateDetail(selectedCandId, jobId),
    enabled: Boolean(selectedCandId && jobId),
  });

  const uncertainQ = useQuery({
    queryKey: ["uncertainQueue", jobId],
    queryFn: () => api.uncertainQueue(jobId),
    enabled: Boolean(jobId),
  });

  useEffect(() => {
    if (uncertainQ.data && uncertainQ.data.pending.length > 0) {
      if (!selectedUncId || !uncertainQ.data.pending.some((u) => u.candidate_id === selectedUncId)) {
        setSelectedUncId(uncertainQ.data.pending[0].candidate_id);
      }
    } else {
      setSelectedUncId("");
    }
  }, [uncertainQ.data, selectedUncId]);

  const compareQ = useQuery({
    queryKey: ["screeningCompare", jobId, topN],
    queryFn: () => api.compareModels(jobId, topN),
    enabled: Boolean(jobId),
  });

  const feedbackMutation = useMutation({
    mutationFn: api.submitFeedback,
    onSuccess: (entry) => {
      setToastMsg(`Review recorded for Candidate #${entry.candidate_id} (${entry.recruiter_decision}) in audit trail.`);
      setNotes("");
      queryClient.invalidateQueries({ queryKey: ["uncertainQueue", jobId] });
      queryClient.invalidateQueries({ queryKey: ["feedback", jobId] });
    },
  });

  const uncFeedbackMutation = useMutation({
    mutationFn: api.submitFeedback,
    onSuccess: (entry) => {
      setUncToast(`Feedback recorded for Candidate #${entry.candidate_id} as '${entry.recruiter_decision}'. Candidate moved to reviewed list.`);
      setUncNotes("");
      queryClient.invalidateQueries({ queryKey: ["uncertainQueue", jobId] });
      queryClient.invalidateQueries({ queryKey: ["feedback", jobId] });
    },
  });

  if (categoriesQ.isLoading || jobsQ.isLoading) return <PageSkeleton />;

  const categories = ["All Categories", ...(categoriesQ.data ?? [])];
  const jobs = jobsQ.data ?? [];
  const jobDetail = jobDetailQ.data;
  const candidates = rankQ.data?.candidates ?? [];
  const candDetail = candDetailQ.data;
  const uncertain = uncertainQ.data;
  const compare = compareQ.data;

  const rankColumns: ColumnDef<RankedCandidate>[] = [
    { key: "rank", header: "Rank", sortable: true, sortValue: (r) => r.rank, render: (r) => <span className="font-bold">#{r.rank}</span> },
    { key: "id", header: "Candidate ID", sortable: true, sortValue: (r) => r.candidate_id, render: (r) => <span className="font-mono text-xs">{r.candidate_id}</span> },
    {
      key: "sim",
      header: "Semantic Match",
      sortable: true,
      sortValue: (r) => r.similarity_score,
      render: (r) => (
        <div className="flex items-center gap-2 min-w-[120px]">
          <div className="flex-1 h-2 bg-[#E6ECF5] rounded-full overflow-hidden">
            <div
              className="h-full bg-[#2563EB] rounded-full"
              style={{ width: `${Math.min(100, Math.max(0, r.similarity_score * 100))}%` }}
            />
          </div>
          <span className="text-xs font-semibold tabular-nums">{r.similarity_score.toFixed(3)}</span>
        </div>
      ),
    },
    {
      key: "tfidf",
      header: "TF-IDF Score",
      align: "right",
      sortable: true,
      sortValue: (r) => r.tfidf_score ?? 0,
      render: (r) => <span className="tabular-nums text-xs">{r.tfidf_score !== null ? r.tfidf_score.toFixed(3) : "—"}</span>,
    },
    {
      key: "cov",
      header: "Skill Coverage %",
      align: "right",
      sortable: true,
      sortValue: (r) => r.skill_coverage_pct,
      render: (r) => <span className="tabular-nums font-medium">{r.skill_coverage_pct.toFixed(1)}%</span>,
    },
    { key: "exp", header: "Exp (Yrs)", align: "right", sortable: true, sortValue: (r) => r.years_experience, render: (r) => r.years_experience.toFixed(1) },
    { key: "role", header: "Current/Target Role", sortable: true, sortValue: (r) => r.target_position, render: (r) => <span className="text-xs">{r.target_position}</span> },
    {
      key: "matched",
      header: "Verified Skills",
      render: (r) => (
        <span className="text-xs text-[#047857]">
          {r.matching_skills.length > 0 ? r.matching_skills.slice(0, 4).join(", ") + (r.matching_skills.length > 4 ? ` +${r.matching_skills.length - 4}` : "") : "None"}
        </span>
      ),
    },
    {
      key: "missing",
      header: "Skill Deficiencies",
      render: (r) => (
        <span className="text-xs text-[#B91C1C]">
          {r.missing_skills.length > 0 ? r.missing_skills.slice(0, 4).join(", ") + (r.missing_skills.length > 4 ? ` +${r.missing_skills.length - 4}` : "") : "None"}
        </span>
      ),
    },
  ];

  const pendingColumns: ColumnDef<UncertainCandidate>[] = [
    { key: "id", header: "Candidate ID", render: (r) => <span className="font-mono text-xs">{r.candidate_id}</span> },
    { key: "name", header: "Candidate Name / Profile", render: (r) => r.candidate_name },
    { key: "sbert", header: "SBERT Score", align: "right", render: (r) => r.similarity_score.toFixed(3) },
    { key: "tfidf", header: "TF-IDF Score", align: "right", render: (r) => (r.tfidf_score !== null ? r.tfidf_score.toFixed(3) : "—") },
    { key: "reason", header: "Reason Flagged Uncertain", render: (r) => <span className="text-xs">{r.uncertainty_reason}</span> },
    { key: "status", header: "Review Status", render: (r) => <span className="text-xs font-semibold text-[#D97706]">{r.review_status}</span> },
  ];

  const reviewedColumns: ColumnDef<UncertainCandidate>[] = [
    { key: "id", header: "Candidate ID", render: (r) => <span className="font-mono text-xs">{r.candidate_id}</span> },
    { key: "name", header: "Candidate Name / Profile", render: (r) => r.candidate_name },
    { key: "sbert", header: "SBERT Score", align: "right", render: (r) => r.similarity_score.toFixed(3) },
    { key: "status", header: "Recruiter Assessment", render: (r) => <span className="font-semibold text-[#059669]">{r.review_status}</span> },
    { key: "ts", header: "Reviewed At", render: (r) => <span className="text-xs text-[#64748B]">{r.review_timestamp.slice(0, 19).replace("T", " ")}</span> },
    { key: "notes", header: "Review Notes", render: (r) => <span className="text-xs">{r.review_notes || "—"}</span> },
  ];

  const selectedUncCandidate = uncertain?.pending.find((u) => u.candidate_id === selectedUncId);

  return (
    <div>
      <h2 className="text-[1.6rem] font-bold text-[#31527F] mb-1">
        AI Resume Screening &amp; Candidate Ranking
      </h2>
      <p className="text-[#475569] mb-5">
        Semantically rank applicants against job requisitions, diagnose skill gaps, and log audit feedback.
      </p>

      {/* 1. Job Requisition Selector */}
      <SectionHeader title="1. Select Target Job Opening" />
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
        <div>
          <label htmlFor="cat-select" className="block text-xs font-semibold text-[#475569] mb-1">
            Filter Job Openings by Role Category:
          </label>
          <select
            id="cat-select"
            value={category}
            onChange={(e) => setCategory(e.target.value)}
            className="w-full px-3 py-2 border border-[#D3DEEF] rounded-lg bg-white text-sm"
          >
            {categories.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </div>

        <div className="md:col-span-2">
          <label htmlFor="job-select" className="block text-xs font-semibold text-[#475569] mb-1">
            Select Requisition ({jobs.length.toLocaleString()} available):
          </label>
          <select
            id="job-select"
            value={jobId}
            onChange={(e) => setJobId(e.target.value)}
            className="w-full px-3 py-2 border border-[#D3DEEF] rounded-lg bg-white text-sm"
          >
            {jobs.map((j) => (
              <option key={j.job_id} value={j.job_id}>
                ID: {j.job_id} | {j.job_title} ({j.category}) – Req: {j.years_required} yrs
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Collapsible Requisition Details */}
      {jobDetail && (
        <div className="bg-white border border-[#E3EAF4] rounded-xl mb-6 overflow-hidden">
          <button
            type="button"
            onClick={() => setDetailsOpen((v) => !v)}
            className="w-full flex items-center justify-between px-4 py-3 text-sm font-semibold text-[#243B60] bg-[#F8FAFD] hover:bg-[#EDF3FC]"
          >
            <span>Requisition Details &amp; Required Competencies</span>
            {detailsOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
          {detailsOpen && (
            <div className="p-4 space-y-3">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="md:col-span-3 space-y-2 text-sm text-[#334155]">
                  <div>
                    <strong>Job Title:</strong> {jobDetail.job_title} &nbsp;|&nbsp; <strong>Category:</strong> {jobDetail.category}
                  </div>
                  <div>
                    <strong>Description Summary:</strong> {jobDetail.description.slice(0, 350)}...
                  </div>
                </div>
                <div className="flex flex-col gap-2">
                  <div className="bg-[#F8FAFC] border border-[#E2E8F0] rounded-lg p-3">
                    <div className="text-xs text-[#64748B] font-semibold">Experience Required</div>
                    <div className="text-lg font-bold text-[#243B60]">{jobDetail.years_required} Years</div>
                  </div>
                  <div className="bg-[#F8FAFC] border border-[#E2E8F0] rounded-lg p-3">
                    <div className="text-xs text-[#64748B] font-semibold">Total Competencies</div>
                    <div className="text-lg font-bold text-[#243B60]">{jobDetail.skills.length}</div>
                  </div>
                </div>
              </div>
              <div>
                <div className="text-xs font-bold text-[#475569] mb-1.5">Required Skills Taxonomy:</div>
                {jobDetail.skills.length > 0 ? (
                  <div>
                    {jobDetail.skills.map((s) => (
                      <SkillBadge key={s} label={s} variant="pill" />
                    ))}
                  </div>
                ) : (
                  <span className="text-xs text-[#64748B]">No explicit skill keywords tagged in requisition.</span>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* 2. Ranked Applicants */}
      <SectionHeader title="2. Ranked Applicants (Sentence-BERT Semantic Matching)" />
      <div className="flex items-center justify-between flex-wrap gap-4 mb-3">
        <div className="flex items-center gap-3">
          <label htmlFor="topn-slider" className="text-sm font-medium text-[#334155]">
            Number of Top Candidates to Display: <strong>{topN}</strong>
          </label>
          <input
            id="topn-slider"
            type="range"
            min={5}
            max={30}
            step={5}
            value={topN}
            onChange={(e) => setTopN(Number(e.target.value))}
            className="accent-[#2563EB]"
          />
        </div>
        {rankQ.data && (
          <span className="text-xs text-[#64748B]">
            Screened across all {rankQ.data.total_pool_size.toLocaleString()} indexed candidates
          </span>
        )}
      </div>

      <DataTable
        data={candidates}
        columns={rankColumns}
        rowKey={(r) => r.candidate_id}
        selectedKey={selectedCandId}
        onRowClick={(r) => setSelectedCandId(r.candidate_id)}
      />
      <p className="text-xs text-[#64748B] mt-2">
        Note: Semantic Match represents high-dimensional textual relevance between the resume and job requisition, not a statistical guarantee of hiring.
      </p>

      {/* 3. Candidate Profile Inspector */}
      <SectionHeader title="3. Candidate Profile Inspector & Skill-Gap Diagnostics" />
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-4">
        <div className="space-y-3">
          <div>
            <label htmlFor="cand-select" className="block text-xs font-semibold text-[#475569] mb-1">
              Inspect Candidate by Rank:
            </label>
            <select
              id="cand-select"
              value={selectedCandId}
              onChange={(e) => setSelectedCandId(e.target.value)}
              className="w-full px-3 py-2 border border-[#D3DEEF] rounded-lg bg-white text-sm"
            >
              {candidates.map((c) => (
                <option key={c.candidate_id} value={c.candidate_id}>
                  Rank {c.rank}: Candidate #{c.candidate_id} (Match: {c.similarity_score.toFixed(3)})
                </option>
              ))}
            </select>
          </div>

          {candDetail && (
            <MetricCard title={`Candidate #${candDetail.candidate_id}`} value="">
              <div className="text-base font-bold text-[#1E293B] mb-2">{candDetail.target_position}</div>
              <div className="text-sm text-[#334155] space-y-1">
                <div><strong>Experience:</strong> {candDetail.years_experience} Years</div>
                <div><strong>Semantic Match Score:</strong> {candDetail.similarity_score.toFixed(3)}</div>
                <div><strong>Skill Coverage:</strong> {candDetail.skill_coverage_pct.toFixed(1)}%</div>
                <div><strong>Education:</strong> {candDetail.education_text}</div>
              </div>
            </MetricCard>
          )}
        </div>

        <div className="lg:col-span-2 bg-white border border-[#E3EAF4] rounded-xl p-5">
          {candDetail && (
            <div className="space-y-4">
              <div className="font-bold text-sm text-[#243B60]">Skill Alignment Breakdown:</div>
              <div className="grid grid-cols-3 gap-3">
                <div className="bg-[#F8FAFC] border border-[#E2E8F0] rounded-lg p-3">
                  <div className="text-xs text-[#64748B]">Matching Skills</div>
                  <div className="text-xl font-bold text-[#059669]">{candDetail.matching_skills.length}</div>
                </div>
                <div className="bg-[#F8FAFC] border border-[#E2E8F0] rounded-lg p-3">
                  <div className="text-xs text-[#64748B]">Missing Skills</div>
                  <div className="text-xl font-bold text-[#DC2626]">{candDetail.missing_skills.length}</div>
                </div>
                <div className="bg-[#F8FAFC] border border-[#E2E8F0] rounded-lg p-3">
                  <div className="text-xs text-[#64748B]">Skill Coverage</div>
                  <div className="text-xl font-bold text-[#243B60]">{candDetail.skill_coverage_pct.toFixed(1)}%</div>
                </div>
              </div>

              <div>
                <div className="text-sm font-semibold text-[#047857] mb-1.5">Matching Competencies:</div>
                {candDetail.matching_skills.length > 0 ? (
                  candDetail.matching_skills.map((s) => <SkillBadge key={s} label={s} variant="match" />)
                ) : (
                  <p className="text-xs text-[#64748B]">None of the job&apos;s tagged skills were explicitly detected in this resume.</p>
                )}
              </div>

              <div>
                <div className="text-sm font-semibold text-[#B91C1C] mb-1.5">Skill Deficiencies (Potential Gaps):</div>
                {candDetail.missing_skills.length > 0 ? (
                  candDetail.missing_skills.map((s) => <SkillBadge key={s} label={s} variant="missing" />)
                ) : (
                  <p className="text-xs text-[#059669]">100% skill coverage; candidate mentions all required competencies.</p>
                )}
              </div>

              <p className="text-xs text-[#64748B] pt-2 border-t border-[#F0F2F7]">
                Responsible AI Note: An unmentioned skill in a resume does not definitively prove absence of actual ability. It highlights areas for recruiter interview probing.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Expandable Resume Text */}
      {candDetail && (
        <div className="bg-white border border-[#E3EAF4] rounded-xl mb-6 overflow-hidden">
          <button
            type="button"
            onClick={() => setResumeOpen((v) => !v)}
            className="w-full flex items-center justify-between px-4 py-3 text-sm font-semibold text-[#243B60] bg-[#F8FAFD] hover:bg-[#EDF3FC]"
          >
            <span>Full Extracted Resume Text – Candidate #{candDetail.candidate_id}</span>
            {resumeOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
          {resumeOpen && (
            <div className="p-4">
              <textarea
                readOnly
                value={candDetail.resume_text}
                rows={8}
                className="w-full p-3 border border-[#E2E8F0] rounded-lg bg-[#F8FAFC] text-xs font-mono text-[#334155]"
              />
            </div>
          )}
        </div>
      )}

      {/* 4. Human Recruiter Feedback Form */}
      <SectionHeader title="4. Recruiter Match Review (Audit Log)" />
      <InfoCallout>
        Recruiter decisions are recorded in a separate audit log to track review history. Feedback is never used to automatically retrain screening models without verified human validation.
      </InfoCallout>

      {candDetail && jobDetail && (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            feedbackMutation.mutate({
              candidate_id: candDetail.candidate_id,
              job_id: jobDetail.job_id,
              job_title: jobDetail.job_title,
              decision,
              relevance_score: candDetail.similarity_score,
              notes,
            });
          }}
          className="bg-white border border-[#E3EAF4] rounded-xl p-4 mb-6 space-y-4"
        >
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
            <div>
              <label htmlFor="fb-decision" className="block text-xs font-semibold text-[#475569] mb-1">
                Recruiter Assessment:
              </label>
              <select
                id="fb-decision"
                value={decision}
                onChange={(e) => setDecision(e.target.value as CanonicalDecision)}
                className="w-full px-3 py-2 border border-[#D3DEEF] rounded-lg bg-white text-sm"
              >
                <option value="Suitable">Suitable</option>
                <option value="Not Suitable">Not Suitable</option>
                <option value="Needs Further Review">Needs Further Review</option>
              </select>
            </div>
            <div className="text-sm text-[#334155]">
              <div><strong>Target:</strong> #{candDetail.candidate_id}</div>
              <div><strong>Score:</strong> {candDetail.similarity_score.toFixed(3)}</div>
            </div>
            <div className="md:col-span-2">
              <label htmlFor="fb-notes" className="block text-xs font-semibold text-[#475569] mb-1">
                Review Notes / Interview Instructions:
              </label>
              <input
                id="fb-notes"
                type="text"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="e.g. Verify deep learning project experience in phone screen"
                className="w-full px-3 py-2 border border-[#D3DEEF] rounded-lg text-sm"
              />
            </div>
          </div>
          <div className="flex items-center gap-4">
            <button
              type="submit"
              disabled={feedbackMutation.isPending}
              className="px-4 py-2 bg-[#2563EB] text-white rounded-lg text-sm font-semibold hover:bg-blue-700 disabled:opacity-50"
            >
              {feedbackMutation.isPending ? "Saving..." : "Record Recruiter Review"}
            </button>
            {toastMsg && <span className="text-sm font-medium text-[#059669]">{toastMsg}</span>}
          </div>
        </form>
      )}

      {/* 5. Active Learning – Uncertainty Review Queue */}
      <SectionHeader title="5. Active Learning – Uncertainty Review Queue" />
      <p className="text-sm text-[#475569] mb-4">
        Human-in-the-Loop review queue prioritizing candidate resumes where the AI screening decision is ambiguous. Flags candidate profiles adjacent to the decision threshold (0.45) or showing significant divergence between neural semantic and keyword match scores.
      </p>

      {uncertain && (
        <div className="space-y-4 mb-6">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <MetricCard title="Candidates Awaiting Review" value={uncertain.total_awaiting_review} />
            <MetricCard title="Candidates Already Reviewed" value={uncertain.total_reviewed} />
            <MetricCard
              title="Decision Boundary Threshold"
              value={`${uncertain.decision_threshold.toFixed(2)} (±${uncertain.uncertainty_margin.toFixed(2)})`}
            />
          </div>

          <div className="flex gap-2 border-b border-[#E2E8F0]">
            <button
              type="button"
              onClick={() => setQueueTab("pending")}
              className={`px-4 py-2 text-sm font-semibold border-b-2 transition-colors ${
                queueTab === "pending"
                  ? "border-[#2563EB] text-[#2563EB]"
                  : "border-transparent text-[#64748B] hover:text-[#243B60]"
              }`}
            >
              Priority Review Queue (Pending)
            </button>
            <button
              type="button"
              onClick={() => setQueueTab("reviewed")}
              className={`px-4 py-2 text-sm font-semibold border-b-2 transition-colors ${
                queueTab === "reviewed"
                  ? "border-[#2563EB] text-[#2563EB]"
                  : "border-transparent text-[#64748B] hover:text-[#243B60]"
              }`}
            >
              Reviewed Candidates History ({uncertain.total_reviewed})
            </button>
          </div>

          {queueTab === "pending" ? (
            uncertain.pending.length > 0 ? (
              <div className="space-y-4">
                <div className="text-sm font-semibold text-[#243B60]">
                  Top {uncertain.pending.length} Uncertain Cases (Ranked by Ambiguity First):
                </div>
                <DataTable
                  data={uncertain.pending}
                  columns={pendingColumns}
                  rowKey={(r) => r.candidate_id}
                  selectedKey={selectedUncId}
                  onRowClick={(r) => setSelectedUncId(r.candidate_id)}
                />

                {selectedUncCandidate && jobDetail && (
                  <form
                    onSubmit={(e) => {
                      e.preventDefault();
                      uncFeedbackMutation.mutate({
                        candidate_id: selectedUncCandidate.candidate_id,
                        job_id: jobDetail.job_id,
                        job_title: jobDetail.job_title,
                        decision: uncDecision,
                        relevance_score: selectedUncCandidate.similarity_score,
                        notes: uncNotes,
                      });
                    }}
                    className="bg-white border border-[#E3EAF4] rounded-xl p-4 space-y-4"
                  >
                    <div className="text-sm font-bold text-[#243B60]">Human-in-the-Loop Feedback Submission:</div>
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
                      <div>
                        <label htmlFor="unc-cand-select" className="block text-xs font-semibold text-[#475569] mb-1">
                          Select Uncertain Candidate:
                        </label>
                        <select
                          id="unc-cand-select"
                          value={selectedUncId}
                          onChange={(e) => setSelectedUncId(e.target.value)}
                          className="w-full px-3 py-2 border border-[#D3DEEF] rounded-lg bg-white text-sm"
                        >
                          {uncertain.pending.map((u) => (
                            <option key={u.candidate_id} value={u.candidate_id}>
                              {u.candidate_id} | SBERT: {u.similarity_score.toFixed(3)} | {u.target_position}
                            </option>
                          ))}
                        </select>
                      </div>
                      <div>
                        <label htmlFor="unc-decision" className="block text-xs font-semibold text-[#475569] mb-1">
                          Assessment:
                        </label>
                        <select
                          id="unc-decision"
                          value={uncDecision}
                          onChange={(e) => setUncDecision(e.target.value as CanonicalDecision)}
                          className="w-full px-3 py-2 border border-[#D3DEEF] rounded-lg bg-white text-sm"
                        >
                          <option value="Suitable">Suitable</option>
                          <option value="Not Suitable">Not Suitable</option>
                          <option value="Needs Further Review">Needs Further Review</option>
                        </select>
                      </div>
                      <div className="md:col-span-2">
                        <label htmlFor="unc-notes" className="block text-xs font-semibold text-[#475569] mb-1">
                          Evaluation Notes:
                        </label>
                        <input
                          id="unc-notes"
                          type="text"
                          value={uncNotes}
                          onChange={(e) => setUncNotes(e.target.value)}
                          placeholder="e.g. Verify depth of hands-on skills during technical screen"
                          className="w-full px-3 py-2 border border-[#D3DEEF] rounded-lg text-sm"
                        />
                      </div>
                    </div>
                    <div className="flex items-center gap-4">
                      <button
                        type="submit"
                        disabled={uncFeedbackMutation.isPending}
                        className="px-4 py-2 bg-[#2563EB] text-white rounded-lg text-sm font-semibold hover:bg-blue-700 disabled:opacity-50"
                      >
                        {uncFeedbackMutation.isPending ? "Recording..." : "Record Active Learning Feedback"}
                      </button>
                      {uncToast && <span className="text-sm font-medium text-[#059669]">{uncToast}</span>}
                    </div>
                  </form>
                )}
              </div>
            ) : (
              <InfoCallout>
                No pending candidates in the uncertainty review queue for this job opening. All identified cases have been reviewed.
              </InfoCallout>
            )
          ) : uncertain.reviewed.length > 0 ? (
            <DataTable
              data={uncertain.reviewed}
              columns={reviewedColumns}
              rowKey={(r) => r.candidate_id}
            />
          ) : (
            <InfoCallout>No candidates have been reviewed yet for this requisition.</InfoCallout>
          )}
        </div>
      )}

      {/* 6. TF-IDF vs Sentence-BERT Comparison */}
      <SectionHeader title="6. Method Comparison: TF-IDF Keyword Baseline vs. Sentence-BERT Semantic Matching" />
      {compare && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <MetricCard title="Spearman Rank Correlation" value={compare.spearman_rank_correlation.toFixed(3)} />
            <MetricCard title="Pearson Score Correlation" value={compare.pearson_score_correlation.toFixed(3)} />
            <MetricCard
              title={`Top-${compare.top_k} Candidate Overlap`}
              value={`${compare.top_k_overlap_count} / ${compare.top_k}`}
            />
            <MetricCard title="Top-K Agreement" value={`${compare.top_k_overlap_percentage.toFixed(1)}%`} />
          </div>

          <div className="bg-white border border-[#E5EBF4] rounded-xl p-4">
            <h3 className="text-sm font-bold text-[#243B60] mb-2">
              Score Comparison: Keyword TF-IDF vs. Semantic Sentence-BERT (N={compare.scatter_sample.length} Sample)
            </h3>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                  <XAxis
                    type="number"
                    dataKey="tfidf_score"
                    name="TF-IDF Score"
                    label={{ value: "TF-IDF Lexical Cosine Similarity", position: "insideBottom", offset: -10, fontSize: 12 }}
                    tick={{ fontSize: 11, fill: "#475569" }}
                  />
                  <YAxis
                    type="number"
                    dataKey="similarity_score"
                    name="SBERT Score"
                    label={{ value: "Sentence-BERT Similarity", angle: -90, position: "insideLeft", fontSize: 12 }}
                    tick={{ fontSize: 11, fill: "#475569" }}
                  />
                  <Tooltip
                    cursor={{ strokeDasharray: "3 3" }}
                    content={({ payload }) => {
                      if (!payload || payload.length === 0) return null;
                      const pt = payload[0].payload as {
                        candidate_id: string;
                        tfidf_score: number;
                        similarity_score: number;
                        years_experience: number;
                      };
                      return (
                        <div className="bg-white border border-[#CBD5E1] rounded-lg p-2.5 text-xs shadow">
                          <div className="font-bold text-[#243B60]">#{pt.candidate_id}</div>
                          <div>SBERT: {pt.similarity_score.toFixed(3)}</div>
                          <div>TF-IDF: {pt.tfidf_score.toFixed(3)}</div>
                          <div>Exp: {pt.years_experience} yrs</div>
                        </div>
                      );
                    }}
                  />
                  <Scatter data={compare.scatter_sample} fill="#2563EB" fillOpacity={0.65} />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="bg-white border border-[#E3EAF4] rounded-xl overflow-hidden">
            <button
              type="button"
              onClick={() => setObsOpen((v) => !v)}
              className="w-full flex items-center justify-between px-4 py-3 text-sm font-semibold text-[#243B60] bg-[#F8FAFD] hover:bg-[#EDF3FC]"
            >
              <span>Methodological Comparison &amp; Observations</span>
              {obsOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>
            {obsOpen && (
              <ul className="p-4 list-disc pl-8 space-y-1.5 text-sm text-[#334155]">
                {compare.observations.map((obs) => (
                  <li key={obs}>{obs}</li>
                ))}
              </ul>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
