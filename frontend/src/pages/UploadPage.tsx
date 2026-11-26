import { useCallback, useEffect, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useDropzone } from "react-dropzone";
import { ChevronDown, ChevronUp, FileUp, Sparkles } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api/client";
import type { UploadScreenResponse } from "../api/types";
import { AlertCallout, InfoCallout } from "../components/Callout";
import { PageSkeleton } from "../components/ErrorBoundary";
import { MetricCard } from "../components/MetricCard";
import { SectionHeader } from "../components/SectionHeader";
import { SkillBadge } from "../components/SkillBadge";

export function UploadPage() {
  const [jobId, setJobId] = useState<string>("");
  const [detailsOpen, setDetailsOpen] = useState<boolean>(false);
  const [textOpen, setTextOpen] = useState<boolean>(false);
  const [result, setResult] = useState<UploadScreenResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string>("");

  const jobsQ = useQuery({ queryKey: ["jobs", "All Categories"], queryFn: () => api.jobs() });

  useEffect(() => {
    if (jobsQ.data && jobsQ.data.length > 0 && !jobId) {
      setJobId(jobsQ.data[0].job_id);
    }
  }, [jobsQ.data, jobId]);

  const jobDetailQ = useQuery({
    queryKey: ["jobDetail", jobId],
    queryFn: () => api.jobDetail(jobId),
    enabled: Boolean(jobId),
  });

  const uploadMut = useMutation({
    mutationFn: ({ file, targetJobId }: { file: File; targetJobId: string }) =>
      api.uploadScreen(file, targetJobId),
    onSuccess: (data) => {
      setErrorMsg("");
      setResult(data);
    },
    onError: (err) => {
      setResult(null);
      setErrorMsg(err instanceof Error ? err.message : "Resume processing failed.");
    },
  });

  const demoMut = useMutation({
    mutationFn: (targetJobId: string) => api.uploadDemo(targetJobId),
    onSuccess: (data) => {
      setErrorMsg("");
      setResult(data);
    },
    onError: (err) => {
      setResult(null);
      setErrorMsg(err instanceof Error ? err.message : "Demo screening failed.");
    },
  });

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0 && jobId) {
        uploadMut.mutate({ file: acceptedFiles[0], targetJobId: jobId });
      }
    },
    [jobId, uploadMut]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    multiple: false,
    // Allow user to drop any file so backend/frontend can show friendly error for unsupported types
  });

  if (jobsQ.isLoading) return <PageSkeleton />;

  const jobs = jobsQ.data ?? [];
  const jobDetail = jobDetailQ.data;
  const isWorking = uploadMut.isPending || demoMut.isPending;

  const pieData = result
    ? [
        { name: "Matching Skills", value: result.skill_gap.matching_skills.length, color: "#059669" },
        { name: "Missing Skills", value: result.skill_gap.missing_skills.length, color: "#DC2626" },
      ]
    : [];

  // Find closest histogram bin label for dashed marker
  const markerLabel = result
    ? result.benchmark.histogram.reduce((closest, bin) => {
        const mid = (bin.bin_start + bin.bin_end) / 2;
        const cMid = (closest.bin_start + closest.bin_end) / 2;
        return Math.abs(mid - result.similarity_score) < Math.abs(cMid - result.similarity_score)
          ? bin
          : closest;
      }, result.benchmark.histogram[0])?.label
    : undefined;

  return (
    <div>
      <h2 className="text-[1.6rem] font-bold text-[#31527F] mb-1">
        Upload Resume &amp; Live Screening
      </h2>
      <p className="text-[#475569] mb-5">
        Ingest an external candidate&apos;s resume (PDF or DOCX) for instant semantic evaluation and skill gap diagnostics.
      </p>

      {/* 1. Target Job Selector */}
      <SectionHeader title="1. Select Target Job Opening" />
      <div className="mb-3">
        <label htmlFor="upload-job-select" className="block text-xs font-semibold text-[#475569] mb-1">
          Select Target Position:
        </label>
        <select
          id="upload-job-select"
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

      {jobDetail && (
        <div className="bg-white border border-[#E3EAF4] rounded-xl mb-6 overflow-hidden">
          <button
            type="button"
            onClick={() => setDetailsOpen((v) => !v)}
            className="w-full flex items-center justify-between px-4 py-3 text-sm font-semibold text-[#243B60] bg-[#F8FAFD] hover:bg-[#EDF3FC]"
          >
            <span>Target Requisition Details</span>
            {detailsOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
          {detailsOpen && (
            <div className="p-4 space-y-2 text-sm text-[#334155]">
              <div>
                <strong>Title:</strong> {jobDetail.job_title} &nbsp;|&nbsp; <strong>Category:</strong> {jobDetail.category}
              </div>
              <div>
                <strong>Description:</strong> {jobDetail.description.slice(0, 300)}...
              </div>
              <div>
                <strong>Required Experience:</strong> {jobDetail.years_required} Years
              </div>
              <div>
                <strong>Required Skills:</strong>{" "}
                {jobDetail.skills.map((s) => (
                  <SkillBadge key={s} label={s} variant="pill" />
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* 2. File Uploader */}
      <SectionHeader title="2. Ingest Resume Document" />
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors bg-white mb-4 ${
          isDragActive ? "border-[#2563EB] bg-[#EFF5FD]" : "border-[#CBD5E1] hover:border-[#2563EB]"
        }`}
      >
        <input {...getInputProps()} aria-label="Upload resume file" />
        <FileUp className="w-10 h-10 text-[#2563EB] mx-auto mb-3" />
        <p className="text-sm font-semibold text-[#243B60]">
          {isDragActive
            ? "Drop the resume file here..."
            : "Drag & drop candidate resume (.pdf or .docx), or click to browse"}
        </p>
        <p className="text-xs text-[#64748B] mt-1">
          Maximum 10 MB. Password-protected PDFs and scanned image PDFs without text layers are detected automatically.
        </p>
      </div>

      {isWorking && (
        <InfoCallout>Extracting document text and calculating semantic profile fit...</InfoCallout>
      )}

      {errorMsg && (
        <AlertCallout>
          <strong>Extraction Error:</strong> {errorMsg}
        </AlertCallout>
      )}

      <div className="flex items-center justify-between flex-wrap gap-3 py-3 border-t border-b border-[#E2E8F0] mb-6">
        <span className="text-sm text-[#475569]">
          <strong>Quick Demo Mode:</strong> Test live screening using a pre-configured senior engineer profile:
        </span>
        <button
          type="button"
          disabled={!jobId || isWorking}
          onClick={() => jobId && demoMut.mutate(jobId)}
          className="inline-flex items-center gap-2 px-4 py-2 bg-white border border-[#D3DEEF] rounded-lg text-sm font-semibold text-[#243B60] hover:border-[#2563EB] hover:text-[#2563EB] disabled:opacity-50"
        >
          <Sparkles className="w-4 h-4" />
          Load Pre-Configured Sample Resume for Demonstration
        </button>
      </div>

      {/* 3. Candidate Evaluation Results */}
      {result && (
        <div className="space-y-6">
          <div>
            <SectionHeader title="3. Candidate Evaluation Results" />
            <span className="inline-block px-3.5 py-1.5 rounded-full text-xs font-semibold bg-[#2563EB] text-white mb-4">
              Live Uploaded Candidate: {result.filename}
            </span>

            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
              <MetricCard
                title="Semantic Match Score"
                value={result.similarity_score.toFixed(3)}
                valueClassName="text-[#2563EB]"
                subtitle="Contextual similarity to requisition"
              />
              <MetricCard
                title="Skill Coverage"
                value={`${result.skill_gap.coverage_percentage.toFixed(1)}%`}
                subtitle={`${result.skill_gap.match_count} of ${result.skill_gap.required_count} skills matched`}
              />
              <MetricCard
                title="Extracted Experience"
                value={`${result.extracted_years_experience.toFixed(1)} Yrs`}
                subtitle={`Required: ${jobDetail?.years_required ?? 0} Yrs`}
              />
              <MetricCard
                title="Highest Degree"
                value=""
                subtitle="Credential detection"
              >
                <div className="text-[1.15rem] font-extrabold text-[#293F60] mt-1">
                  {result.extracted_education.length > 0
                    ? result.extracted_education.join(", ")
                    : "Self-Stated"}
                </div>
              </MetricCard>
            </div>
          </div>

          {/* Skill-Gap Diagnostics */}
          <div>
            <SectionHeader title="Skill-Gap Diagnostics" />
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div className="bg-white border border-[#E3EAF4] rounded-xl p-5 space-y-4">
                <div>
                  <div className="text-sm font-semibold text-[#047857] mb-2">
                    Matching Competencies Detected in Resume:
                  </div>
                  {result.skill_gap.matching_skills.length > 0 ? (
                    result.skill_gap.matching_skills.map((s) => (
                      <SkillBadge key={s} label={s} variant="match" />
                    ))
                  ) : (
                    <p className="text-xs text-[#64748B]">No direct overlap between resume skills and job requirements.</p>
                  )}
                </div>

                <div>
                  <div className="text-sm font-semibold text-[#B91C1C] mb-2">
                    Skill Deficiencies (Missing from Resume):
                  </div>
                  {result.skill_gap.missing_skills.length > 0 ? (
                    result.skill_gap.missing_skills.map((s) => (
                      <SkillBadge key={s} label={s} variant="missing" />
                    ))
                  ) : (
                    <p className="text-xs text-[#059669]">Zero identified deficiencies; candidate covers all specified job skills.</p>
                  )}
                </div>
              </div>

              <div className="bg-white border border-[#E5EBF4] rounded-xl p-4">
                <h3 className="text-sm font-bold text-[#243B60] mb-2">Skill Alignment Ratio</h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={pieData}
                        dataKey="value"
                        nameKey="name"
                        innerRadius={55}
                        outerRadius={90}
                        paddingAngle={2}
                        label
                      >
                        {pieData.map((entry) => (
                          <Cell key={entry.name} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>

          {/* Benchmark Placement */}
          <div>
            <SectionHeader title="Candidate Benchmark Placement" />
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-4">
              <MetricCard
                title="Pool Placement Rank"
                value={`#${result.benchmark.placement_rank.toLocaleString()} of ${result.benchmark.total_candidates.toLocaleString()}`}
              />
              <MetricCard
                title="Relevance Percentile"
                value={`${result.benchmark.percentile.toFixed(1)}%`}
              />
              <MetricCard
                title="Pool Median Score"
                value={result.benchmark.pool_median.toFixed(3)}
              />
            </div>

            <div className="bg-white border border-[#E5EBF4] rounded-xl p-4">
              <h3 className="text-sm font-bold text-[#243B60] mb-2">
                Relevance Score Distribution for Job &apos;{jobDetail?.job_title ?? jobId}&apos;
              </h3>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={result.benchmark.histogram} margin={{ top: 18, right: 20, left: 0, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                    <XAxis dataKey="label" tick={{ fontSize: 10, fill: "#475569" }} interval={4} />
                    <YAxis tick={{ fontSize: 11, fill: "#475569" }} />
                    <Tooltip />
                    <Bar dataKey="count" name="Candidates" fill="#94A3B8" radius={[3, 3, 0, 0]} />
                    {markerLabel && (
                      <ReferenceLine
                        x={markerLabel}
                        stroke="#2563EB"
                        strokeWidth={3}
                        strokeDasharray="5 5"
                        label={{
                          value: `Uploaded (${result.similarity_score.toFixed(3)})`,
                          position: "top",
                          fill: "#2563EB",
                          fontSize: 12,
                          fontWeight: 700,
                        }}
                      />
                    )}
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Extracted Document Text */}
          <div className="bg-white border border-[#E3EAF4] rounded-xl overflow-hidden">
            <button
              type="button"
              onClick={() => setTextOpen((v) => !v)}
              className="w-full flex items-center justify-between px-4 py-3 text-sm font-semibold text-[#243B60] bg-[#F8FAFD] hover:bg-[#EDF3FC]"
            >
              <span>View Sanitized Extracted Document Text</span>
              {textOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>
            {textOpen && (
              <div className="p-4">
                <textarea
                  readOnly
                  value={result.cleaned_text}
                  rows={8}
                  className="w-full p-3 border border-[#E2E8F0] rounded-lg bg-[#F8FAFC] text-xs font-mono text-[#334155]"
                />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
