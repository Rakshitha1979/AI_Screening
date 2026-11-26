import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { PageSkeleton } from "../components/ErrorBoundary";
import { MetricCard } from "../components/MetricCard";
import { SectionHeader } from "../components/SectionHeader";

export function OverviewPage() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["overview"],
    queryFn: api.overview,
  });

  if (isLoading) return <PageSkeleton />;
  if (error || !data) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-xl p-5 text-red-700 text-sm">
        Failed to load project overview: {error instanceof Error ? error.message : "Unknown error"}
      </div>
    );
  }

  const { counts, offers } = data;

  return (
    <div>
      <h1 className="text-[1.95rem] font-extrabold text-[#243B60] tracking-[-0.7px] mb-1">
        AI-Assisted Candidate Screening Pipeline
      </h1>
      <p className="text-[1.08rem] text-[#475569] mb-6">
        Integrated Resume Matching &amp; Recruitment Analytics
      </p>

      {/* 4 KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <MetricCard
          title="Candidate Resumes"
          value={counts.resumes.toLocaleString()}
          subtitle="Indexed candidate profiles"
        />
        <MetricCard
          title="Job Descriptions"
          value={counts.jobs.toLocaleString()}
          subtitle="Active & historical openings"
        />
        <MetricCard
          title="Total Applications"
          value={counts.applications.toLocaleString()}
          subtitle="Recruitment pipeline records"
        />
        <MetricCard
          title="Confirmed Offers"
          value={counts.offers.toLocaleString()}
          subtitle="Formal job offers recorded"
        />
      </div>

      {/* Offer Decision Summary */}
      <SectionHeader title="Offer Decision Summary" />
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 items-stretch">
        <MetricCard title="Offer Outcomes" value="" className="lg:col-span-1 flex flex-col justify-between">
          <div className="space-y-2 text-[1.02rem] mt-1">
            <div className="flex items-center justify-between">
              <span className="text-[#059669] font-bold">● Accepted:</span>
              <span>
                <strong>{offers.accepted.toLocaleString()}</strong> ({offers.accepted_pct.toFixed(1)}%)
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[#DC2626] font-bold">● Declined:</span>
              <span>
                <strong>{offers.declined.toLocaleString()}</strong> ({offers.declined_pct.toFixed(1)}%)
              </span>
            </div>
          </div>
          <div className="text-[0.82rem] text-[#78869A] mt-4 pt-3 border-t border-[#F0F2F7]">
            Total Verified Offers: <strong>{offers.total.toLocaleString()}</strong>
          </div>
        </MetricCard>

        <div className="lg:col-span-2 bg-white border border-[#E3EAF4] rounded-[13px] p-5 shadow-[0_3px_12px_rgba(45,70,110,0.055)] flex flex-col justify-between">
          <div className="flex items-center justify-between mb-3">
            <span className="font-bold text-[#243B60] text-sm">Offer Distribution</span>
            <div className="flex items-center gap-4 text-xs text-[#475569]">
              <span className="inline-flex items-center gap-1.5">
                <span className="w-3 h-3 rounded-sm bg-[#10B981] inline-block" /> Accepted
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span className="w-3 h-3 rounded-sm bg-[#EF4444] inline-block" /> Declined
              </span>
            </div>
          </div>

          <div
            className="w-full h-14 rounded-xl overflow-hidden flex border border-[#E2E8F0] my-2"
            role="img"
            aria-label={`Accepted ${offers.accepted} (${offers.accepted_pct}%), Declined ${offers.declined} (${offers.declined_pct}%)`}
          >
            <div
              className="bg-[#10B981] text-white font-semibold text-xs sm:text-sm flex items-center justify-center border-r-2 border-white transition-all"
              style={{ width: `${offers.accepted_pct}%` }}
              title={`Accepted Offers: ${offers.accepted.toLocaleString()} (${offers.accepted_pct.toFixed(1)}%)`}
            >
              {offers.accepted.toLocaleString()} &nbsp;|&nbsp; {offers.accepted_pct.toFixed(1)}%
            </div>
            <div
              className="bg-[#EF4444] text-white font-semibold text-xs sm:text-sm flex items-center justify-center transition-all"
              style={{ width: `${offers.declined_pct}%` }}
              title={`Declined Offers: ${offers.declined.toLocaleString()} (${offers.declined_pct.toFixed(1)}%)`}
            >
              {offers.declined.toLocaleString()} &nbsp;|&nbsp; {offers.declined_pct.toFixed(1)}%
            </div>
          </div>

          <div className="text-xs text-[#64748B] mt-2">
            Proportional breakdown across all {offers.total.toLocaleString()} confirmed offers in the recruitment dataset.
          </div>
        </div>
      </div>

      {/* Core Platform Modules */}
      <SectionHeader title="Core Platform Modules" />
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <MetricCard
          title="Module 1: AI Resume Screening"
          value=""
          leftAccentColor="#2563EB"
          className="h-full"
        >
          <div className="text-[0.94rem] text-[#1E293B] leading-relaxed">
            <strong>Key Capabilities:</strong>
            <ul className="list-disc pl-5 mt-2 space-y-1.5">
              <li>
                <strong>Candidate Ranking:</strong> Ranks candidates by relevance to the selected job.
              </li>
              <li>
                <strong>Semantic Matching:</strong> Compares skills, experience and job requirements.
              </li>
              <li>
                <strong>Skill-Gap Analysis:</strong> Identifies matching and missing skills.
              </li>
              <li>
                <strong>Resume Upload:</strong> Supports PDF and DOCX resumes for screening.
              </li>
            </ul>
          </div>
        </MetricCard>

        <MetricCard
          title="Module 2: Recruitment Analytics & Prediction"
          value=""
          leftAccentColor="#0D9488"
          className="h-full"
        >
          <div className="text-[0.94rem] text-[#1E293B] leading-relaxed">
            <strong>Key Capabilities:</strong>
            <ul className="list-disc pl-5 mt-2 space-y-1.5">
              <li>
                <strong>Recruitment Funnel:</strong> Shows candidate counts across recruitment stages.
              </li>
              <li>
                <strong>Sourcing Trends:</strong> Compares application volume and offer outcomes by source.
              </li>
              <li>
                <strong>Interview Outcomes:</strong> Summarizes pass, fail and hold results.
              </li>
              <li>
                <strong>Offer Prediction:</strong> Estimates offer acceptance likelihood using relevant recruitment data.
              </li>
            </ul>
          </div>
        </MetricCard>
      </div>

      {/* System Workflow (Preserved Exactly) */}
      <SectionHeader title="System Workflow" />
      <div className="flex flex-col gap-3">
        <div className="bg-[#F8FAFC] border border-[#E2E8F0] rounded-lg px-5 py-3.5 flex items-center justify-between flex-wrap gap-2">
          <span className="font-semibold text-[#2563EB]">AI Resume Screening Flow:</span>
          <span className="text-[#334155] font-semibold">Resume + Job Description</span>
          <span className="text-[#64748B]">➔</span>
          <span className="text-[#334155] font-semibold">AI Screening</span>
          <span className="text-[#64748B]">➔</span>
          <span className="text-[#334155] font-semibold">Candidate Ranking</span>
        </div>

        <div className="bg-[#F8FAFC] border border-[#E2E8F0] rounded-lg px-5 py-3.5 flex items-center justify-between flex-wrap gap-2">
          <span className="font-semibold text-[#0D9488]">Recruitment Analytics Flow:</span>
          <span className="text-[#334155] font-semibold">Recruitment Data</span>
          <span className="text-[#64748B]">➔</span>
          <span className="text-[#334155] font-semibold">Analytics</span>
          <span className="text-[#64748B]">➔</span>
          <span className="text-[#334155] font-semibold">Offer Prediction</span>
        </div>
      </div>
    </div>
  );
}
