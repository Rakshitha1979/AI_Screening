import { useQuery } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api/client";
import type { SourceStat } from "../api/types";
import { AlertCallout } from "../components/Callout";
import { DataTable, type ColumnDef } from "../components/DataTable";
import { PageSkeleton } from "../components/ErrorBoundary";
import { MetricCard } from "../components/MetricCard";
import { SectionHeader } from "../components/SectionHeader";

const FUNNEL_COLORS = ["#64B5F6", "#42A5F5", "#1E88E5", "#0D47A1"];
const INTERVIEW_COLORS: Record<string, string> = {
  Pass: "#10B981",
  Fail: "#EF4444",
  Hold: "#F59E0B",
};

export function AnalyticsPage() {
  const summaryQ = useQuery({ queryKey: ["analytics", "summary"], queryFn: api.analyticsSummary });
  const funnelQ = useQuery({ queryKey: ["analytics", "funnel"], queryFn: api.analyticsFunnel });
  const monthlyQ = useQuery({ queryKey: ["analytics", "monthly"], queryFn: api.analyticsMonthly });
  const sourcesQ = useQuery({ queryKey: ["analytics", "sources"], queryFn: api.analyticsSources });
  const interviewsQ = useQuery({ queryKey: ["analytics", "interviews"], queryFn: api.analyticsInterviews });
  const offersQ = useQuery({ queryKey: ["analytics", "offers"], queryFn: api.analyticsOffers });
  const rolesQ = useQuery({ queryKey: ["analytics", "roles"], queryFn: api.analyticsRoles });
  const timeQ = useQuery({ queryKey: ["analytics", "time"], queryFn: api.analyticsTime });

  const isLoading =
    summaryQ.isLoading ||
    funnelQ.isLoading ||
    monthlyQ.isLoading ||
    sourcesQ.isLoading ||
    interviewsQ.isLoading ||
    offersQ.isLoading ||
    rolesQ.isLoading ||
    timeQ.isLoading;

  if (isLoading) return <PageSkeleton />;

  const summary = summaryQ.data;
  const funnel = funnelQ.data ?? [];
  const monthly = monthlyQ.data ?? [];
  const sources = sourcesQ.data ?? [];
  const interviews = interviewsQ.data ?? [];
  const offers = offersQ.data ?? [];
  const roles = rolesQ.data ?? [];
  const time = timeQ.data;

  if (!summary || !time) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-xl p-5 text-red-700 text-sm">
        Failed to load recruitment analytics data.
      </div>
    );
  }

  const sourceColumns: ColumnDef<SourceStat>[] = [
    { key: "source", header: "Channel", sortable: true, sortValue: (r) => r.source, render: (r) => <span className="font-medium">{r.source}</span> },
    { key: "apps", header: "Applications", align: "right", sortable: true, sortValue: (r) => r.total_applications, render: (r) => r.total_applications.toLocaleString() },
    { key: "offers", header: "Offers", align: "right", sortable: true, sortValue: (r) => r.offers_received, render: (r) => r.offers_received.toLocaleString() },
    { key: "acc", header: "Accepted", align: "right", sortable: true, sortValue: (r) => r.offers_accepted, render: (r) => r.offers_accepted.toLocaleString() },
    { key: "rate", header: "Acceptance %", align: "right", sortable: true, sortValue: (r) => r.acceptance_rate_pct, render: (r) => `${r.acceptance_rate_pct.toFixed(1)}%` },
  ];

  return (
    <div>
      <h2 className="text-[1.6rem] font-bold text-[#31527F] mb-1">
        Recruitment Analytics &amp; Pipeline Intelligence
      </h2>
      <p className="text-[#475569] mb-5">
        Empirical analysis across {summary.total_applications.toLocaleString()} candidate applications, sourcing performance, and hiring velocity.
      </p>

      {/* 5 Top-Level KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        <MetricCard
          title="Total Applications"
          value={summary.total_applications.toLocaleString()}
          subtitle="All active pipelines"
        />
        <MetricCard
          title="Total Offers Made"
          value={summary.total_offers.toLocaleString()}
          subtitle="Candidates reached offer"
        />
        <MetricCard
          title="Accepted Offers"
          value={summary.accepted_offers.toLocaleString()}
          valueClassName="text-[#059669]"
          subtitle={`Acceptance: ${summary.acceptance_rate_pct.toFixed(1)}%`}
        />
        <MetricCard
          title="Declined Offers"
          value={summary.rejected_offers.toLocaleString()}
          valueClassName="text-[#DC2626]"
          subtitle={`Declined: ${summary.declined_rate_pct.toFixed(1)}%`}
        />
        <MetricCard
          title="Avg Acceptance Time"
          value={`${summary.avg_time_to_acceptance_days.toFixed(1)} Days`}
          subtitle="Offer to acceptance"
        />
      </div>

      {/* 1. Pipeline Funnel & Monthly Trends */}
      <SectionHeader title="1. Pipeline Funnel Snapshot & Monthly Application Trends" />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-white border border-[#E5EBF4] rounded-xl p-4">
          <h3 className="text-sm font-bold text-[#243B60] mb-2">
            Current Pipeline Stage Snapshot (Applicant Distribution)
          </h3>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={funnel} layout="vertical" margin={{ top: 10, right: 30, left: 20, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis type="number" tick={{ fontSize: 12, fill: "#475569" }} />
                <YAxis type="category" dataKey="label" tick={{ fontSize: 12, fill: "#263449" }} width={80} />
                <Tooltip
                  formatter={(value: number, _name: string, props: { payload?: { percentage?: number } }) => [
                    `${Number(value).toLocaleString()} (${props?.payload?.percentage ?? 0}%)`,
                    "Candidates",
                  ]}
                />
                <Bar dataKey="candidate_count" radius={[0, 6, 6, 0]}>
                  {funnel.map((entry, idx) => (
                    <Cell key={entry.stage} fill={FUNNEL_COLORS[idx % FUNNEL_COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="text-xs text-[#64748B] mt-2">
            Note: Snapshot reflects current applicant distribution across pipeline stages. Historical conversion rates are not claimed as transitional timestamps are not logged for earlier dropped stages.
          </p>
        </div>

        <div className="bg-white border border-[#E5EBF4] rounded-xl p-4">
          <h3 className="text-sm font-bold text-[#243B60] mb-2">Monthly Application Volume Trends</h3>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={monthly} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis dataKey="year_month" tick={{ fontSize: 11, fill: "#475569" }} angle={-35} textAnchor="end" height={50} />
                <YAxis tick={{ fontSize: 12, fill: "#475569" }} />
                <Tooltip />
                <Line type="monotone" dataKey="application_count" name="Applications" stroke="#2563EB" strokeWidth={2.5} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="text-xs text-[#64748B] mt-2">
            Tracks monthly applicant inflow volume across the recorded recruitment timeframe.
          </p>
        </div>
      </div>

      {/* 2. Sourcing Channel Performance */}
      <SectionHeader title="2. Sourcing Channel Efficacy & Offer Conversion" />
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
        <div className="lg:col-span-3 bg-white border border-[#E5EBF4] rounded-xl p-4">
          <h3 className="text-sm font-bold text-[#243B60] mb-2">
            Application Volume and Offer Acceptances by Source Channel
          </h3>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={sources} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis dataKey="source" tick={{ fontSize: 12, fill: "#475569" }} />
                <YAxis tick={{ fontSize: 12, fill: "#475569" }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="total_applications" name="Applications" fill="#2563EB" radius={[6, 6, 0, 0]} />
                <Bar dataKey="offers_accepted" name="Accepted Offers" fill="#0D9488" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="lg:col-span-2 flex flex-col gap-2">
          <div className="text-sm font-bold text-[#243B60]">Channel Conversion Performance Table:</div>
          <DataTable data={sources} columns={sourceColumns} rowKey={(r) => r.source} />
          <p className="text-xs text-[#64748B]">
            Evaluates volume vs quality: Referral and Outbound channels demonstrate higher offer-to-acceptance ratios compared to generic job boards.
          </p>
        </div>
      </div>

      {/* 3. Interview Outcomes & Offer Decisions */}
      <SectionHeader title="3. Evaluation Outcomes & Final Offer Decisions" />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-white border border-[#E5EBF4] rounded-xl p-4">
          <h3 className="text-sm font-bold text-[#243B60] mb-2">Overall Interview Round Outcomes Distribution</h3>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={interviews} dataKey="count" nameKey="outcome" innerRadius={55} outerRadius={95} paddingAngle={2} label>
                  {interviews.map((entry) => (
                    <Cell key={entry.outcome} fill={INTERVIEW_COLORS[entry.outcome] ?? "#64748B"} />
                  ))}
                </Pie>
                <Tooltip formatter={(v: number) => v.toLocaleString()} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <p className="text-xs text-[#64748B]">
            Aggregated results across all interview assessments (Pass, Fail, Hold).
          </p>
        </div>

        <div className="bg-white border border-[#E5EBF4] rounded-xl p-4">
          <h3 className="text-sm font-bold text-[#243B60] mb-2">
            Candidate Offer Decision Breakdown (Total Offers: {summary.total_offers.toLocaleString()})
          </h3>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={offers} dataKey="count" nameKey="label" innerRadius={55} outerRadius={95} paddingAngle={2} label>
                  {offers.map((entry) => (
                    <Cell key={entry.label} fill={entry.accepted ? "#10B981" : "#EF4444"} />
                  ))}
                </Pie>
                <Tooltip formatter={(v: number) => v.toLocaleString()} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <p className="text-xs text-[#64748B]">
            Final decisions across all {summary.total_offers.toLocaleString()} extended offers: {summary.accepted_offers.toLocaleString()} Accepted ({summary.acceptance_rate_pct.toFixed(1)}%), {summary.rejected_offers.toLocaleString()} Declined ({summary.declined_rate_pct.toFixed(1)}%).
          </p>
        </div>
      </div>

      {/* 4. Role Family Distribution */}
      <SectionHeader title="4. Application Volume by Job Role Family" />
      <div className="bg-white border border-[#E5EBF4] rounded-xl p-4">
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={roles} layout="vertical" margin={{ top: 10, right: 30, left: 30, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
              <XAxis type="number" tick={{ fontSize: 12, fill: "#475569" }} />
              <YAxis type="category" dataKey="role_family" tick={{ fontSize: 12, fill: "#263449" }} width={140} />
              <Tooltip />
              <Bar dataKey="count" name="Applications" fill="#0D9488" radius={[0, 6, 6, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 5. Hiring Velocity & Time Duration Analytics */}
      <SectionHeader title="5. Hiring Velocity & Time Duration Analytics" />
      <AlertCallout>
        <strong>Data Integrity Protocol – {time.anomalies_excluded.toLocaleString()} Date Anomalies Excluded:</strong>
        <br />
        In the raw pipeline records, exactly <strong>{time.anomalies_excluded.toLocaleString()} offer rows</strong> exhibit timestamp corruption where <code>accepted_at &lt; offered_at</code>. To ensure empirical integrity, these records are completely isolated and excluded from duration metrics. Calculations below use the <strong>{summary.valid_tta_samples.toLocaleString()} genuine non-anomalous accepted offers</strong>.
      </AlertCallout>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
        <MetricCard title="Avg Time-to-Acceptance" value={`${time.tta_mean.toFixed(1)} Days`} />
        <MetricCard title="Median Time-to-Acceptance" value={`${time.tta_median.toFixed(1)} Days`} />
        <MetricCard title="Avg Full Hire Duration" value={`${time.full_mean.toFixed(1)} Days`} />
        <MetricCard title="Median Full Hire Duration" value={`${time.full_median.toFixed(1)} Days`} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-white border border-[#E5EBF4] rounded-xl p-4">
          <h3 className="text-sm font-bold text-[#243B60] mb-2">
            Time-to-Acceptance Distribution (Days from Offer to Acceptance)
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={time.tta_histogram}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis dataKey="label" tick={{ fontSize: 10, fill: "#475569" }} interval={3} />
                <YAxis tick={{ fontSize: 12, fill: "#475569" }} />
                <Tooltip />
                <Bar dataKey="count" name="Offers" fill="#2563EB" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="text-xs text-[#64748B] mt-2">
            Distribution of days between offer release and candidate acceptance decision.
          </p>
        </div>

        <div className="bg-white border border-[#E5EBF4] rounded-xl p-4">
          <h3 className="text-sm font-bold text-[#243B60] mb-2">
            Application-to-Accepted-Offer Duration (Total Days)
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={time.full_histogram}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis dataKey="label" tick={{ fontSize: 10, fill: "#475569" }} interval={4} />
                <YAxis tick={{ fontSize: 12, fill: "#475569" }} />
                <Tooltip />
                <Bar dataKey="count" name="Offers" fill="#0D9488" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="text-xs text-[#64748B] mt-2">
            Distribution of total elapsed hiring days from initial application submission to formal offer acceptance.
          </p>
        </div>
      </div>
    </div>
  );
}
