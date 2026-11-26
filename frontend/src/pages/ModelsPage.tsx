import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api/client";
import { PageSkeleton } from "../components/ErrorBoundary";
import { MetricCard } from "../components/MetricCard";
import { SectionHeader } from "../components/SectionHeader";

const MODEL_COLORS: Record<string, string> = {
  "Majority-Class Baseline": "#78909C",
  "Logistic Regression": "#2563EB",
  "Decision Tree": "#F59E0B",
  "Random Forest": "#10B981",
};

const METRIC_KEYS = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"] as const;

export function ModelsPage() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["models"],
    queryFn: api.modelsComparison,
  });

  const [selectedModelFeat, setSelectedModelFeat] = useState<string>("Random Forest");

  const bestPerMetric = useMemo(() => {
    const best: Record<string, number> = {};
    if (!data) return best;
    for (const key of METRIC_KEYS) {
      best[key] = Math.max(...data.metrics_table.map((row) => row[key]));
    }
    return best;
  }, [data]);

  const groupedBarData = useMemo(() => {
    if (!data) return [];
    return METRIC_KEYS.map((metric) => {
      const point: Record<string, string | number> = { Metric: metric };
      for (const row of data.metrics_table) {
        point[row.Model] = Number(row[metric].toFixed(3));
      }
      return point;
    });
  }, [data]);

  const rocSeriesData = useMemo(() => {
    if (!data) return [];
    // Build unified grid 0..1 with interpolated TPR for clean Recharts multi-line rendering
    const steps = 40;
    const grid = Array.from({ length: steps + 1 }, (_, i) => Number((i / steps).toFixed(3)));
    return grid.map((x) => {
      const point: Record<string, number> = { fpr: x, random: x };
      for (const [modelName, curve] of Object.entries(data.roc_data)) {
        let bestTpr = 0;
        for (let idx = 0; idx < curve.fpr.length; idx++) {
          if (curve.fpr[idx] <= x + 1e-6) {
            bestTpr = curve.tpr[idx];
          }
        }
        point[modelName] = Number(bestTpr.toFixed(3));
      }
      return point;
    });
  }, [data]);

  if (isLoading) return <PageSkeleton />;
  if (error || !data) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-xl p-5 text-red-700 text-sm">
        Failed to load model comparison data: {error instanceof Error ? error.message : "Unknown error"}
      </div>
    );
  }

  const acceptPct = data.test_sample_count > 0 ? ((data.test_accepted_count / data.test_sample_count) * 100).toFixed(1) : "0.0";
  const lrTn = data.confusion_matrices["Logistic Regression"]?.[0]?.[0] ?? 0;
  const rfTn = data.confusion_matrices["Random Forest"]?.[0]?.[0] ?? 0;
  const featItems = [...(data.feature_importances[selectedModelFeat] ?? [])].reverse();
  const isLinear = selectedModelFeat === "Logistic Regression";

  return (
    <div>
      <h2 className="text-[1.6rem] font-bold text-[#31527F] mb-1">
        Predictive Offer Acceptance – Model Comparison
      </h2>
      <p className="text-[#475569] mb-5">
        Rigorous comparative evaluation of four architectures trained exclusively on pre-decision signals.
      </p>

      {/* 1. Executive Summary Table */}
      <SectionHeader
        title={`1. Held-Out Test Set Performance (N=${data.test_sample_count} Stratified Samples)`}
      />
      <div className="bg-white border border-[#E3EAF4] rounded-[11px] overflow-x-auto shadow-[0_2px_9px_rgba(45,70,110,0.04)]">
        <table className="w-full text-left border-collapse text-sm">
          <thead>
            <tr className="bg-[#F8FAFD] border-b border-[#E3EAF4] text-[#31527F]">
              <th className="px-4 py-3 font-semibold">Model</th>
              {METRIC_KEYS.map((k) => (
                <th key={k} className="px-4 py-3 font-semibold text-right">
                  {k}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-[#F0F2F7]">
            {data.metrics_table.map((row) => (
              <tr key={row.Model} className="hover:bg-[#FAFBFD]">
                <td className="px-4 py-3 font-semibold text-[#243B60]">{row.Model}</td>
                {METRIC_KEYS.map((k) => {
                  const val = row[k];
                  const isBest = Math.abs(val - bestPerMetric[k]) < 1e-6;
                  return (
                    <td
                      key={k}
                      className={`px-4 py-3 text-right tabular-nums ${
                        isBest ? "bg-[#E0F5EB]/60 text-[#059669] font-bold" : "text-[#263449]"
                      }`}
                    >
                      {val.toFixed(3)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-[#64748B] mt-2">
        All models evaluated on the identical 20% held-out test split ({data.test_accepted_count} Accepted, {data.test_rejected_count} Declined). Preprocessing is encapsulated inside Scikit-learn Pipelines to eliminate data leakage.
      </p>

      {/* 2. Comparative Multi-Bar Chart */}
      <SectionHeader title="2. Cross-Model Metric Comparison" />
      <div className="bg-white border border-[#E5EBF4] rounded-xl p-4">
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={groupedBarData} margin={{ top: 15, right: 20, left: 0, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
              <XAxis dataKey="Metric" tick={{ fontSize: 12, fill: "#263449" }} />
              <YAxis domain={[0, 1.05]} tick={{ fontSize: 12, fill: "#475569" }} />
              <Tooltip />
              <Legend />
              {data.metrics_table.map((row) => (
                <Bar
                  key={row.Model}
                  dataKey={row.Model}
                  fill={MODEL_COLORS[row.Model] ?? "#475569"}
                  radius={[4, 4, 0, 0]}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 3. Model Architecture Descriptions */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4 mt-5">
        <MetricCard title="Majority-Class Baseline" value="" className="h-full">
          <div className="text-[0.84rem] text-[#334155] space-y-1.5 leading-relaxed">
            <p><strong>Type:</strong> Empirical Reference Benchmark</p>
            <p><strong>Mechanism:</strong> Predicts majority outcome (Accepted) for 100% of cases without feature inspection.</p>
            <p><strong>Role &amp; Limitation:</strong> Benchmarks {acceptPct}% class imbalance; catches 0% of declined offers (ROC-AUC = 0.500).</p>
          </div>
        </MetricCard>
        <MetricCard title="Model 1: Logistic Regression" value="" className="h-full">
          <div className="text-[0.84rem] text-[#334155] space-y-1.5 leading-relaxed">
            <p><strong>Type:</strong> Generalized Linear Model</p>
            <p><strong>Mechanism:</strong> Fits linear decision boundary with balanced class weighting and sigmoid probability transformation.</p>
            <p><strong>Key Strength:</strong> Highly interpretable feature log-odds coefficients.</p>
          </div>
        </MetricCard>
        <MetricCard title="Model 2: Decision Tree" value="" className="h-full">
          <div className="text-[0.84rem] text-[#334155] space-y-1.5 leading-relaxed">
            <p><strong>Type:</strong> Non-linear Rule Induction</p>
            <p><strong>Mechanism:</strong> Recursively partitions feature space using Gini impurity with maximum depth constraint to prevent memorization.</p>
            <p><strong>Key Strength:</strong> Discovers non-linear thresholds and salary cutoff rules.</p>
          </div>
        </MetricCard>
        <MetricCard title="Model 3: Random Forest" value="" className="h-full">
          <div className="text-[0.84rem] text-[#334155] space-y-1.5 leading-relaxed">
            <p><strong>Type:</strong> Ensemble of 100 Decision Trees</p>
            <p><strong>Mechanism:</strong> Aggregates bootstrap subsamples with random feature subset selection (bagging).</p>
            <p><strong>Key Strength:</strong> Superior variance reduction, yielding highest F1-score and balanced test accuracy.</p>
          </div>
        </MetricCard>
      </div>

      {/* 4. Confusion Matrices */}
      <SectionHeader title="3. Confusion Matrix Diagnostics" />
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        {data.metrics_table.map((row) => {
          const cm = data.confusion_matrices[row.Model];
          if (!cm) return null;
          const tn = cm[0][0];
          const fp = cm[0][1];
          const fn = cm[1][0];
          const tp = cm[1][1];
          return (
            <div key={row.Model} className="bg-white border border-[#E3EAF4] rounded-xl p-4 shadow-sm">
              <div className="text-xs font-bold text-[#243B60] mb-3 text-center">
                Confusion Matrix – {row.Model}
              </div>
              <div className="grid grid-cols-2 gap-2 text-center text-xs">
                <div className="bg-[#EBF3FE] border border-[#C9DAF5] rounded-lg p-3">
                  <div className="text-[#475569] font-medium">True Negative (TN)</div>
                  <div className="text-xl font-extrabold text-[#1E3A8A] mt-1">{tn}</div>
                </div>
                <div className="bg-[#F8FAFC] border border-[#E2E8F0] rounded-lg p-3">
                  <div className="text-[#64748B] font-medium">False Positive (FP)</div>
                  <div className="text-xl font-extrabold text-[#475569] mt-1">{fp}</div>
                </div>
                <div className="bg-[#F8FAFC] border border-[#E2E8F0] rounded-lg p-3">
                  <div className="text-[#64748B] font-medium">False Negative (FN)</div>
                  <div className="text-xl font-extrabold text-[#475569] mt-1">{fn}</div>
                </div>
                <div className="bg-[#E0F5EB] border border-[#A7F3D0] rounded-lg p-3">
                  <div className="text-[#047857] font-medium">True Positive (TP)</div>
                  <div className="text-xl font-extrabold text-[#065F46] mt-1">{tp}</div>
                </div>
              </div>
              <div className="text-[0.7rem] text-[#64748B] text-center mt-2">
                Rows: Actual (0=Declined, 1=Accepted)
              </div>
            </div>
          );
        })}
      </div>
      <p className="text-xs text-[#64748B] mt-2">
        Class Breakdown: Actual 0 = Candidate Declined Offer ({data.test_rejected_count} cases) | Actual 1 = Candidate Accepted Offer ({data.test_accepted_count} cases). Notice: The Majority-Class Baseline predicts 100% accepted, catching 0 declined offers (TN=0). Balanced class weighting in Logistic Regression (TN={lrTn}) and Random Forest (TN={rfTn}) actively detects offer rejections despite the class imbalance.
      </p>

      {/* 5. Combined ROC Curves */}
      <SectionHeader title="4. Discriminative Power (ROC Curves)" />
      <div className="bg-white border border-[#E5EBF4] rounded-xl p-4">
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={rocSeriesData} margin={{ top: 10, right: 20, left: 10, bottom: 15 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
              <XAxis
                dataKey="fpr"
                type="number"
                domain={[0, 1]}
                label={{ value: "False Positive Rate (1 - Specificity)", position: "insideBottom", offset: -8, fontSize: 12 }}
                tick={{ fontSize: 11, fill: "#475569" }}
              />
              <YAxis
                domain={[0, 1.02]}
                label={{ value: "True Positive Rate", angle: -90, position: "insideLeft", fontSize: 12 }}
                tick={{ fontSize: 11, fill: "#475569" }}
              />
              <Tooltip />
              <Legend verticalAlign="top" height={36} />
              <Line type="linear" dataKey="random" name="Random Classifier (AUC = 0.500)" stroke="#94A3B8" strokeDasharray="5 5" dot={false} strokeWidth={1.5} />
              {Object.entries(data.roc_data).map(([modelName, curve]) => (
                <Line
                  key={modelName}
                  type="monotone"
                  dataKey={modelName}
                  name={`${modelName} (AUC = ${curve.auc.toFixed(3)})`}
                  stroke={MODEL_COLORS[modelName] ?? "#334155"}
                  dot={false}
                  strokeWidth={2.5}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
        <p className="text-xs text-[#64748B] mt-2">
          Curves illustrate trade-off between True Positive Rate (Sensitivity) and False Positive Rate across all decision probability thresholds.
        </p>
      </div>

      {/* 6. Feature Importance & Predictive Drivers */}
      <SectionHeader title="5. Empirical Feature Drivers of Offer Acceptance" />
      <div className="bg-white border border-[#E5EBF4] rounded-xl p-4">
        <div className="max-w-xs mb-4">
          <label htmlFor="feat-model-select" className="block text-sm font-semibold text-[#243B60] mb-1">
            Select Model to Inspect Feature Influence:
          </label>
          <select
            id="feat-model-select"
            value={selectedModelFeat}
            onChange={(e) => setSelectedModelFeat(e.target.value)}
            className="w-full px-3 py-2 border border-[#D3DEEF] rounded-lg text-sm bg-white text-[#263449]"
          >
            {["Random Forest", "Logistic Regression", "Decision Tree"].map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>

        <div className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={featItems}
              layout="vertical"
              margin={{ top: 10, right: 30, left: 40, bottom: 10 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
              <XAxis type="number" tick={{ fontSize: 11, fill: "#475569" }} />
              <YAxis type="category" dataKey="display_feature" width={190} tick={{ fontSize: 11, fill: "#263449" }} />
              <Tooltip />
              {isLinear && <ReferenceLine x={0} stroke="#64748B" />}
              <Bar dataKey={isLinear ? "coefficient" : "importance"} name={isLinear ? "Coefficient" : "Importance"} radius={[0, 4, 4, 0]}>
                {featItems.map((entry) => {
                  const val = isLinear ? entry.coefficient ?? 0 : entry.importance ?? 0;
                  const fill = isLinear ? (val >= 0 ? "#10B981" : "#EF4444") : "#0D9488";
                  return <Cell key={entry.feature} fill={fill} />;
                })}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="mt-4 pt-4 border-t border-[#EDF0F6] text-sm text-[#334155] space-y-1.5">
          <div className="font-bold text-[#243B60]">Operational Takeaways:</div>
          <ul className="list-disc pl-5 space-y-1">
            <li>
              <strong>Offered Base Salary &amp; Salary-to-Midpoint Ratio:</strong> Highest relative importance across models; offers that meet or exceed the requisition midpoint have significantly higher acceptance likelihoods.
            </li>
            <li>
              <strong>Interview Score:</strong> Candidates with higher technical interview evaluation scores exhibit stronger candidate-role mutual fit.
            </li>
            <li>
              <strong>Years of Experience:</strong> Senior candidates with 10+ years demonstrate more selective acceptance patterns compared to junior applicants.
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
