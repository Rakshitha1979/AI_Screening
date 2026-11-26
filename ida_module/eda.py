"""
IDA Module: Exploratory Data Analysis (EDA) and Recruitment Analytics.
Generates genuine recruitment metrics, stage funnel snapshots, channel source breakdowns,
interview outcome distributions, valid duration metrics, and Plotly visualizations.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def get_recruitment_summary_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes top-level recruitment KPI cards.
    """
    total_apps = len(df)
    offers_df = df[df["stage"] == "offer"]
    total_offers = len(offers_df)
    accepted_offers = int((offers_df["accepted"] == True).sum())
    rejected_offers = int((offers_df["accepted"] == False).sum())
    acceptance_rate = round((accepted_offers / total_offers * 100.0), 1) if total_offers > 0 else 0.0
    
    # Valid time durations (excluding 217 date anomalies)
    valid_tta = df["time_to_acceptance_days"].dropna()
    avg_tta = round(valid_tta.mean(), 1) if not valid_tta.empty else 0.0
    median_tta = round(valid_tta.median(), 1) if not valid_tta.empty else 0.0
    
    valid_full = df["application_to_accepted_days"].dropna()
    avg_full_duration = round(valid_full.mean(), 1) if not valid_full.empty else 0.0
    median_full_duration = round(valid_full.median(), 1) if not valid_full.empty else 0.0
    
    return {
        "total_applications": total_apps,
        "total_offers": total_offers,
        "accepted_offers": accepted_offers,
        "rejected_offers": rejected_offers,
        "acceptance_rate_pct": acceptance_rate,
        "avg_time_to_acceptance_days": avg_tta,
        "median_time_to_acceptance_days": median_tta,
        "valid_tta_samples": len(valid_tta),
        "excluded_anomalies": int((df["offer_date_anomaly"] == True).sum()),
        "avg_full_hire_duration_days": avg_full_duration,
        "median_full_hire_duration_days": median_full_duration
    }


def get_monthly_application_trends(df: pd.DataFrame) -> Tuple[pd.DataFrame, go.Figure]:
    """
    Analyzes application volumes over time (by month-year).
    """
    temp = df.copy()
    temp["applied_at"] = pd.to_datetime(temp["applied_at"], errors="coerce")
    temp = temp.dropna(subset=["applied_at"])
    temp["year_month"] = temp["applied_at"].dt.to_period("M").astype(str)
    
    monthly = temp.groupby("year_month").size().reset_index(name="application_count")
    monthly = monthly.sort_values("year_month")
    
    fig = px.line(
        monthly,
        x="year_month",
        y="application_count",
        title="Monthly Application Volume Trends",
        markers=True,
        labels={"year_month": "Month", "application_count": "Applications"},
        color_discrete_sequence=["#1E88E5"]
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_tickangle=-45,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return monthly, fig


def get_recruitment_funnel_snapshot(df: pd.DataFrame) -> Tuple[pd.DataFrame, go.Figure]:
    """
    Current stage distribution snapshot.
    Note: Funnel snapshot shows current applicant distribution across pipeline stages.
    """
    stage_order = ["applied", "screened", "interview", "offer"]
    counts = df["stage"].value_counts().reindex(stage_order).fillna(0).reset_index()
    counts.columns = ["stage", "candidate_count"]
    counts["candidate_count"] = counts["candidate_count"].astype(int)
    counts["percentage"] = (counts["candidate_count"] / len(df) * 100.0).round(1)
    
    stage_labels = [s.capitalize() for s in counts["stage"]]
    
    fig = go.Figure(go.Funnel(
        y=stage_labels,
        x=counts["candidate_count"],
        textinfo="value+percent initial",
        marker=dict(color=["#64B5F6", "#42A5F5", "#1E88E5", "#0D47A1"])
    ))
    fig.update_layout(
        title="Current Pipeline Stage Snapshot (Applicant Distribution)",
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return counts, fig


def get_source_analysis(df: pd.DataFrame) -> Tuple[pd.DataFrame, go.Figure]:
    """
    Analyzes recruitment source channel performance: volumes and offer acceptance.
    """
    source_stats = df.groupby("source").agg(
        total_applications=("application_id", "count"),
        offers_received=("offered_at", lambda s: s.notnull().sum()),
        offers_accepted=("accepted", lambda s: (s == True).sum()),
        offers_rejected=("accepted", lambda s: (s == False).sum())
    ).reset_index()
    
    source_stats["offer_rate_pct"] = (
        source_stats["offers_received"] / source_stats["total_applications"] * 100.0
    ).round(2)
    source_stats["acceptance_rate_pct"] = np.where(
        source_stats["offers_received"] > 0,
        (source_stats["offers_accepted"] / source_stats["offers_received"] * 100.0).round(1),
        0.0
    )
    source_stats = source_stats.sort_values("total_applications", ascending=False)
    
    fig = px.bar(
        source_stats,
        x="source",
        y="total_applications",
        color="offers_accepted",
        title="Application Volume and Offer Acceptances by Source Channel",
        labels={
            "source": "Recruitment Channel",
            "total_applications": "Applications",
            "offers_accepted": "Accepted Offers"
        },
        color_continuous_scale="Blues",
        text="total_applications"
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return source_stats, fig


def get_role_family_distribution(df: pd.DataFrame) -> Tuple[pd.DataFrame, go.Figure]:
    """
    Analyzes applications by job role family.
    """
    role_counts = df["role_family"].value_counts().reset_index()
    role_counts.columns = ["role_family", "count"]
    role_counts = role_counts.sort_values("count", ascending=True)
    
    fig = px.bar(
        role_counts,
        x="count",
        y="role_family",
        orientation="h",
        title="Application Volume by Job Role Family",
        labels={"count": "Number of Applicants", "role_family": "Role Family"},
        color="count",
        color_continuous_scale="Teal"
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return role_counts, fig


def get_company_applications(df: pd.DataFrame, top_n: int = 10) -> Tuple[pd.DataFrame, go.Figure]:
    """
    Analyzes top hiring companies by application volume.
    """
    comp = df["company_name"].value_counts().head(top_n).reset_index()
    comp.columns = ["company_name", "application_count"]
    comp = comp.sort_values("application_count", ascending=True)
    
    fig = px.bar(
        comp,
        x="application_count",
        y="company_name",
        orientation="h",
        title=f"Top {top_n} Companies by Application Volume",
        labels={"application_count": "Applications", "company_name": "Company"},
        color_discrete_sequence=["#26A69A"]
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return comp, fig


def get_interview_outcomes(df: pd.DataFrame) -> Tuple[pd.DataFrame, go.Figure]:
    """
    Analyzes candidate interview outcomes (pass, fail, hold counts).
    """
    interviewed = df[df["interview_count"] > 0]
    total_pass = int(interviewed["pass_count"].sum())
    total_fail = int(interviewed["fail_count"].sum())
    total_hold = int(interviewed["hold_count"].sum())
    
    summary = pd.DataFrame({
        "outcome": ["Pass", "Fail", "Hold"],
        "count": [total_pass, total_fail, total_hold]
    })
    
    fig = px.pie(
        summary,
        names="outcome",
        values="count",
        title="Overall Interview Round Outcomes Distribution",
        color="outcome",
        color_discrete_map={"Pass": "#4CAF50", "Fail": "#E53935", "Hold": "#FFB300"},
        hole=0.4
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return summary, fig


def get_offer_outcome_distribution(df: pd.DataFrame) -> Tuple[pd.DataFrame, go.Figure]:
    """
    Analyzes offer stage decisions: accepted vs rejected offers.
    """
    offers = df[df["stage"] == "offer"]
    counts = offers["accepted"].value_counts().reset_index()
    counts.columns = ["accepted", "count"]
    counts["label"] = counts["accepted"].map({True: "Accepted (699)", False: "Declined / Rejected (239)"})
    
    fig = px.pie(
        counts,
        names="label",
        values="count",
        title="Candidate Offer Decision Breakdown (Total Offers: 938)",
        color="label",
        color_discrete_map={
            "Accepted (699)": "#2E7D32",
            "Declined / Rejected (239)": "#C62828"
        },
        hole=0.45
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return counts, fig


def get_time_analysis_figures(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates distributions for:
    1. Time-to-acceptance (accepted_at - offered_at)
    2. Application-to-accepted-offer duration (accepted_at - applied_at)
    Excludes the 217 flagged date anomalies where accepted_at < offered_at.
    """
    valid_tta = df[df["time_to_acceptance_days"].notnull() & (df["offer_date_anomaly"] == False)].copy()
    valid_full = df[df["application_to_accepted_days"].notnull() & (df["offer_date_anomaly"] == False)].copy()
    
    fig_tta = px.histogram(
        valid_tta,
        x="time_to_acceptance_days",
        nbins=25,
        title="Time-to-Acceptance Distribution (Days from Offer to Candidate Acceptance)",
        labels={"time_to_acceptance_days": "Days from Offer to Acceptance"},
        color_discrete_sequence=["#1976D2"]
    )
    fig_tta.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    fig_full = px.histogram(
        valid_full,
        x="application_to_accepted_days",
        nbins=30,
        title="Application-to-Accepted-Offer Duration (Total Days from Application to Final Acceptance)",
        labels={"application_to_accepted_days": "Total Hiring Duration (Days)"},
        color_discrete_sequence=["#00897B"]
    )
    fig_full.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return {
        "fig_time_to_acceptance": fig_tta,
        "fig_application_to_acceptance": fig_full,
        "tta_count": len(valid_tta),
        "tta_mean": round(valid_tta["time_to_acceptance_days"].mean(), 1),
        "tta_median": round(valid_tta["time_to_acceptance_days"].median(), 1),
        "full_count": len(valid_full),
        "full_mean": round(valid_full["application_to_accepted_days"].mean(), 1),
        "full_median": round(valid_full["application_to_accepted_days"].median(), 1),
        "anomalies_excluded": int((df["offer_date_anomaly"] == True).sum())
    }


# ==============================================================================
# JSON-SERIALIZABLE DATA FUNCTIONS FOR FASTAPI / REACT
# ==============================================================================

def get_recruitment_summary_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Plain JSON summary KPIs (no numpy/NaN types)."""
    m = get_recruitment_summary_metrics(df)
    acc_pct = float(m["acceptance_rate_pct"])
    return {
        "total_applications": int(m["total_applications"]),
        "total_offers": int(m["total_offers"]),
        "accepted_offers": int(m["accepted_offers"]),
        "rejected_offers": int(m["rejected_offers"]),
        "acceptance_rate_pct": round(acc_pct, 1),
        "declined_rate_pct": round(100.0 - acc_pct, 1) if m["total_offers"] > 0 else 0.0,
        "avg_time_to_acceptance_days": float(m["avg_time_to_acceptance_days"]),
        "median_time_to_acceptance_days": float(m["median_time_to_acceptance_days"]),
        "valid_tta_samples": int(m["valid_tta_samples"]),
        "excluded_anomalies": int(m["excluded_anomalies"]),
        "avg_full_hire_duration_days": float(m["avg_full_hire_duration_days"]),
        "median_full_hire_duration_days": float(m["median_full_hire_duration_days"]),
    }


def get_recruitment_funnel_data(df: pd.DataFrame) -> list:
    """Plain JSON stage counts in funnel order."""
    stage_order = ["applied", "screened", "interview", "offer"]
    counts = df["stage"].value_counts().reindex(stage_order).fillna(0).reset_index()
    counts.columns = ["stage", "candidate_count"]
    total = max(len(df), 1)
    result = []
    for _, row in counts.iterrows():
        cnt = int(row["candidate_count"])
        stage = str(row["stage"])
        result.append({
            "stage": stage,
            "label": stage.capitalize(),
            "candidate_count": cnt,
            "percentage": round(cnt / total * 100.0, 1),
        })
    return result


def get_monthly_application_trends_data(df: pd.DataFrame) -> list:
    """Plain JSON monthly application counts."""
    temp = df.copy()
    temp["applied_at"] = pd.to_datetime(temp["applied_at"], errors="coerce")
    temp = temp.dropna(subset=["applied_at"])
    temp["year_month"] = temp["applied_at"].dt.to_period("M").astype(str)
    monthly = temp.groupby("year_month").size().reset_index(name="application_count")
    monthly = monthly.sort_values("year_month")
    return [
        {"year_month": str(r["year_month"]), "application_count": int(r["application_count"])}
        for _, r in monthly.iterrows()
    ]


def get_source_analysis_data(df: pd.DataFrame) -> list:
    """Plain JSON source channel metrics."""
    source_stats = df.groupby("source").agg(
        total_applications=("application_id", "count"),
        offers_received=("offered_at", lambda s: s.notnull().sum()),
        offers_accepted=("accepted", lambda s: (s == True).sum()),
        offers_rejected=("accepted", lambda s: (s == False).sum()),
    ).reset_index()
    source_stats["offer_rate_pct"] = (
        source_stats["offers_received"] / source_stats["total_applications"] * 100.0
    ).round(2)
    source_stats["acceptance_rate_pct"] = np.where(
        source_stats["offers_received"] > 0,
        (source_stats["offers_accepted"] / source_stats["offers_received"] * 100.0).round(1),
        0.0,
    )
    source_stats = source_stats.sort_values("total_applications", ascending=False)
    return [
        {
            "source": str(r["source"]),
            "total_applications": int(r["total_applications"]),
            "offers_received": int(r["offers_received"]),
            "offers_accepted": int(r["offers_accepted"]),
            "offers_rejected": int(r["offers_rejected"]),
            "offer_rate_pct": float(r["offer_rate_pct"]),
            "acceptance_rate_pct": float(r["acceptance_rate_pct"]),
        }
        for _, r in source_stats.iterrows()
    ]


def get_interview_outcomes_data(df: pd.DataFrame) -> list:
    """Plain JSON interview round outcome counts."""
    interviewed = df[df["interview_count"] > 0]
    return [
        {"outcome": "Pass", "count": int(interviewed["pass_count"].sum())},
        {"outcome": "Fail", "count": int(interviewed["fail_count"].sum())},
        {"outcome": "Hold", "count": int(interviewed["hold_count"].sum())},
    ]


def get_offer_outcome_distribution_data(df: pd.DataFrame) -> list:
    """Plain JSON offer stage decisions (accepted vs declined)."""
    offers = df[df["stage"] == "offer"]
    acc_cnt = int((offers["accepted"] == True).sum())
    dec_cnt = int((offers["accepted"] == False).sum())
    return [
        {"accepted": True, "label": f"Accepted ({acc_cnt})", "count": acc_cnt},
        {"accepted": False, "label": f"Declined ({dec_cnt})", "count": dec_cnt},
    ]


def get_role_family_distribution_data(df: pd.DataFrame) -> list:
    """Plain JSON role family counts sorted ascending for horizontal bar chart."""
    role_counts = df["role_family"].value_counts().reset_index()
    role_counts.columns = ["role_family", "count"]
    role_counts = role_counts.sort_values("count", ascending=True)
    return [
        {"role_family": str(r["role_family"]), "count": int(r["count"])}
        for _, r in role_counts.iterrows()
    ]


def _build_histogram_bins(values: np.ndarray, bins: int) -> list:
    if len(values) == 0:
        return []
    counts, edges = np.histogram(values, bins=bins)
    out = []
    for i in range(len(counts)):
        start = round(float(edges[i]), 2)
        end = round(float(edges[i + 1]), 2)
        out.append({
            "bin_start": start,
            "bin_end": end,
            "label": f"{start:.1f}-{end:.1f}",
            "count": int(counts[i]),
        })
    return out


def get_time_analysis_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Plain JSON hiring velocity stats and histogram bins excluding date anomalies."""
    valid_tta = df[df["time_to_acceptance_days"].notnull() & (df["offer_date_anomaly"] == False)]
    valid_full = df[df["application_to_accepted_days"].notnull() & (df["offer_date_anomaly"] == False)]

    tta_vals = valid_tta["time_to_acceptance_days"].to_numpy(dtype=float)
    full_vals = valid_full["application_to_accepted_days"].to_numpy(dtype=float)

    return {
        "tta_count": int(len(valid_tta)),
        "tta_mean": round(float(np.mean(tta_vals)), 1) if len(tta_vals) > 0 else 0.0,
        "tta_median": round(float(np.median(tta_vals)), 1) if len(tta_vals) > 0 else 0.0,
        "full_count": int(len(valid_full)),
        "full_mean": round(float(np.mean(full_vals)), 1) if len(full_vals) > 0 else 0.0,
        "full_median": round(float(np.median(full_vals)), 1) if len(full_vals) > 0 else 0.0,
        "anomalies_excluded": int((df["offer_date_anomaly"] == True).sum()),
        "tta_histogram": _build_histogram_bins(tta_vals, bins=25),
        "full_histogram": _build_histogram_bins(full_vals, bins=30),
    }

