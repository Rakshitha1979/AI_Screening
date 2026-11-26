"""
Page 4: IDA Recruitment Analytics & Pipeline Intelligence.
Displays verified empirical metrics across the 15,000 application records:
funnel snapshots, monthly trends, sourcing channel effectiveness,
interview outcome distributions, and mathematically validated hiring velocities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from ida_module.eda import (
    get_recruitment_summary_metrics,
    get_monthly_application_trends,
    get_recruitment_funnel_snapshot,
    get_source_analysis,
    get_role_family_distribution,
    get_interview_outcomes,
    get_offer_outcome_distribution,
    get_time_analysis_figures
)


def render_page4(cleaned_df):
    """
    Renders IDA Recruitment Analytics page.
    """
    st.markdown("<h2 style='margin-bottom: 0px;'>Recruitment Analytics & Pipeline Intelligence</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #475569; margin-bottom: 20px;'>Empirical analysis across 15,000 candidate applications, sourcing performance, and hiring velocity.</p>", unsafe_allow_html=True)
    
    # 1. Top-Level Recruitment KPI Cards
    summary_metrics = get_recruitment_summary_metrics(cleaned_df)
    
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Total Applications</div>
            <div class="metric-card-value">{summary_metrics['total_applications']:,}</div>
            <div class="metric-card-sub">All active pipelines</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Total Offers Made</div>
            <div class="metric-card-value">{summary_metrics['total_offers']:,}</div>
            <div class="metric-card-sub">Candidates reached offer</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Accepted Offers</div>
            <div class="metric-card-value" style="color: #059669;">{summary_metrics['accepted_offers']:,}</div>
            <div class="metric-card-sub">Acceptance: {summary_metrics['acceptance_rate_pct']:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Declined Offers</div>
            <div class="metric-card-value" style="color: #DC2626;">{summary_metrics['rejected_offers']:,}</div>
            <div class="metric-card-sub">Declined: {100.0 - summary_metrics['acceptance_rate_pct']:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with k5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Avg Acceptance Time</div>
            <div class="metric-card-value">{summary_metrics['avg_time_to_acceptance_days']:.1f} Days</div>
            <div class="metric-card-sub">Offer to acceptance</div>
        </div>
        """, unsafe_allow_html=True)
        
    # 2. Pipeline Funnel Snapshot & Application Trends
    st.markdown("<div class='section-header'>1. Pipeline Funnel Snapshot & Monthly Application Trends</div>", unsafe_allow_html=True)
    
    col_funnel, col_trends = st.columns(2)
    with col_funnel:
        funnel_df, fig_funnel = get_recruitment_funnel_snapshot(cleaned_df)
        st.plotly_chart(fig_funnel, width="stretch")
        st.caption("Note: Snapshot reflects current applicant distribution across pipeline stages. Historical conversion rates are not claimed as transitional timestamps are not logged for earlier dropped stages.")
        
    with col_trends:
        monthly_df, fig_monthly = get_monthly_application_trends(cleaned_df)
        st.plotly_chart(fig_monthly, width="stretch")
        st.caption("Tracks monthly applicant inflow volume across the recorded recruitment timeframe.")

    # 3. Recruitment Source Channel Performance
    st.markdown("<div class='section-header'>2. Sourcing Channel Efficacy & Offer Conversion</div>", unsafe_allow_html=True)
    
    source_df, fig_source = get_source_analysis(cleaned_df)
    
    s_col1, s_col2 = st.columns([3, 2])
    with s_col1:
        st.plotly_chart(fig_source, width="stretch")
    with s_col2:
        st.markdown("**Channel Conversion Performance Table:**")
        display_source = source_df[[
            "source", "total_applications", "offers_received", "offers_accepted", "acceptance_rate_pct"
        ]].rename(columns={
            "source": "Channel",
            "total_applications": "Applications",
            "offers_received": "Offers",
            "offers_accepted": "Accepted",
            "acceptance_rate_pct": "Acceptance %"
        })
        st.dataframe(display_source, width="stretch", hide_index=True)
        st.caption("Evaluates volume vs quality: Referral and Outbound channels demonstrate higher offer-to-acceptance ratios compared to generic job boards.")

    # 4. Interview Rounds & Offer Decision Breakdown
    st.markdown("<div class='section-header'>3. Evaluation Outcomes & Final Offer Decisions</div>", unsafe_allow_html=True)
    
    int_col, off_col = st.columns(2)
    with int_col:
        int_summary, fig_int = get_interview_outcomes(cleaned_df)
        st.plotly_chart(fig_int, width="stretch")
        st.caption("Aggregated results across all interview assessments (Pass, Fail, Hold).")
    with off_col:
        off_summary, fig_off = get_offer_outcome_distribution(cleaned_df)
        st.plotly_chart(fig_off, width="stretch")
        st.caption(f"Final decisions across all 938 extended offers: 699 Accepted ({699/938*100:.1f}%), 239 Declined ({239/938*100:.1f}%).")

    # 5. Role Family Distribution
    st.markdown("<div class='section-header'>4. Application Volume by Job Role Family</div>", unsafe_allow_html=True)
    role_df, fig_role = get_role_family_distribution(cleaned_df)
    st.plotly_chart(fig_role, width="stretch")

    # 6. Valid Time Duration Analytics (Excluding Anomalies)
    st.markdown("<div class='section-header'>5. Hiring Velocity & Time Duration Analytics</div>", unsafe_allow_html=True)
    
    # Anomaly exclusion callout box
    st.markdown(f"""
    <div class="alert-callout">
        <b>Data Integrity Protocol – 217 Date Anomalies Excluded:</b><br>
        In the raw pipeline records, exactly <b>217 offer rows</b> exhibit timestamp corruption where <code>accepted_at &lt; offered_at</code> (e.g. candidate recorded as accepting 23 days before the offer was issued).
        To ensure empirical integrity, these 217 records are completely isolated and excluded from duration metrics. Calculations below use the <b>{summary_metrics['valid_tta_samples']} genuine non-anomalous accepted offers</b>.
    </div>
    """, unsafe_allow_html=True)
    
    time_info = get_time_analysis_figures(cleaned_df)
    
    t_card1, t_card2, t_card3, t_card4 = st.columns(4)
    with t_card1:
        st.metric("Avg Time-to-Acceptance", f"{time_info['tta_mean']:.1f} Days")
    with t_card2:
        st.metric("Median Time-to-Acceptance", f"{time_info['tta_median']:.1f} Days")
    with t_card3:
        st.metric("Avg Full Hire Duration", f"{time_info['full_mean']:.1f} Days")
    with t_card4:
        st.metric("Median Full Hire Duration", f"{time_info['full_median']:.1f} Days")
        
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.plotly_chart(time_info["fig_time_to_acceptance"], width="stretch")
        st.caption("Distribution of days between offer release and candidate acceptance decision.")
    with t_col2:
        st.plotly_chart(time_info["fig_application_to_acceptance"], width="stretch")
        st.caption("Distribution of total elapsed hiring days from initial application submission to formal offer acceptance.")
