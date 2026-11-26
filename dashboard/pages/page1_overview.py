"""
Page 1: Executive Overview & System Dashboard.
Delivers a concise, high-level operational summary for recruiters and evaluators:
verified KPI metrics, offer outcomes, module capabilities, and pipeline workflow.
"""

import pandas as pd
import streamlit as st
import plotly.graph_objects as go


def render_page1(df_cv, df_job, df_matches, df_ida, audit_info):
    """
    Renders the executive Project Overview page.
    """
    # 1. Header
    st.markdown("<h1 style='margin-bottom: 2px;'>AI-Assisted Candidate Screening Pipeline</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.15rem; color: #475569; margin-top: 0px; margin-bottom: 20px;'>Integrated Resume Matching & Recruitment Analytics</p>", unsafe_allow_html=True)

    # 2. Four KPI Cards (Actual Dataframe Counts)
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Candidate Resumes</div>
            <div class="metric-card-value">{len(df_cv):,}</div>
            <div class="metric-card-sub">Indexed candidate profiles</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Job Descriptions</div>
            <div class="metric-card-value">{len(df_job):,}</div>
            <div class="metric-card-sub">Active & historical openings</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Total Applications</div>
            <div class="metric-card-value">{len(df_ida):,}</div>
            <div class="metric-card-sub">Recruitment pipeline records</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        total_offers = audit_info.get("total_offers", 938)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Confirmed Offers</div>
            <div class="metric-card-value">{total_offers:,}</div>
            <div class="metric-card-sub">Formal job offers recorded</div>
        </div>
        """, unsafe_allow_html=True)

    # 3. Offer Summary & Compact Visual Comparison
    accepted_count = audit_info.get("offers_accepted_count", 699)
    declined_count = audit_info.get("offers_rejected_count", 239)
    accepted_pct = (accepted_count / total_offers * 100.0) if total_offers > 0 else 0.0
    declined_pct = (declined_count / total_offers * 100.0) if total_offers > 0 else 0.0

    st.markdown("<div class='section-header'>Offer Decision Summary</div>", unsafe_allow_html=True)

    off_col1, off_col2 = st.columns([1, 2])
    with off_col1:
        st.markdown(f"""
        <div class="metric-card" style="height: 100%;">
            <div class="metric-card-title">Offer Outcomes</div>
            <div style="margin-top: 10px; font-size: 1.05rem;">
                <span style="color: #059669; font-weight: 700;">● Accepted:</span> <b>{accepted_count:,}</b> ({accepted_pct:.1f}%)<br>
                <span style="color: #DC2626; font-weight: 700;">● Declined:</span> <b>{declined_count:,}</b> ({declined_pct:.1f}%)<br>
            </div>
            <div class="metric-card-sub" style="margin-top: 10px;">
                Total Verified Offers: <b>{total_offers:,}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    
    with off_col2:

        st.markdown("**Offer Distribution**")

        fig_offers = go.Figure()

        fig_offers.add_trace(go.Bar(
            y=["Offers"],
            x=[accepted_count],
            name="Accepted",
            orientation="h",
            marker=dict(
                color="#10B981",
                line=dict(color="#FFFFFF", width=3)
            ),
            text=[f"{accepted_count:,}  |  {accepted_pct:.1f}%"],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(
                color="white",
                size=13,
                family="Arial"
            ),
            hovertemplate=(
                "<b>Accepted Offers</b><br>"
                "Count: %{x:,}<br>"
                f"Share: {accepted_pct:.1f}%"
                "<extra></extra>"
            )
        ))

        fig_offers.add_trace(go.Bar(
            y=["Offers"],
            x=[declined_count],
            name="Declined",
            orientation="h",
            marker=dict(
                color="#EF4444",
                line=dict(color="#FFFFFF", width=3)
            ),
            text=[f"{declined_count:,}  |  {declined_pct:.1f}%"],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(
                color="white",
                size=13,
                family="Arial"
            ),
            hovertemplate=(
                "<b>Declined Offers</b><br>"
                "Count: %{x:,}<br>"
                f"Share: {declined_pct:.1f}%"
                "<extra></extra>"
            )
        ))

        fig_offers.update_layout(
            barmode="stack",
            height=175,
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            margin=dict(l=12, r=12, t=35, b=15),
            bargap=0.45,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.05,
                xanchor="center",
                x=0.5,
                font=dict(size=12, color="#475569")
            ),
            xaxis=dict(
                range=[0, max(total_offers, 1)],
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                fixedrange=True
            ),
            yaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                fixedrange=True
            ),
            hoverlabel=dict(
                bgcolor="#FFFFFF",
                bordercolor="#CBD5E1",
                font=dict(color="#1E293B", size=13)
            )
        )

        st.plotly_chart(
            fig_offers,
            width="stretch",
            config={"displayModeBar": False}
        )


    # 4. Two Side-by-Side Module Sections
    st.markdown("<div class='section-header'>Core Platform Modules</div>", unsafe_allow_html=True)
    mod_col1, mod_col2 = st.columns(2)

    with mod_col1:
        st.markdown("""
        <div class="metric-card" style="border-left: 5px solid #2563EB; height: 100%;">
            <div class="metric-card-title">Module 1: AI Resume Screening</div>
            <div style="font-size: 0.95rem; color: #1E293B; margin-top: 6px; line-height: 1.6;">
                <b>Key Capabilities:</b>
            <ul style="margin-top: 6px; padding-left: 20px;">
               <li><b>Candidate Ranking:</b> Ranks candidates by relevance to the selected job.</li>
               <li><b>Semantic Matching:</b> Compares skills, experience and job requirements.</li>
               <li><b>Skill-Gap Analysis:</b> Identifies matching and missing skills.</li>
               <li><b>Resume Upload:</b> Supports PDF and DOCX resumes for screening.</li>
            </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with mod_col2:
        st.markdown("""
        <div class="metric-card" style="border-left: 5px solid #0D9488; height: 100%;">
            <div class="metric-card-title">Module 2: Recruitment Analytics & Prediction</div>
            <div style="font-size: 0.95rem; color: #1E293B; margin-top: 6px; line-height: 1.6;">
                <b>Key Capabilities:</b>
            <ul style="margin-top: 6px; padding-left: 20px;">
                <li><b>Recruitment Funnel:</b> Shows candidate counts across recruitment stages.</li>
                <li><b>Sourcing Trends:</b> Compares application volume and offer outcomes by source.</li>
                <li><b>Interview Outcomes:</b> Summarizes pass, fail and hold results.</li>
                <li><b>Offer Prediction:</b> Estimates offer acceptance likelihood using relevant recruitment data.</li>
            </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 5. Compact End-to-End Workflow
    st.markdown("<div class='section-header'>System Workflow</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="display: flex; flex-direction: column; gap: 12px; margin-top: 6px;">
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px 18px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
            <span style="font-weight: 600; color: #2563EB;">AI Resume Screening Flow:</span>
            <span style="color: #334155;"><b>Resume + Job Description</b></span>
            <span style="color: #64748B;">➔</span>
            <span style="color: #334155;"><b>AI Screening</b></span>
            <span style="color: #64748B;">➔</span>
            <span style="color: #334155;"><b>Candidate Ranking</b></span>
        </div>
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px 18px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
            <span style="font-weight: 600; color: #0D9488;">Recruitment Analytics Flow:</span>
            <span style="color: #334155;"><b>Recruitment Data</b></span>
            <span style="color: #64748B;">➔</span>
            <span style="color: #334155;"><b>Analytics</b></span>
            <span style="color: #64748B;">➔</span>
            <span style="color: #334155;"><b>Offer Prediction</b></span>
        </div>
    </div>
    """, unsafe_allow_html=True)