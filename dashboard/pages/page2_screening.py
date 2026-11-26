"""
Page 2: AI Resume Screening & Candidate Ranking.
Enables recruiters to select job openings, rank candidate resumes using fresh
Sentence-BERT semantic similarity, inspect skill gaps, compare with TF-IDF baseline,
and log recruiter human feedback without automatic unverified retraining.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from ai_module.similarity import compute_cosine_similarity, compare_matching_models
from ai_module.skill_extraction import parse_skills_field, compute_skill_gap
from ai_module.candidate_ranking import (
    record_recruiter_feedback,
    load_recruiter_feedback,
    identify_uncertain_candidates
)


def render_page2(df_cv, df_job, cv_embeddings, job_embeddings, tfidf_matcher=None):
    """
    Renders AI Resume Screening page.
    """
    st.markdown("<h2 style='margin-bottom: 0px;'>AI Resume Screening & Candidate Ranking</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #475569; margin-bottom: 20px;'>Semantically rank applicants against job requisitions, diagnose skill gaps, and log audit feedback.</p>", unsafe_allow_html=True)
    
    # 1. Job Requisition Selector
    st.markdown("<div class='section-header'>1. Select Target Job Opening</div>", unsafe_allow_html=True)
    
    # Filter by category if desired
    categories = ["All Categories"] + sorted([c for c in df_job["category"].dropna().unique().tolist() if c])
    selected_cat = st.selectbox("Filter Job Openings by Role Category:", categories, index=0)
    
    if selected_cat != "All Categories":
        filtered_jobs = df_job[df_job["category"] == selected_cat].reset_index(drop=True)
    else:
        filtered_jobs = df_job.reset_index(drop=True)
        
    job_options = [
        f"ID: {r.job_id} | {r.job_title} ({r.category}) – Req: {int(r.years_required)} yrs" 
        for _, r in filtered_jobs.iterrows()
    ]
    
    selected_job_idx = st.selectbox("Select Requisition:", range(len(job_options)), format_func=lambda i: job_options[i])
    selected_job = filtered_jobs.iloc[selected_job_idx]
    
    # Find original index in df_job to match embedding row
    orig_job_idx = df_job[df_job["job_id"] == selected_job["job_id"]].index[0]
    selected_job_embedding = job_embeddings[orig_job_idx]
    
    # Display Job Overview Card
    job_skills = parse_skills_field(selected_job.get("clean_skills", []))
    with st.expander("Requisition Details & Required Competencies", expanded=True):
        j_col1, j_col2 = st.columns([3, 1])
        with j_col1:
            st.markdown(f"**Job Title:** {selected_job['job_title']} | **Category:** {selected_job['category']}")
            st.markdown(f"**Description Summary:** {selected_job['job_description'][:350]}...")
        with j_col2:
            st.metric("Experience Required", f"{int(selected_job['years_required'])} Years")
            st.metric("Total Competencies", f"{len(job_skills)}")
            
        st.markdown("**Required Skills Taxonomy:**")
        if job_skills:
            st.markdown(" ".join([f"<span class='badge-pill'>{s}</span>" for s in job_skills]), unsafe_allow_html=True)
        else:
            st.write("No explicit skill keywords tagged in requisition.")

    # 2. Candidate Semantic Ranking
    st.markdown("<div class='section-header'>2. Ranked Applicants (Sentence-BERT Semantic Matching)</div>", unsafe_allow_html=True)
    
    # Calculate similarities against active candidate subset
    active_cv_count = len(cv_embeddings)
    active_cv_df = df_cv.iloc[:active_cv_count].copy()
    
    sbert_similarities = compute_cosine_similarity(selected_job_embedding, cv_embeddings)
    active_cv_df["similarity_score"] = np.round(sbert_similarities, 4)
    
    # Calculate TF-IDF baseline scores if matcher available
    tfidf_scores = None
    if tfidf_matcher is not None and tfidf_matcher.is_fitted:
        job_query_text = selected_job.get("combined_job_text", "")
        tfidf_scores = tfidf_matcher.match(job_query_text)[:active_cv_count]
        active_cv_df["tfidf_score"] = np.round(tfidf_scores, 4)
        
    # Sort by semantic similarity
    ranked_df = active_cv_df.sort_values("similarity_score", ascending=False).reset_index(drop=True)
    
    # Precompute skill gaps for top candidates
    top_n = st.slider("Number of Top Candidates to Display:", min_value=5, max_value=30, value=15, step=5)
    top_candidates = ranked_df.head(top_n).copy()
    
    gap_records = []
    for _, cand in top_candidates.iterrows():
        cand_skills = cand.get("parsed_skills", [])
        gap = compute_skill_gap(cand_skills, job_skills)
        gap_records.append({
            "matching_skills_str": ", ".join(gap["matching_skills"]) if gap["matching_skills"] else "None",
            "missing_skills_str": ", ".join(gap["missing_skills"]) if gap["missing_skills"] else "None",
            "skill_coverage_pct": gap["coverage_percentage"],
            "matching_count": gap["match_count"],
            "missing_count": len(gap["missing_skills"])
        })
    gap_df = pd.DataFrame(gap_records)
    for col in gap_df.columns:
        top_candidates[col] = gap_df[col].values
        
    top_candidates["Rank"] = range(1, len(top_candidates) + 1)
    
    # Recruiter display table
    table_cols = [
        "Rank", "candidate_id", "similarity_score", "skill_coverage_pct",
        "years_experience", "target_position", "matching_skills_str", "missing_skills_str"
    ]
    if "tfidf_score" in top_candidates.columns:
        table_cols.insert(3, "tfidf_score")
        
    display_table = top_candidates[table_cols].rename(columns={
        "candidate_id": "Candidate ID",
        "similarity_score": "Semantic Match",
        "tfidf_score": "TF-IDF Score",
        "skill_coverage_pct": "Skill Coverage %",
        "years_experience": "Exp (Yrs)",
        "target_position": "Current/Target Role",
        "matching_skills_str": "Verified Skills",
        "missing_skills_str": "Skill Deficiencies"
    })
    
    st.dataframe(
        display_table,
        width="stretch",
        hide_index=True,
        column_config={
            "Semantic Match": st.column_config.ProgressColumn(
                "Semantic Match",
                min_value=0.0,
                max_value=1.0,
                format="%.3f"
            ),
            "Skill Coverage %": st.column_config.NumberColumn(
                "Skill Coverage %",
                format="%.1f%%"
            )
        }
    )
    
    st.caption("Note: Semantic Match represents high-dimensional textual relevance between the resume and job requisition, not a statistical guarantee of hiring.")

    # 3. Candidate Profile Inspector & Skill-Gap Analysis
    st.markdown("<div class='section-header'>3. Candidate Profile Inspector & Skill-Gap Diagnostics</div>", unsafe_allow_html=True)
    
    inspect_col1, inspect_col2 = st.columns([1, 2])
    with inspect_col1:
        selected_cand_rank = st.selectbox(
            "Inspect Candidate by Rank:",
            range(1, len(top_candidates) + 1),
            format_func=lambda r: f"Rank {r}: Candidate #{top_candidates.iloc[r-1]['candidate_id']} (Match: {top_candidates.iloc[r-1]['similarity_score']:.3f})"
        )
        cand_row = top_candidates.iloc[selected_cand_rank - 1]
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Candidate #{cand_row['candidate_id']}</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #1E293B;">{cand_row.get('target_position', 'Applicant')}</div>
            <div style="margin-top: 8px;">
                <b>Experience:</b> {cand_row.get('years_experience', 0)} Years<br>
                <b>Semantic Match Score:</b> {cand_row['similarity_score']:.3f}<br>
                <b>Skill Coverage:</b> {cand_row['skill_coverage_pct']:.1f}%<br>
                <b>Education:</b> {cand_row.get('education_text', 'Not specified')}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with inspect_col2:
        cand_skills = cand_row.get("parsed_skills", [])
        gap_info = compute_skill_gap(cand_skills, job_skills)
        
        st.markdown("**Skill Alignment Breakdown:**")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.metric("Matching Skills", len(gap_info["matching_skills"]))
        with sc2:
            st.metric("Missing Skills", len(gap_info["missing_skills"]))
        with sc3:
            st.metric("Skill Coverage", f"{gap_info['coverage_percentage']:.1f}%")
            
        st.markdown("<p style='margin-bottom: 4px; font-weight: 600; color: #047857;'>Matching Competencies:</p>", unsafe_allow_html=True)
        if gap_info["matching_skills"]:
            st.markdown(" ".join([f"<span class='badge-match'>{s}</span>" for s in gap_info["matching_skills"]]), unsafe_allow_html=True)
        else:
            st.write("None of the job's tagged skills were explicitly detected in this resume.")
            
        st.markdown("<p style='margin-top: 10px; margin-bottom: 4px; font-weight: 600; color: #B91C1C;'>Skill Deficiencies (Potential Gaps):</p>", unsafe_allow_html=True)
        if gap_info["missing_skills"]:
            st.markdown(" ".join([f"<span class='badge-missing'>{s}</span>" for s in gap_info["missing_skills"]]), unsafe_allow_html=True)
        else:
            st.write("100% skill coverage; candidate mentions all required competencies.")
            
        st.caption("Responsible AI Note: An unmentioned skill in a resume does not definitively prove absence of actual ability. It highlights areas for recruiter interview probing.")

    # Full Resume Text Inspector
    with st.expander(f"Full Extracted Resume Text – Candidate #{cand_row['candidate_id']}"):
        st.text_area("Resume Content:", cand_row.get("resume_text", ""), height=220, disabled=True)

    # 4. Human Recruiter Feedback Form
    st.markdown("<div class='section-header'>4. Recruiter Match Review (Audit Log)</div>", unsafe_allow_html=True)
    st.markdown("<div class='info-callout'>Recruiter decisions are recorded in a separate audit log to track review history. Feedback is never used to automatically retrain screening models without verified human validation.</div>", unsafe_allow_html=True)
    
    with st.form("recruiter_feedback_form"):
        fb_c1, fb_c2, fb_c3 = st.columns([1, 1, 2])
        with fb_c1:
            decision = st.selectbox("Recruiter Assessment:", ["Suitable", "Not Suitable", "Needs Further Review"])
        with fb_c2:
            st.markdown(f"**Target:** #{cand_row['candidate_id']}<br>**Score:** {cand_row['similarity_score']:.3f}", unsafe_allow_html=True)
        with fb_c3:
            notes = st.text_input("Review Notes / Interview Instructions:", placeholder="e.g. Verify deep learning project experience in phone screen")
            
        submitted = st.form_submit_button("Record Recruiter Review")
        if submitted:
            record_recruiter_feedback(
                candidate_id=cand_row["candidate_id"],
                job_id=selected_job["job_id"],
                job_title=selected_job["job_title"],
                decision=decision,
                relevance_score=cand_row["similarity_score"],
                notes=notes
            )
            st.success(f"Review recorded for Candidate #{cand_row['candidate_id']} ({decision}) in audit trail.")

    # 5. Active Learning – Uncertainty Review Queue
    st.markdown("<div class='section-header'>5. Active Learning – Uncertainty Review Queue</div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color: #475569; margin-bottom: 15px;'>"
        "Human-in-the-Loop review queue prioritizing candidate resumes where the AI screening decision is ambiguous. "
        "Flags candidate profiles adjacent to the decision threshold (0.45) or showing significant divergence between neural semantic and keyword match scores."
        "</p>",
        unsafe_allow_html=True
    )
    
    uncertain_data = identify_uncertain_candidates(
        candidates_df=active_cv_df,
        sbert_similarities=sbert_similarities,
        job_id=selected_job["job_id"],
        tfidf_scores=active_cv_df["tfidf_score"].values if "tfidf_score" in active_cv_df.columns else None,
        decision_threshold=0.45,
        uncertainty_margin=0.05,
        disagreement_threshold=0.20,
        top_n=30
    )
    
    pending_df = uncertain_data["pending_candidates"]
    reviewed_df = uncertain_data["reviewed_candidates"]
    total_awaiting = uncertain_data["total_awaiting_review"]
    total_reviewed = uncertain_data["total_reviewed"]
    
    # Uncertainty Queue Status Metrics
    q_col1, q_col2, q_col3 = st.columns(3)
    with q_col1:
        st.metric("Candidates Awaiting Review", total_awaiting)
    with q_col2:
        st.metric("Candidates Already Reviewed", total_reviewed)
    with q_col3:
        st.metric("Decision Boundary Threshold", f"{uncertain_data['decision_threshold']:.2f} (±{uncertain_data['uncertainty_margin']:.2f})")
        
    queue_tab1, queue_tab2 = st.tabs(["Priority Review Queue (Pending)", "Reviewed Candidates History"])
    
    with queue_tab1:
        if not pending_df.empty:
            st.markdown(f"**Top {len(pending_df)} Uncertain Cases (Ranked by Ambiguity First):**")
            
            disp_cols = ["candidate_id", "candidate_name", "similarity_score"]
            if "tfidf_score" in pending_df.columns and pending_df["tfidf_score"].notna().any():
                disp_cols.append("tfidf_score")
            disp_cols.extend(["uncertainty_reason", "review_status"])
            
            queue_display = pending_df[disp_cols].rename(columns={
                "candidate_id": "Candidate ID",
                "candidate_name": "Candidate Name / Profile",
                "similarity_score": "SBERT Score",
                "tfidf_score": "TF-IDF Score",
                "uncertainty_reason": "Reason Flagged Uncertain",
                "review_status": "Review Status"
            })
            
            st.dataframe(
                queue_display,
                width="stretch",
                hide_index=True,
                column_config={
                    "SBERT Score": st.column_config.NumberColumn(format="%.3f"),
                    "TF-IDF Score": st.column_config.NumberColumn(format="%.3f")
                }
            )
            
            st.markdown("<p style='font-weight: 600; margin-top: 15px; margin-bottom: 6px;'>Human-in-the-Loop Feedback Submission:</p>", unsafe_allow_html=True)
            u_options = [
                f"{r['candidate_id']} | SBERT: {r['similarity_score']:.3f} | {r.get('target_position', 'Applicant')}"
                for _, r in pending_df.iterrows()
            ]
            selected_u_idx = st.selectbox(
                "Select Uncertain Candidate to Review:",
                range(len(u_options)),
                format_func=lambda i: u_options[i],
                key="select_uncertain_queue_candidate"
            )
            u_cand = pending_df.iloc[selected_u_idx]
            
            with st.form("uncertain_candidate_review_form"):
                u_c1, u_c2, u_c3 = st.columns([1, 1, 2])
                with u_c1:
                    u_decision = st.selectbox(
                        "Assessment:",
                        ["Suitable", "Not Suitable", "Needs Further Review"],
                        key="u_review_decision"
                    )
                with u_c2:
                    st.markdown(
                        f"**Candidate:** #{u_cand['candidate_id']}<br>"
                        f"**Role:** {u_cand.get('target_position', 'Applicant')}<br>"
                        f"**Exp:** {u_cand.get('years_experience', 0)} Years",
                        unsafe_allow_html=True
                    )
                with u_c3:
                    u_notes = st.text_input(
                        "Evaluation Notes:",
                        placeholder="e.g. Verify depth of hands-on skills during technical screen",
                        key="u_review_notes"
                    )
                
                u_submit = st.form_submit_button("Record Active Learning Feedback")
                if u_submit:
                    record_recruiter_feedback(
                        candidate_id=u_cand["candidate_id"],
                        job_id=selected_job["job_id"],
                        job_title=selected_job["job_title"],
                        decision=u_decision,
                        relevance_score=u_cand["similarity_score"],
                        notes=u_notes
                    )
                    st.success(f"Feedback recorded for Candidate #{u_cand['candidate_id']} as '{u_decision}'. Candidate moved to reviewed list.")
                    st.rerun()
        else:
            st.info("No pending candidates in the uncertainty review queue for this job opening. All identified cases have been reviewed.")
            
    with queue_tab2:
        if not reviewed_df.empty:
            rev_cols = ["candidate_id", "candidate_name", "similarity_score", "review_status", "review_timestamp", "review_notes"]
            rev_display = reviewed_df[[c for c in rev_cols if c in reviewed_df.columns]].rename(columns={
                "candidate_id": "Candidate ID",
                "candidate_name": "Candidate Name / Profile",
                "similarity_score": "SBERT Score",
                "review_status": "Recruiter Assessment",
                "review_timestamp": "Reviewed At",
                "review_notes": "Review Notes"
            })
            st.dataframe(rev_display, width="stretch", hide_index=True)
        else:
            st.info("No candidates have been reviewed yet for this requisition.")

    # 6. TF-IDF vs Sentence-BERT Model Comparison
    st.markdown("<div class='section-header'>6. Method Comparison: TF-IDF Keyword Baseline vs. Sentence-BERT Semantic Matching</div>", unsafe_allow_html=True)
    
    if tfidf_scores is not None:
        comp = compare_matching_models(
            tfidf_scores=active_cv_df["tfidf_score"].values,
            sbert_scores=active_cv_df["similarity_score"].values,
            candidate_ids=active_cv_df["candidate_id"].tolist(),
            top_k=top_n
        )
        
        cmp_col1, cmp_col2, cmp_col3, cmp_col4 = st.columns(4)
        with cmp_col1:
            st.metric("Spearman Rank Correlation", f"{comp['spearman_rank_correlation']:.3f}")
        with cmp_col2:
            st.metric("Pearson Score Correlation", f"{comp['pearson_score_correlation']:.3f}")
        with cmp_col3:
            st.metric(f"Top-{top_n} Candidate Overlap", f"{comp['top_k_overlap_count']} / {top_n}")
        with cmp_col4:
            st.metric("Top-K Agreement", f"{comp['top_k_overlap_percentage']:.1f}%")
            
        # Scatter correlation plot
        sample_df = active_cv_df.sample(min(300, len(active_cv_df)), random_state=42)
        fig_scatter = px.scatter(
            sample_df,
            x="tfidf_score",
            y="similarity_score",
            hover_data=["candidate_id", "years_experience"],
            title=f"Score Comparison: Keyword TF-IDF vs. Semantic Sentence-BERT (N={len(sample_df)} Sample)",
            labels={"tfidf_score": "TF-IDF Lexical Cosine Similarity", "similarity_score": "Sentence-BERT Semantic Cosine Similarity"},
            color="similarity_score",
            color_continuous_scale="Viridis"
        )
        fig_scatter.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_scatter, width="stretch")
        
        with st.expander("Methodological Comparison & Observations", expanded=False):
            for obs in comp["observations"]:
                st.markdown(f"- {obs}")
    else:
        st.info("TF-IDF matcher baseline is ready. Fit corpus to visualize live comparative correlation.")
