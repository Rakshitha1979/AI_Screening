"""
Page 3: Resume Upload & Real-Time Candidate Screening.
Ingests PDF and DOCX files, extracts text, handles edge cases (password protection,
scanned documents), extracts competencies, and computes real-time semantic fit
and benchmark placement against existing applicants.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from ai_module.resume_upload import extract_text_from_file, screen_uploaded_resume
from ai_module.skill_extraction import parse_skills_field


def render_page3(df_job, job_embeddings, df_cv=None, cv_embeddings=None):
    """
    Renders Resume Upload page.
    """
    st.markdown("<h2 style='margin-bottom: 0px;'>Upload Resume & Live Screening</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #475569; margin-bottom: 20px;'>Ingest an external candidate's resume (PDF or DOCX) for instant semantic evaluation and skill gap diagnostics.</p>", unsafe_allow_html=True)
    
    # Target Job Selector
    st.markdown("<div class='section-header'>1. Select Target Job Opening</div>", unsafe_allow_html=True)
    job_options = [
        f"ID: {r.job_id} | {r.job_title} ({r.category}) – Req: {int(r.years_required)} yrs" 
        for _, r in df_job.iterrows()
    ]
    selected_job_idx = st.selectbox("Select Target Position:", range(len(job_options)), format_func=lambda i: job_options[i])
    selected_job = df_job.iloc[selected_job_idx]
    selected_job_embedding = job_embeddings[selected_job_idx]
    job_skills = parse_skills_field(selected_job.get("clean_skills", []))
    
    with st.expander("Target Requisition Details", expanded=False):
        st.markdown(f"**Title:** {selected_job['job_title']} | **Category:** {selected_job['category']}")
        st.markdown(f"**Description:** {selected_job['job_description'][:300]}...")
        st.markdown(f"**Required Experience:** {int(selected_job['years_required'])} Years")
        st.markdown("**Required Skills:** " + " ".join([f"<span class='badge-pill'>{s}</span>" for s in job_skills]), unsafe_allow_html=True)
        
    # File Uploader
    st.markdown("<div class='section-header'>2. Ingest Resume Document</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload candidate resume (.pdf or .docx):",
        type=["pdf", "docx"],
        help="Upload standard PDF or Word DOCX resume documents. Scanned image PDFs without text layers and encrypted files are detected."
    )
    
    if uploaded_file is None:
        st.info("Please upload a PDF or DOCX file to initiate automated candidate evaluation.")
        # Provide sample test button for quick recruiter evaluation demo
        st.markdown("---")
        st.markdown("**Quick Demo Mode:** Test live screening using a pre-configured senior engineer profile:")
        if st.button("Load Pre-Configured Sample Resume for Demonstration"):
            sample_text = (
                "SENIOR SOFTWARE ENGINEER & ML SPECIALIST\n"
                "Summary: Over 6 years of experience building high-throughput distributed systems and machine learning workflows.\n"
                "Core Competencies: Python, Machine Learning, Deep Learning, PyTorch, Docker, Kubernetes, AWS, PostgreSQL, REST API, Agile.\n"
                "Education: Master of Science (M.S.) in Computer Science, B.Tech in Information Technology.\n"
                "Experience: Led development of real-time recommendation engines on AWS using Python and PyTorch. Containerized microservices using Docker and Kubernetes."
            )
            # Create in-memory evaluation
            from ai_module.sentence_bert import encode_single_resume
            from ai_module.similarity import compute_cosine_similarity
            from ai_module.skill_extraction import extract_skills_from_text, extract_education, extract_years_experience, compute_skill_gap
            
            cand_skills = extract_skills_from_text(sample_text)
            cand_edu = extract_education(sample_text)
            cand_exp = extract_years_experience(sample_text)
            gap = compute_skill_gap(cand_skills, job_skills)
            
            cand_emb = encode_single_resume(sample_text)
            sim = float(compute_cosine_similarity(selected_job_embedding, cand_emb.reshape(1, -1))[0])
            
            st.session_state["demo_result"] = {
                "success": True,
                "filename": "Sample_Candidate_Profile.pdf",
                "is_uploaded_candidate": True,
                "cleaned_text": sample_text,
                "extracted_skills": cand_skills,
                "extracted_education": cand_edu,
                "extracted_years_experience": cand_exp,
                "similarity_score": round(sim, 4),
                "skill_gap": gap,
                "comparison": {}
            }
            
    # Process uploaded file or demo result
    result = None
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        filename = uploaded_file.name
        with st.spinner("Extracting document text and calculating semantic profile fit..."):
            result = screen_uploaded_resume(
                file_bytes=file_bytes,
                filename=filename,
                job_row=selected_job,
                job_embedding=selected_job_embedding
            )
    elif "demo_result" in st.session_state:
        result = st.session_state["demo_result"]
        
    if result is not None:
        if not result["success"]:
            st.error(f"Extraction Error: {result['error']}")
            return
            
        st.markdown("<div class='section-header'>3. Candidate Evaluation Results</div>", unsafe_allow_html=True)
        st.markdown(f"<span class='badge-pill' style='background: #3B82F6; color: white;'>Live Uploaded Candidate: {result['filename']}</span>", unsafe_allow_html=True)
        
        # Key KPI cards
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">Semantic Match Score</div>
                <div class="metric-card-value" style="color: #2563EB;">{result['similarity_score']:.3f}</div>
                <div class="metric-card-sub">Contextual similarity to requisition</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">Skill Coverage</div>
                <div class="metric-card-value">{result['skill_gap']['coverage_percentage']:.1f}%</div>
                <div class="metric-card-sub">{result['skill_gap']['match_count']} of {result['skill_gap']['required_count']} skills matched</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">Extracted Experience</div>
                <div class="metric-card-value">{result['extracted_years_experience']:.1f} Yrs</div>
                <div class="metric-card-sub">Required: {int(selected_job['years_required'])} Yrs</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            edu_str = ", ".join(result['extracted_education']) if result['extracted_education'] else "Self-Stated"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">Highest Degree</div>
                <div class="metric-card-value" style="font-size: 1.25rem;">{edu_str}</div>
                <div class="metric-card-sub">Credential detection</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Skill-gap visualization
        st.markdown("<div class='section-header'>Skill-Gap Diagnostics</div>", unsafe_allow_html=True)
        sg_col1, sg_col2 = st.columns(2)
        with sg_col1:
            st.markdown("<p style='font-weight: 600; color: #047857;'>Matching Competencies Detected in Resume:</p>", unsafe_allow_html=True)
            matching = result["skill_gap"]["matching_skills"]
            if matching:
                st.markdown(" ".join([f"<span class='badge-match'>{s}</span>" for s in matching]), unsafe_allow_html=True)
            else:
                st.write("No direct overlap between resume skills and job requirements.")
                
            st.markdown("<p style='margin-top: 14px; font-weight: 600; color: #B91C1C;'>Skill Deficiencies (Missing from Resume):</p>", unsafe_allow_html=True)
            missing = result["skill_gap"]["missing_skills"]
            if missing:
                st.markdown(" ".join([f"<span class='badge-missing'>{s}</span>" for s in missing]), unsafe_allow_html=True)
            else:
                st.write("Zero identified deficiencies; candidate covers all specified job skills.")
                
        with sg_col2:
            # Doughnut chart of skills
            labels = ["Matching Skills", "Missing Skills"]
            values = [len(matching), len(missing)]
            if sum(values) > 0:
                fig_doughnut = px.pie(
                    names=labels,
                    values=values,
                    title="Skill Alignment Ratio",
                    color=labels,
                    color_discrete_map={"Matching Skills": "#059669", "Missing Skills": "#DC2626"},
                    hole=0.55
                )
                fig_doughnut.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_doughnut, width="stretch")
                
        # Benchmark Comparison with Existing Candidates
        if cv_embeddings is not None and df_cv is not None:
            st.markdown("<div class='section-header'>Candidate Benchmark Placement</div>", unsafe_allow_html=True)
            from ai_module.similarity import compute_cosine_similarity
            existing_sims = compute_cosine_similarity(selected_job_embedding, cv_embeddings)
            
            percentile = float((existing_sims < result["similarity_score"]).mean() * 100.0)
            placement = int((existing_sims >= result["similarity_score"]).sum()) + 1
            
            b1, b2, b3 = st.columns(3)
            with b1:
                st.metric("Pool Placement Rank", f"#{placement} of {len(existing_sims)}")
            with b2:
                st.metric("Relevance Percentile", f"{percentile:.1f}%")
            with b3:
                st.metric("Pool Median Score", f"{np.median(existing_sims):.3f}")
                
            # Distribution histogram with candidate marker
            fig_hist = px.histogram(
                x=existing_sims,
                nbins=30,
                title=f"Relevance Score Distribution for Job '{selected_job['job_title']}'",
                labels={"x": "Semantic Relevance Score"},
                color_discrete_sequence=["#94A3B8"]
            )
            fig_hist.add_vline(
                x=result["similarity_score"],
                line_width=3,
                line_dash="dash",
                line_color="#2563EB",
                annotation_text=f"Uploaded Candidate ({result['similarity_score']:.3f})",
                annotation_position="top right"
            )
            fig_hist.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_hist, width="stretch")

        with st.expander("View Sanitized Extracted Document Text", expanded=False):
            st.text_area("Extracted Resume Text:", result["cleaned_text"], height=200, disabled=True)
