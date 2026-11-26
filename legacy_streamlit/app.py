"""
Main Streamlit Application.
AI-Assisted Candidate Screening Pipeline with Recruitment Analytics.
Integrates semantic screening, live resume document upload, pipeline analytics,
predictive offer acceptance models, and responsible AI governance into one dashboard.
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Candidate Screening & Recruitment Analytics",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# APPLY CUSTOM CSS
# ==========================================

css_path = os.path.join("dashboard", "styles", "custom.css")

if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

# ==========================================
# IMPORT PROJECT MODULES
# ==========================================

from ai_module.resume_processing import (
    load_and_preprocess_cvs,
    load_and_preprocess_jobs
)

from ai_module.sentence_bert import (
    get_or_create_job_embeddings,
    get_or_create_cv_embeddings
)

from ai_module.similarity import TfidfBaselineMatcher

from ida_module.preprocessing import preprocess_recruitment_data
from ida_module.model_training import train_offer_models
from ida_module.evaluation import evaluate_all_models

from dashboard.pages.page1_overview import render_page1
from dashboard.pages.page2_screening import render_page2
from dashboard.pages.page3_upload import render_page3
from dashboard.pages.page4_analytics import render_page4
from dashboard.pages.page5_models import render_page5


# ==========================================
# LOAD AND CACHE DATASETS
# ==========================================

@st.cache_data(show_spinner="Loading and preprocessing datasets...")
def get_cached_datasets():

    cv_df = load_and_preprocess_cvs("data/cv.csv")
    job_df = load_and_preprocess_jobs("data/job.csv")
    matches_df = pd.read_csv("data/matches.csv")

    cleaned_ida_path = "outputs/cleaned_data/ida_cleaned_recruitment.csv"

    if os.path.exists(cleaned_ida_path):

        ida_df = pd.read_csv(cleaned_ida_path)

        audit_info = {
            "total_offers": int(ida_df["offered_at"].notnull().sum()),
            "offers_accepted_count": int(
                (ida_df["accepted"] == True).sum()
            ),
            "offers_rejected_count": int(
                (ida_df["accepted"] == False).sum()
            ),
            "excluded_anomaly_records": int(
                (ida_df["offer_date_anomaly"] == True).sum()
            )
        }

    else:
        ida_df, audit_info = preprocess_recruitment_data()

    return cv_df, job_df, matches_df, ida_df, audit_info


# ==========================================
# LOAD AND CACHE EMBEDDINGS
# ==========================================

@st.cache_resource(show_spinner="Loading semantic models and embeddings...")
def get_cached_embeddings(job_df, cv_df):

    job_embs = get_or_create_job_embeddings(job_df)
    cv_embs = get_or_create_cv_embeddings(cv_df)

    tfidf_matcher = TfidfBaselineMatcher(max_features=5000)

    corpus_texts = cv_df["clean_resume_text"].tolist()
    tfidf_matcher.fit_corpus(corpus_texts)

    return job_embs, cv_embs, tfidf_matcher


# ==========================================
# LOAD AND CACHE MODEL EVALUATION
# ==========================================

@st.cache_data(show_spinner="Loading predictive evaluation results...")
def get_cached_evaluation(ida_df):

    eval_json_path = "outputs/models/evaluation_results.json"

    if os.path.exists(eval_json_path):
        with open(eval_json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    train_out = train_offer_models(ida_df)

    results = evaluate_all_models(
        models_dict=train_out["models"],
        X_test=train_out["data"]["X_test"],
        y_test=train_out["data"]["y_test"],
        feature_names=train_out["all_feature_names"]
    )

    return results


# ==========================================
# SIDEBAR DESIGN
# ==========================================

def render_sidebar(df_cv, df_job, df_ida, audit_info):

    # Brand section
    st.sidebar.markdown("# 👥 Recruitment Suite")

    st.sidebar.caption(
        "AI Screening & Pipeline Analytics"
    )

    st.sidebar.divider()

    # Navigation section
    st.sidebar.markdown("### WORKSPACE")

    pages = [
        "Project Overview",
        "AI Resume Screening",
        "Upload Resume",
        "Recruitment Analytics",
        "Model Comparison"
    ]

    selected_page = st.sidebar.radio(
        "Navigation Menu",
        pages,
        index=0,
        label_visibility="collapsed"
    )

    st.sidebar.divider()

    # Compact Data Status section
    st.sidebar.markdown("**🟢 DATA STATUS**")

    st.sidebar.caption(
        f"📄 Resumes: **{len(df_cv):,}**  ·  "
        f"💼 Jobs: **{len(df_job):,}**"
    )

    st.sidebar.caption(
        f"📊 Applications: **{len(df_ida):,}**"
    )

    st.sidebar.caption(
        f"🤝 Offers: **{audit_info.get('total_offers', 0):,}**  ·  "
        f"⚠️ Anomalies: **{audit_info.get('excluded_anomaly_records', 0):,}**"
    )

    st.sidebar.divider()

    # Footer
    st.sidebar.caption(
        "AI Candidate Screening & Recruitment Analytics"
    )

    st.sidebar.caption(
        "Final College Demonstration Project"
    )

    return selected_page


# ==========================================
# MAIN APPLICATION
# ==========================================

def main():

    # Load project data
    df_cv, df_job, df_matches, df_ida, audit_info = (
        get_cached_datasets()
    )

    # Load AI models and embeddings
    job_embeddings, cv_embeddings, tfidf_matcher = (
        get_cached_embeddings(df_job, df_cv)
    )

    # Load model evaluation results
    evaluation_results = get_cached_evaluation(df_ida)

    # Render sidebar
    selected_page = render_sidebar(
        df_cv,
        df_job,
        df_ida,
        audit_info
    )

    # ======================================
    # PAGE NAVIGATION
    # ======================================

    if selected_page == "Project Overview":

        render_page1(
            df_cv,
            df_job,
            df_matches,
            df_ida,
            audit_info
        )

    elif selected_page == "AI Resume Screening":

        render_page2(
            df_cv,
            df_job,
            cv_embeddings,
            job_embeddings,
            tfidf_matcher
        )

    elif selected_page == "Upload Resume":

        render_page3(
            df_job,
            job_embeddings,
            df_cv,
            cv_embeddings
        )

    elif selected_page == "Recruitment Analytics":

        render_page4(df_ida)

    elif selected_page == "Model Comparison":

        render_page5(evaluation_results)


# ==========================================
# APPLICATION ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()