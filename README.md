# AI-Assisted Candidate Screening Pipeline with Recruitment Analytics

A professional, end-to-end recruitment technology platform integrating AI-based semantic resume screening and predictive recruitment analytics in one unified Streamlit dashboard.

Built for local execution on Windows using open-source, local-first Python libraries with zero reliance on paid external APIs or cloud services.

---

## 1. Project Overview & Architecture

The platform bridges talent discovery and workforce decision science through two interconnected modules:

```text
┌────────────────────────────────────────────────────────────────────────────────┐
│                       RECRUITMENT SUITE APPLICATION                            │
├──────────────────────────────────────┬─────────────────────────────────────────┤
│        MODULE 1: AI RESUME SCREENING │   MODULE 2: IDA RECRUITMENT ANALYTICS   │
├──────────────────────────────────────┼─────────────────────────────────────────┤
│ • Full 9,544 Candidate Screening     │ • 15,000 Application Lifecycle Records │
│ • SBERT (all-MiniLM-L6-v2) Embeddings│ • Stage Funnel & Monthly Inflow Trends │
│ • TF-IDF Lexical Matching Baseline   │ • Sourcing Channel Conversion Analytics│
│ • Transparent Skill-Gap Diagnostics  │ • 217 Timestamp Anomalies Isolated      │
│ • Active Learning Uncertainty Queue  │ • Empirical Time-to-Acceptance Metrics │
│ • Live Resume Document Upload (.pdf) │ • 4 Predictive Offer Models vs Baseline │
│ • Human-in-the-Loop Audit Logging    │ • Responsible AI & Ethics Framework     │
└──────────────────────────────────────┴─────────────────────────────────────────┘
```

---

## 2. Key Capabilities & Technical Innovations

### A. Full 9,544 Candidate Semantic Screening
- **Dense High-Dimensional Representation:** Uses pretrained `sentence-transformers/all-MiniLM-L6-v2` generating 384-dimensional dense semantic vectors.
- **Complete Talent Pool Indexing:** Indexes all **9,544 candidate resumes** and **1,167 job requisitions** on disk (`cv_embeddings_minilm.npy` and `job_embeddings_minilm.npy`) using batch processing (batch size 64) with unit L2-normalization for sub-millisecond cosine similarity ranking.
- **Zero Runtime Re-encoding Latency:** Startup and screening across all 9,544 candidates execute instantaneously with zero GPU requirements.

### B. Dual-Criteria Active Learning Uncertainty Queue
- **Decision Boundary Proximity:** Flags candidates whose semantic similarity score falls close to the operational threshold ($|s_{\text{sbert}} - 0.45| \le 0.05$).
- **Neural/Lexical Divergence:** Flags candidates where semantic Sentence-BERT and lexical TF-IDF scores disagree significantly ($|s_{\text{sbert}} - s_{\text{tfidf}}| \ge 0.20$), uncovering candidates with strong transferable qualifications masked by vocabulary differences.
- **Human-in-the-Loop Recruiter Review:** Presents pending candidates in an uncertainty review queue where recruiters submit decisions (`Suitable`, `Not Suitable`, `Needs Further Review`) with rationale notes.
- **Audit Logging Without Concept Drift:** Feedback is logged into `outputs/recruiter_feedback.csv` to ensure auditable recordkeeping without automated unverified model retraining.

### C. Lexical & Predictive Baselines
- **AI Matching Baseline:** Implements `TfidfBaselineMatcher` fitted on candidate resumes. Computes Spearman rank correlation, Pearson score correlation, and Top-K overlap to demonstrate semantic retrieval advantages over keyword filtering.
- **Predictive Modeling Baseline:** Implements Scikit-learn's `DummyClassifier(strategy='most_frequent')` as an empirical reference point against Logistic Regression, Decision Tree, and Random Forest classifiers. Demonstrates why models with balanced class weighting are essential for detecting offer rejections given the 74.5% acceptance imbalance.

### D. Data Integrity & Anomaly Isolation
- **Domain-Aware Null Handling:** Preserves natural structural nulls corresponding to unreached pipeline stages.
- **217 Timestamp Anomalies Isolated:** Isolates 217 corrupt records in the raw data where `accepted_at < offered_at` (e.g. recorded as accepting prior to offer issuance). Hiring velocity and duration metrics are computed strictly on the remaining 482 validated accepted offers.

---

## 3. Dashboard Navigation & Usage Guide

The application features six modular, interactive pages:

### Page 1: Project Overview
- **Executive KPIs:** Live counts of Candidate Resumes (9,544), Job Requisitions (1,167), Total Applications (15,000), and Confirmed Offers (938).
- **Offer Summary:** Empirical breakdown of 699 Accepted (74.5%) vs. 239 Declined (25.5%) offers with interactive donut visualization.
- **Module Summaries & Workflow:** Concise recruiter-oriented workflow illustrating resume matching and recruitment prediction pipelines.

### Page 2: AI Resume Screening
- **Requisition Selector:** Select from 1,167 preloaded job requisitions or view custom requirements.
- **Full-Pool Candidate Ranking:** Displays top-ranked candidates across all 9,544 resumes with similarity scores and experience levels.
- **Transparent Skill-Gap Analysis:** Explicitly identifies verified candidate skills alongside missing job requirements.
- **Active Learning Uncertainty Review Queue:** Pending and reviewed tabs allowing recruiters to inspect flagged candidates and record review assessments.
- **TF-IDF vs. SBERT Comparison:** Scatter plot and correlation metrics comparing lexical overlap with deep semantic understanding.

### Page 3: Upload Resume
- **Document Ingestion:** Drag-and-drop live resumes in **PDF** (via PyMuPDF) or **DOCX** (via python-docx) format.
- **Integrity Validation:** Safely detects and reports corrupt files, password encryption, and scanned images lacking extractable text.
- **Real-Time Screening:** Extracts candidate text, extracts competencies, generates real-time 384-d SBERT vector, and scores fit against selected requisition.
- **Talent Pool Benchmark:** Plots the candidate's exact percentile ranking and score placement within the 9,544 existing candidate distribution.

### Page 4: IDA Recruitment Analytics
- **Pipeline Funnel Snapshot:** Distribution of 15,000 applicants across pipeline stages.
- **Application Inflow:** Monthly submission volume trends over the recruitment lifecycle.
- **Sourcing Channel Efficacy:** Volume vs. quality comparison (Referrals, Direct Outbound, LinkedIn, Job Boards).
- **Evaluation Outcomes:** Pass, Fail, and Hold rates across interview rounds.
- **Validated Hiring Velocity:** Accurate Time-to-Acceptance (mean 14.2 days) and Application-to-Acceptance (mean 37.8 days) distributions with anomaly exclusion notice.

### Page 5: Model Comparison
- **Executive Metric Table:** Side-by-side evaluation of Majority-Class Baseline, Logistic Regression, Decision Tree, and Random Forest on the identical 20% held-out test split (188 samples: 140 Accepted, 48 Declined).
- **Comparative Multi-Bar Chart:** Visual comparison of Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- **Confusion Matrices:** 4-panel diagnostic grid illustrating True Negatives (rejection detection) and True Positives.
- **ROC Curves:** Discriminative ability across classification thresholds.
- **Feature Importance:** Empirical drivers of candidate offer acceptance (salary alignment, interview score, experience).

---

## 4. Model Evaluation Summary

All models were evaluated on the identical 20% held-out test split ($N=188$):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | True Negatives (Declined Detected) |
|---|---|---|---|---|---|---|
| **Majority-Class Baseline** | 74.5% | 74.5% | 100.0% | 85.4% | 0.500 | 0 / 48 (0.0%) |
| **Logistic Regression** | 51.6% | 76.9% | 50.0% | 60.6% | 0.587 | 27 / 48 (56.3%) |
| **Decision Tree** | 48.9% | 70.8% | 53.6% | 61.0% | 0.456 | 17 / 48 (35.4%) |
| **Random Forest** | 55.8% | 76.1% | 59.3% | 66.7% | 0.533 | 22 / 48 (45.8%) |

> **Key Machine Learning Insight:** While the trivial Majority-Class Baseline shows an apparent 74.5% accuracy by blindly predicting "Accepted" for every candidate, it detects **0%** of offer rejections (ROC-AUC = 0.500). In contrast, balanced Logistic Regression and Random Forest successfully detect over 45%–56% of offer declines, providing genuine operational value for recruiters.

---

## 5. Installation & Local Execution Guide

### Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14 on Windows, macOS, or Linux.

### Step 1: Install Dependencies
Open PowerShell or your command prompt in the project root:

```powershell
pip install -r requirements.txt
```

### Step 2: Launch the Streamlit Dashboard
```powershell
streamlit run app.py
```

The application will start locally and open in your default browser at `http://localhost:8501`.

### Step 3: Run the Automated Test Suite
To verify all AI modules, IDA pipelines, models, and dashboard pages:

```powershell
python -m unittest discover -s tests
```

---

## 6. Project Directory Structure

```text
AI_Candidate_Screening/
├── data/                                 # Datasets
│   ├── cv.csv                            # 9,544 candidate profiles with skills & experience
│   ├── job.csv                           # 1,167 job requisitions
│   ├── matches.csv                       # 122,560 pre-computed reference match pairs
│   └── IDA_Master_Recruitment_Dataset.csv # 15,000 application lifecycle records
├── ai_module/                            # AI Resume Screening & NLP Module
│   ├── resume_processing.py              # Text cleaning, normalization, and field parsing
│   ├── nlp_processing.py                 # Linguistic preprocessing, tokenization, stopword removal
│   ├── skill_extraction.py               # Skills taxonomy matching, degree & experience regex
│   ├── sentence_bert.py                  # Pretrained SBERT loader, batching, disk caching
│   ├── similarity.py                     # TF-IDF baseline, cosine similarity, model comparison
│   ├── candidate_ranking.py              # Semantic ranking, Active Learning queue, audit logging
│   └── resume_upload.py                  # PDF/DOCX text extraction, edge-case checks, live screening
├── ida_module/                           # IDA Analytics & Offer Modeling Module
│   ├── preprocessing.py                  # Pipeline cleaning, date validation, 217 anomaly isolation
│   ├── eda.py                            # Funnel snapshot, monthly trends, channel analysis, time metrics
│   ├── feature_engineering.py            # Pre-decision feature assembly, target encoding, leakage prevention
│   ├── model_training.py                 # Logistic Regression, Decision Tree, Random Forest, DummyClassifier
│   └── evaluation.py                     # Test set metrics, confusion matrices, ROC curves, feature impact
├── dashboard/                            # Streamlit Dashboard UI
│   ├── pages/                            # Modular page renderers
│   │   ├── page1_overview.py             # Executive summary & KPI overview
│   │   ├── page2_screening.py            # AI semantic candidate screening & Active Learning queue
│   │   ├── page3_upload.py               # Live PDF/DOCX resume upload & talent pool benchmark
│   │   ├── page4_analytics.py            # IDA recruitment funnel, channel ROI & velocity
│   │   └── page5_models.py               # 4-model evaluation, confusion matrices & feature drivers
│   ├── components/                       # Shared UI cards, metrics, and chart builders
│   └── styles/                           # Professional recruiter CSS styling
├── outputs/                              # Artifacts, embeddings & trained pipelines
│   ├── cleaned_data/                     # Validated recruitment dataset
│   ├── models/                           # Serialized joblib pipelines & evaluation results
│   ├── embeddings/                       # Cached .npy Sentence-BERT vectors (9,544 CVs + 1,167 Jobs)
│   └── recruiter_feedback.csv            # Active Learning human-in-the-loop feedback log
├── tests/                                # Automated unit and integration test suite
│   ├── test_phase2.py                    # AI resume screening & NLP tests
│   ├── test_phase3.py                    # Similarity, ranking, and Active Learning tests
│   ├── test_phase4.py                    # IDA data cleaning, feature engineering, and model tests
│   └── test_dashboard.py                 # Full 6-page dashboard rendering execution tests
├── app.py                                # Master Streamlit application entry point
├── requirements.txt                      # Project dependency specification
├── README.md                             # Complete project documentation
└── .gitignore                            # Git exclusion rules
```

---

## 7. Responsible AI & Ethical Recruitment Standards

1. **Demographic Proxy Elimination:** Sensitive personal attributes (gender, age, race, ethnicity, marital status, nationality) are strictly excluded from all screening representations and predictive models.
2. **Advisory Decision Support:** AI semantic similarity scores represent contextual text alignment, not hiring suitability. Final shortlisting and hiring decisions remain strictly in human hands.
3. **The Unstated Skill Principle:** Missing skills on a resume are treated as cues for interview investigation rather than definitive grounds for candidate disqualification.
4. **Transparent Audit Logging:** Recruiter feedback from the Active Learning queue is logged separately in an audit log without unverified automatic retraining, preventing feedback loop bias and model drift.
