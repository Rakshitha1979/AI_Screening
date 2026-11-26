# AI-Assisted Candidate Screening Pipeline with Recruitment Analytics

A professional, end-to-end recruitment technology platform integrating AI-based semantic resume screening and predictive recruitment analytics into a modern **React + FastAPI** web application.

Built for local execution on Windows using open-source, local-first Python and TypeScript libraries with zero reliance on paid external APIs or cloud services.

---

## 1. Architecture Overview

The application uses a decoupled full-stack architecture:

- **Backend (`backend/`)**: **FastAPI** + **Pydantic v2** + **Uvicorn** serving the existing `ai_module/` and `ida_module/` Python ML pipelines. Datasets, precomputed Sentence-BERT embeddings (`all-MiniLM-L6-v2`), the TF-IDF baseline matcher, and IDA model evaluation results are loaded once into memory at startup via FastAPI `lifespan`.
- **Frontend (`frontend/`)**: **React 18** + **TypeScript** + **Vite** + **Tailwind CSS** + **TanStack Query** + **Recharts**, using a Vite dev proxy (`/api` → `http://127.0.0.1:8000`).
- **Legacy Streamlit UI**: Archived at `legacy_streamlit/app.py` (with `dashboard/` preserved for backwards-compatible tests).

### Key Design Decisions
- **JSON-Serializable Analytics Siblings**: Added `*_data()` functions in `ida_module/eda.py` returning plain Python JSON structures while keeping all original Plotly functions intact for existing unit tests.
- **Thread-Safe Audit Log**: Writes to `outputs/recruiter_feedback.csv` are protected by a `threading.Lock`. Legacy `"Relevant Candidate"` entries are preserved when reading, while new submissions enforce canonical values (`"Suitable"`, `"Not Suitable"`, `"Needs Further Review"`).
- **Consistent ID Serialization**: `candidate_id` and `job_id` are always serialized as strings in JSON so active-learning reviewed-status lookups match reliably.

---

## 2. REST API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Backend health check & loaded dataset/embedding counts |
| `GET` | `/api/overview` | Live dataset counts & offer acceptance summary |
| `GET` | `/api/jobs/categories` | Sorted list of job categories |
| `GET` | `/api/jobs?category=` | Filtered list of job requisitions |
| `GET` | `/api/jobs/{job_id}` | Full job details and parsed skill taxonomy |
| `GET` | `/api/screening/rank?job_id=&top_n=15` | SBERT + TF-IDF ranking & skill-gap breakdown across all 9,544 resumes |
| `GET` | `/api/screening/candidate/{candidate_id}?job_id=` | Candidate profile, full resume text & skill gap |
| `GET` | `/api/screening/uncertain?job_id=` | Active Learning uncertainty queue (threshold `0.45 ± 0.05`, divergence `>= 0.20`) |
| `GET` | `/api/screening/compare?job_id=&top_k=15` | Spearman/Pearson correlations, Top-K overlap & 300-point scatter sample |
| `POST` | `/api/feedback` | Record recruiter review into `outputs/recruiter_feedback.csv` |
| `GET` | `/api/feedback?job_id=` | Read recruiter feedback audit log |
| `POST` | `/api/upload/screen` | Upload `.pdf` / `.docx` (max 10 MB) for live semantic screening & pool benchmark |
| `POST` | `/api/upload/demo` | Run live screening demo with pre-configured senior engineer profile |
| `GET` | `/api/analytics/{summary,funnel,monthly,sources,interviews,offers,roles,time}` | Chart-ready JSON for all IDA recruitment analytics |
| `GET` | `/api/models` | 4-model evaluation metrics, confusion matrices, ROC curves & feature importances |

Interactive Swagger documentation is available at `http://127.0.0.1:8000/docs` when the backend is running.

---

## 3. Running Locally on Windows (PowerShell)

### Prerequisites
- **Python 3.10+** (tested on Python 3.14)
- **Node.js 18+** & **npm** (tested on Node v24.12, npm 11.6)

### Step 1: Install Dependencies
```powershell
python -m pip install -r requirements.txt
cd frontend
npm install
cd ..
```

### Step 2: Start Both Servers
Option A — Use the included launcher script:
```powershell
.\run_dev.bat
```

Option B — Start manually in two PowerShell terminals:

Terminal 1 (FastAPI Backend on port 8000):
```powershell
python -m uvicorn backend.main:app --reload --port 8000
```

Terminal 2 (React Vite Frontend on port 5173):
```powershell
cd frontend
npm run dev
```

Open **`http://localhost:5173`** in your browser.

---

## 4. Automated Tests & Production Build

Run the full Python test suite (AI modules, IDA modules, legacy dashboard tests, and FastAPI `TestClient` integration tests):
```powershell
python -m unittest discover -s tests
```

Verify TypeScript types and Vite production build:
```powershell
cd frontend
npx tsc --noEmit
npm run build
```

---

## 5. Five Dashboard Pages (1:1 Feature Parity)

1. **Project Overview (`/`)**: Executive KPIs, Offer Decision Summary stacked bar, Core Platform Modules, and System Workflow.
2. **AI Resume Screening (`/screening`)**: Category/job selector, Top-N ranked table with Semantic Match progress bars, Candidate Inspector, Recruiter Match Review audit form, Active Learning Uncertainty Queue (Pending/Reviewed), and TF-IDF vs. SBERT scatter comparison.
3. **Upload Resume (`/upload`)**: Drag-and-drop `.pdf`/`.docx` ingestion (with corrupt, encrypted, scanned-PDF, and unsupported-type detection), Quick Demo mode, skill-gap diagnostics, and 30-bin talent pool benchmark placement.
4. **Recruitment Analytics (`/analytics`)**: 5 KPIs, Pipeline Funnel, Monthly Application Volume, Sourcing Channel Efficacy, Interview & Offer Donuts, Role Family Distribution, and Hiring Velocity histograms with date-anomaly exclusion callout.
5. **Model Comparison (`/models`)**: Held-out test metrics table (best per column highlighted), grouped bar chart, 4 model architecture cards (Majority-Class Baseline, Logistic Regression, Decision Tree, Random Forest), 4 confusion matrices, combined ROC curves, and interactive feature importance chart.

---

## 6. Responsible AI & Ethical Standards

1. **Demographic Proxy Elimination**: Age, gender, race, nationality, and personal demographic attributes are strictly excluded from all scoring and modeling pipelines.
2. **Advisory Decision Support**: Similarity scores measure contextual text relevance, not hiring suitability. Recruiters retain full decision authority.
3. **The Unstated Skill Principle**: Missing skills on a resume serve as interview probing cues rather than automatic disqualification.
4. **Auditable Feedback**: Recruiter reviews are logged to `outputs/recruiter_feedback.csv` without unverified automatic model retraining.
