"""
FastAPI Main Application Entry Point.
Replaces Streamlit's app.py shell with a high-performance REST API.
"""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from backend.routers import analytics, models, overview, screening, upload
from backend.schemas import HealthResponse
from backend.state import AppState, get_app_state, get_or_init_state


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.app_state = get_or_init_state()
    yield


app = FastAPI(
    title="AI Candidate Screening & Recruitment Analytics API",
    version="2.0.0",
    description="FastAPI backend serving Sentence-BERT semantic screening, resume parsing, IDA recruitment analytics, and offer acceptance prediction models.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

app.include_router(overview.router)
app.include_router(screening.router)
app.include_router(upload.router)
app.include_router(analytics.router)
app.include_router(models.router)


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
def health_check(state: AppState = Depends(get_app_state)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        resumes_loaded=int(len(state.cv_df)),
        jobs_loaded=int(len(state.job_df)),
        applications_loaded=int(len(state.ida_df)),
        cv_embeddings_shape=[int(x) for x in state.cv_embeddings.shape],
        job_embeddings_shape=[int(x) for x in state.job_embeddings.shape],
    )
