"""IDA Recruitment Analytics endpoints router."""

from typing import List
from fastapi import APIRouter, Depends

from backend.schemas import (
    AnalyticsSummaryResponse,
    FunnelStage,
    InterviewOutcome,
    MonthlyTrend,
    OfferOutcome,
    RoleFamilyStat,
    SourceStat,
    TimeAnalysisResponse,
)
from backend.state import AppState, get_app_state
from ida_module.eda import (
    get_interview_outcomes_data,
    get_monthly_application_trends_data,
    get_offer_outcome_distribution_data,
    get_recruitment_funnel_data,
    get_recruitment_summary_data,
    get_role_family_distribution_data,
    get_source_analysis_data,
    get_time_analysis_data,
)

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])


@router.get("/summary", response_model=AnalyticsSummaryResponse)
def analytics_summary(state: AppState = Depends(get_app_state)) -> AnalyticsSummaryResponse:
    return AnalyticsSummaryResponse(**get_recruitment_summary_data(state.ida_df))


@router.get("/funnel", response_model=List[FunnelStage])
def analytics_funnel(state: AppState = Depends(get_app_state)) -> List[FunnelStage]:
    return [FunnelStage(**item) for item in get_recruitment_funnel_data(state.ida_df)]


@router.get("/monthly", response_model=List[MonthlyTrend])
def analytics_monthly(state: AppState = Depends(get_app_state)) -> List[MonthlyTrend]:
    return [MonthlyTrend(**item) for item in get_monthly_application_trends_data(state.ida_df)]


@router.get("/sources", response_model=List[SourceStat])
def analytics_sources(state: AppState = Depends(get_app_state)) -> List[SourceStat]:
    return [SourceStat(**item) for item in get_source_analysis_data(state.ida_df)]


@router.get("/interviews", response_model=List[InterviewOutcome])
def analytics_interviews(state: AppState = Depends(get_app_state)) -> List[InterviewOutcome]:
    return [InterviewOutcome(**item) for item in get_interview_outcomes_data(state.ida_df)]


@router.get("/offers", response_model=List[OfferOutcome])
def analytics_offers(state: AppState = Depends(get_app_state)) -> List[OfferOutcome]:
    return [OfferOutcome(**item) for item in get_offer_outcome_distribution_data(state.ida_df)]


@router.get("/roles", response_model=List[RoleFamilyStat])
def analytics_roles(state: AppState = Depends(get_app_state)) -> List[RoleFamilyStat]:
    return [RoleFamilyStat(**item) for item in get_role_family_distribution_data(state.ida_df)]


@router.get("/time", response_model=TimeAnalysisResponse)
def analytics_time(state: AppState = Depends(get_app_state)) -> TimeAnalysisResponse:
    return TimeAnalysisResponse(**get_time_analysis_data(state.ida_df))
