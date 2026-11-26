"""Overview endpoint router."""

from fastapi import APIRouter, Depends

from backend.schemas import OfferSummary, OverviewCounts, OverviewResponse
from backend.state import AppState, get_app_state

router = APIRouter(prefix="/api", tags=["Overview"])


@router.get("/overview", response_model=OverviewResponse)
def get_overview(state: AppState = Depends(get_app_state)) -> OverviewResponse:
    resumes = int(len(state.cv_df))
    jobs = int(len(state.job_df))
    applications = int(len(state.ida_df))
    offers_total = int(state.audit_info.get("total_offers", int(state.ida_df["offered_at"].notnull().sum())))
    accepted = int(state.audit_info.get("offers_accepted_count", int((state.ida_df["accepted"] == True).sum())))
    declined = int(state.audit_info.get("offers_rejected_count", int((state.ida_df["accepted"] == False).sum())))
    anomalies = int(state.audit_info.get("excluded_anomaly_records", int((state.ida_df["offer_date_anomaly"] == True).sum())))

    accepted_pct = round((accepted / offers_total * 100.0), 1) if offers_total > 0 else 0.0
    declined_pct = round((declined / offers_total * 100.0), 1) if offers_total > 0 else 0.0

    return OverviewResponse(
        counts=OverviewCounts(
            resumes=resumes,
            jobs=jobs,
            applications=applications,
            offers=offers_total,
            anomalies=anomalies,
        ),
        offers=OfferSummary(
            accepted=accepted,
            declined=declined,
            total=offers_total,
            accepted_pct=accepted_pct,
            declined_pct=declined_pct,
        ),
    )
