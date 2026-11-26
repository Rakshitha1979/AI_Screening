"""Predictive Model Comparison endpoint router."""

from typing import Dict, List
from fastapi import APIRouter, Depends

from backend.schemas import (
    FeatureImportanceItem,
    ModelComparisonResponse,
    ModelMetricRow,
    RocCurveData,
)
from backend.state import AppState, get_app_state

router = APIRouter(prefix="/api", tags=["Models"])


@router.get("/models", response_model=ModelComparisonResponse)
def get_models_comparison(state: AppState = Depends(get_app_state)) -> ModelComparisonResponse:
    ev = state.evaluation_results
    cm_dict: Dict[str, List[List[int]]] = ev.get("confusion_matrices", {})

    # Derive test-set class counts dynamically from confusion matrices if not explicitly stored
    test_rejected = ev.get("test_rejected_count")
    test_accepted = ev.get("test_accepted_count")
    test_total = ev.get("test_sample_count")
    if (test_rejected is None or test_accepted is None or test_total is None) and cm_dict:
        first_cm = next(iter(cm_dict.values()))
        test_rejected = int(first_cm[0][0] + first_cm[0][1])
        test_accepted = int(first_cm[1][0] + first_cm[1][1])
        test_total = test_rejected + test_accepted

    raw_feat = ev.get("feature_importances", {})
    cleaned_feat: Dict[str, List[FeatureImportanceItem]] = {}
    for model_name, items in raw_feat.items():
        cleaned_feat[model_name] = [
            FeatureImportanceItem(
                feature=str(item["feature"]),
                display_feature=str(item["feature"]).replace("num__", "").replace("cat__", ""),
                importance=float(item["importance"]) if "importance" in item and item["importance"] is not None else None,
                coefficient=float(item["coefficient"]) if "coefficient" in item and item["coefficient"] is not None else None,
                abs_impact=float(item["abs_impact"]) if "abs_impact" in item and item["abs_impact"] is not None else None,
            )
            for item in items
        ]

    metrics_table = [ModelMetricRow(**row) for row in ev.get("metrics_table", [])]
    roc_data = {k: RocCurveData(**v) for k, v in ev.get("roc_data", {}).items()}

    return ModelComparisonResponse(
        metrics_table=metrics_table,
        confusion_matrices=cm_dict,
        roc_data=roc_data,
        feature_importances=cleaned_feat,
        test_sample_count=int(test_total or 0),
        test_accepted_count=int(test_accepted or 0),
        test_rejected_count=int(test_rejected or 0),
    )
