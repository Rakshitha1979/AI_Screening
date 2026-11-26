"""
IDA Module: Feature Engineering for Offer Acceptance Prediction.
Constructs valid, pre-decision feature matrices without data leakage.
Filters strictly to applicants with confirmed offer outcomes (N=938).
"""

from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np


# Explicit documentation of features
SELECTED_NUMERICAL_FEATURES: List[str] = [
    "years_experience",
    "score",
    "interview_count",
    "pass_count",
    "fail_count",
    "hold_count",
    "pass_rate",
    "salary_low",
    "salary_high",
    "salary_midpoint",
    "salary_spread",
    "base_salary",
    "salary_to_midpoint_ratio"
]

SELECTED_CATEGORICAL_FEATURES: List[str] = [
    "role_family",
    "level",
    "location_type",
    "employment_type",
    "industry",
    "company_size"
]

EXCLUDED_FEATURES_RATIONALE: Dict[str, str] = {
    "application_id": "Arbitrary surrogate identifier; holds no generalizable predictive value.",
    "job_id": "Categorical identifier; risks target leakage and overfitting to specific job slots.",
    "candidate_id": "Unique individual identifier; violates privacy and causes memorization.",
    "company_id": "Company surrogate ID; high cardinality without generalizable predictive signal.",
    "offer_id": "Post-decision transactional key.",
    "candidate_name": "Protected personally identifiable information (PII); causes bias and target leakage.",
    "summary": "Unstructured self-reported text; high risk of memorizing specific candidates.",
    "current_city": "Demographic location attribute; excluded to prevent geographic bias.",
    "hq_city": "High-cardinality location field prone to overfitting.",
    "company_name": "Specific brand name; causes memorization rather than learning industry patterns.",
    "title": "Specific job posting title; represented systematically by role_family and level.",
    "accepted_at": "Target outcome timestamp; direct target leakage.",
    "offered_at": "Post-screening event timestamp; excluded to avoid temporal target leakage.",
    "offer_date_anomaly": "Calculated only from accepted_at vs offered_at; target leakage.",
    "time_to_acceptance_days": "Derived from target acceptance timestamp; extreme leakage.",
    "application_to_offer_days": "Post-application timing variable.",
    "application_to_accepted_days": "Derived from target acceptance timestamp; extreme leakage.",
    "status": "Posting lifecycle flag that may reflect post-acceptance state.",
    "first_interview_date": "Temporal timestamp without normalized reference.",
    "last_interview_date": "Temporal timestamp without normalized reference.",
    "posted_at": "Job posting timestamp.",
    "applied_at": "Application date timestamp."
}


def build_offer_prediction_dataset(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Prepares the feature matrix (X) and target vector (y) for offer acceptance modeling.
    
    Target:
    - accepted (1 = Offer Accepted, 0 = Offer Declined / Rejected)
    - Strictly filtered to records where accepted is not null (938 offers).
    """
    # 1. Filter to offer stage records with confirmed decisions
    offer_records = df[df["accepted"].notnull()].copy()
    if offer_records.empty:
        raise ValueError("No records found with valid accepted offer outcome.")
        
    # 2. Construct engineered numerical features
    # Safe pass rate calculation
    total_rounds = offer_records["pass_count"] + offer_records["fail_count"] + offer_records["hold_count"]
    total_rounds = np.where(total_rounds > 0, total_rounds, offer_records["interview_count"])
    total_rounds = np.where(total_rounds > 0, total_rounds, 1.0)
    offer_records["pass_rate"] = np.round(offer_records["pass_count"] / total_rounds, 3)
    
    # Salary range metrics
    offer_records["salary_midpoint"] = (offer_records["salary_low"] + offer_records["salary_high"]) / 2.0
    offer_records["salary_spread"] = offer_records["salary_high"] - offer_records["salary_low"]
    
    # Ratio of offered base salary relative to posting midpoint
    midpoint_safe = np.where(offer_records["salary_midpoint"] > 0, offer_records["salary_midpoint"], 1.0)
    offer_records["salary_to_midpoint_ratio"] = np.round(offer_records["base_salary"] / midpoint_safe, 3)
    
    # 3. Assemble feature matrix X
    feature_columns = SELECTED_NUMERICAL_FEATURES + SELECTED_CATEGORICAL_FEATURES
    X = offer_records[feature_columns].copy()
    
    # Handle missing values if any exist in feature subset
    for col in SELECTED_NUMERICAL_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(X[col].median())
        
    for col in SELECTED_CATEGORICAL_FEATURES:
        X[col] = X[col].fillna("missing").astype(str).str.lower().str.strip()
        
    # 4. Target encoding: True -> 1, False -> 0
    y = offer_records["accepted"].astype(bool).astype(int)
    
    metadata = {
        "total_offer_records": len(offer_records),
        "accepted_count": int(y.sum()),
        "rejected_count": int((y == 0).sum()),
        "acceptance_rate": round(float(y.mean() * 100.0), 2),
        "selected_numerical_features": SELECTED_NUMERICAL_FEATURES,
        "selected_categorical_features": SELECTED_CATEGORICAL_FEATURES,
        "excluded_features_documentation": EXCLUDED_FEATURES_RATIONALE
    }
    
    return X, y, metadata
