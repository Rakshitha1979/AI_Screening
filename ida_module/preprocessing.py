"""
IDA Module: Data Preprocessing and Cleaning.
Handles recruitment dataset cleaning, validation, stage-aware null preservation,
and date anomaly isolation without target or ID imputation.
"""

import os
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np


def preprocess_recruitment_data(
    input_path: str = "data/IDA_Master_Recruitment_Dataset.csv",
    output_path: str = "outputs/cleaned_data/ida_cleaned_recruitment.csv"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans and validates the IDA recruitment master dataset.
    
    Principles strictly followed:
    1. Preserves domain-valid missing values for stages not yet reached.
    2. Identifies and documents 217 offer-date anomalies (accepted_at < offered_at).
    3. Excludes anomaly records from valid duration metrics.
    4. Never imputes IDs or target offer outcomes.
    5. Saves cleaned dataset to output_path.
    """
    df = pd.read_csv(input_path)
    audit: Dict[str, Any] = {}
    audit["initial_record_count"] = len(df)
    audit["initial_columns"] = list(df.columns)
    
    # Check duplicate application IDs
    duplicate_apps = df["application_id"].duplicated().sum()
    audit["duplicate_application_ids"] = int(duplicate_apps)
    
    # 1. Standardize text categories
    categorical_cols = [
        "source", "stage", "role_family", "level", "location_type",
        "employment_type", "industry", "company_size", "status", "top_skill_area"
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].replace({"nan": np.nan, "none": np.nan, "": np.nan})
            
    # 2. Date parsing and validation
    date_cols = [
        "applied_at", "posted_at", "first_interview_date",
        "last_interview_date", "offered_at", "accepted_at"
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            
    # 3. Numeric columns validation
    numeric_cols = [
        "years_experience", "score", "salary_low", "salary_high",
        "interview_count", "pass_count", "fail_count", "hold_count", "base_salary"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    # 4. Target validation: accepted
    # accepted is boolean for offer stage, NaN for pre-offer stages
    audit["total_offers"] = int(df["offered_at"].notnull().sum())
    audit["offers_accepted_count"] = int((df["accepted"] == True).sum())
    audit["offers_rejected_count"] = int((df["accepted"] == False).sum())
    audit["pre_offer_applicants_count"] = int(df["accepted"].isnull().sum())
    
    # 5. Date Anomaly Analysis
    # Anomaly condition: candidate accepted_at is earlier than offered_at
    if "offer_date_anomaly" in df.columns:
        audit["flagged_anomalies_count"] = int((df["offer_date_anomaly"] == True).sum())
    else:
        audit["flagged_anomalies_count"] = int(
            ((df["accepted_at"].notnull()) & (df["offered_at"].notnull()) & 
             (df["accepted_at"] < df["offered_at"])).sum()
        )
        df["offer_date_anomaly"] = (df["accepted_at"].notnull()) & (df["offered_at"].notnull()) & (df["accepted_at"] < df["offered_at"])

    # Calculate time metrics strictly for valid, non-anomalous accepted offers
    # Time-to-acceptance: accepted_at - offered_at
    df["time_to_acceptance_days"] = np.nan
    df["application_to_offer_days"] = np.nan
    df["application_to_accepted_days"] = np.nan
    
    # Application to offer duration (for all offers where offered_at >= applied_at)
    valid_offer_mask = df["offered_at"].notnull() & df["applied_at"].notnull() & (df["offered_at"] >= df["applied_at"])
    df.loc[valid_offer_mask, "application_to_offer_days"] = (
        (df.loc[valid_offer_mask, "offered_at"] - df.loc[valid_offer_mask, "applied_at"]).dt.total_seconds() / 86400.0
    )
    
    # Time-to-acceptance (strictly non-anomalous accepted offers)
    valid_acceptance_mask = (
        (df["accepted"] == True) &
        (df["offered_at"].notnull()) &
        (df["accepted_at"].notnull()) &
        (df["offer_date_anomaly"] == False) &
        (df["accepted_at"] >= df["offered_at"])
    )
    df.loc[valid_acceptance_mask, "time_to_acceptance_days"] = (
        (df.loc[valid_acceptance_mask, "accepted_at"] - df.loc[valid_acceptance_mask, "offered_at"]).dt.total_seconds() / 86400.0
    )
    
    # Application-to-accepted-offer duration (strictly valid accepted offers)
    valid_full_duration_mask = (
        (df["accepted"] == True) &
        (df["applied_at"].notnull()) &
        (df["accepted_at"].notnull()) &
        (df["offer_date_anomaly"] == False) &
        (df["accepted_at"] >= df["applied_at"])
    )
    df.loc[valid_full_duration_mask, "application_to_accepted_days"] = (
        (df.loc[valid_full_duration_mask, "accepted_at"] - df.loc[valid_full_duration_mask, "applied_at"]).dt.total_seconds() / 86400.0
    )
    
    audit["valid_acceptance_records"] = int(valid_acceptance_mask.sum())
    audit["excluded_anomaly_records"] = int((df["offer_date_anomaly"] == True).sum())
    
    # Stage distribution audit
    audit["stage_counts"] = df["stage"].value_counts(dropna=False).to_dict()
    
    # Save cleaned dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    return df, audit


if __name__ == "__main__":
    cleaned_df, audit_info = preprocess_recruitment_data()
    print("Preprocessed Recruitment Data successfully!")
    print(f"Shape: {cleaned_df.shape}")
    print(f"Total Offers: {audit_info['total_offers']} (Accepted: {audit_info['offers_accepted_count']}, Rejected: {audit_info['offers_rejected_count']})")
    print(f"Flagged Anomalies Excluded from Durations: {audit_info['excluded_anomaly_records']}")
    print(f"Valid Acceptance Timing Records: {audit_info['valid_acceptance_records']}")
