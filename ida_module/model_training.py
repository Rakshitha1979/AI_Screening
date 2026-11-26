"""
IDA Module: Model Training for Offer Acceptance Classification.
Trains three distinct classification architectures:
1. Logistic Regression (linear probability baseline)
2. Decision Tree (interpretable rule-based classifier)
3. Random Forest (non-linear ensemble of decision trees)
Uses Scikit-learn Pipelines to guarantee no data leakage during preprocessing.
"""

import os
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
try:
    from sklearn.ensemble._forest import RandomForestClassifier
except ImportError:
    from sklearn.ensemble import RandomForestClassifier

from ida_module.feature_engineering import (
    build_offer_prediction_dataset,
    SELECTED_NUMERICAL_FEATURES,
    SELECTED_CATEGORICAL_FEATURES
)

MODELS_DIR = "outputs/models"


def create_preprocessor() -> ColumnTransformer:
    """
    Builds a Scikit-learn ColumnTransformer.
    StandardScales numerical features and OneHotEncodes categoricals.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), SELECTED_NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), SELECTED_CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor


def train_offer_models(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Prepares dataset, performs stratified 80/20 train-test split,
    and trains all three classification pipelines.
    Saves trained models to outputs/models/.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 1. Feature engineering
    X, y, meta = build_offer_prediction_dataset(df)
    
    # 2. Stratified train/test split to handle class imbalance consistently
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    # 3. Baseline: Majority-Class Baseline (Empirical Reference Benchmark)
    pipe_dummy = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", DummyClassifier(strategy="most_frequent"))
    ])
    pipe_dummy.fit(X_train, y_train)

    # 4. Model 1: Logistic Regression
    pipe_lr = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs"
        ))
    ])
    pipe_lr.fit(X_train, y_train)
    
    # 5. Model 2: Decision Tree
    pipe_dt = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=5,
            min_samples_leaf=10,
            random_state=random_state
        ))
    ])
    pipe_dt.fit(X_train, y_train)
    
    # 6. Model 3: Random Forest
    pipe_rf = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", RandomForestClassifier(
            class_weight="balanced",
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=5,
            random_state=random_state
        ))
    ])
    pipe_rf.fit(X_train, y_train)
    
    # Save models to disk
    joblib.dump(pipe_dummy, os.path.join(MODELS_DIR, "majority_baseline.joblib"))
    joblib.dump(pipe_lr, os.path.join(MODELS_DIR, "logistic_regression.joblib"))
    joblib.dump(pipe_dt, os.path.join(MODELS_DIR, "decision_tree.joblib"))
    joblib.dump(pipe_rf, os.path.join(MODELS_DIR, "random_forest.joblib"))
    
    # Get feature names after one-hot encoding
    fitted_preprocessor = pipe_lr.named_steps["preprocessor"]
    cat_encoder = fitted_preprocessor.named_transformers_["cat"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(SELECTED_CATEGORICAL_FEATURES))
    all_feature_names = SELECTED_NUMERICAL_FEATURES + cat_feature_names
    
    return {
        "models": {
            "Majority-Class Baseline": pipe_dummy,
            "Logistic Regression": pipe_lr,
            "Decision Tree": pipe_dt,
            "Random Forest": pipe_rf
        },
        "data": {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        },
        "all_feature_names": all_feature_names,
        "metadata": meta
    }


def load_trained_models() -> Dict[str, Any]:
    """
    Loads saved model pipelines from disk.
    """
    models = {}
    paths = {
        "Majority-Class Baseline": os.path.join(MODELS_DIR, "majority_baseline.joblib"),
        "Logistic Regression": os.path.join(MODELS_DIR, "logistic_regression.joblib"),
        "Decision Tree": os.path.join(MODELS_DIR, "decision_tree.joblib"),
        "Random Forest": os.path.join(MODELS_DIR, "random_forest.joblib")
    }
    for name, p in paths.items():
        if os.path.exists(p):
            models[name] = joblib.load(p)
    return models
