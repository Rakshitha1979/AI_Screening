"""
IDA Module: Model Evaluation and Performance Comparison.
Evaluates all three models on the exact same held-out test set (20% = 188 samples).
Calculates real Accuracy, Precision, Recall, F1-score, ROC-AUC, and Confusion Matrices.
Extracts feature importances and generates interactive Plotly visualizations.
"""

import json
import os
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)
import plotly.express as px
import plotly.graph_objects as go


def evaluate_all_models(
    models_dict: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: List[str]
) -> Dict[str, Any]:
    """
    Computes rigorous evaluation metrics on the identical test split for all three models.
    Guarantees fair comparison and never fabricates scores.
    """
    metrics_list = []
    confusion_matrices = {}
    roc_data = {}
    feature_importances = {}
    
    for model_name, pipeline in models_dict.items():
        # Predict class labels (threshold 0.5)
        y_pred = pipeline.predict(X_test)
        
        # Predict probabilities for positive class (Accepted = 1)
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            roc_auc = float(roc_auc_score(y_test, y_proba))
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_data[model_name] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": round(roc_auc, 3)
            }
        else:
            y_proba = None
            roc_auc = 0.0
            
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices[model_name] = cm.tolist()
        
        metrics_list.append({
            "Model": model_name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4)
        })
        
        # Feature importances / coefficients
        classifier = pipeline.named_steps["classifier"]
        if hasattr(classifier, "feature_importances_"):
            importances = classifier.feature_importances_
            if len(importances) == len(feature_names):
                feat_df = pd.DataFrame({
                    "feature": feature_names,
                    "importance": importances
                }).sort_values("importance", ascending=False)
                feature_importances[model_name] = feat_df.head(15).to_dict(orient="records")
        elif hasattr(classifier, "coef_"):
            coefs = classifier.coef_[0]
            if len(coefs) == len(feature_names):
                feat_df = pd.DataFrame({
                    "feature": feature_names,
                    "coefficient": coefs,
                    "abs_impact": np.abs(coefs)
                }).sort_values("abs_impact", ascending=False)
                feature_importances[model_name] = feat_df.head(15).to_dict(orient="records")
                
    comparison_df = pd.DataFrame(metrics_list)
    
    # Save results to json
    results = {
        "metrics_table": comparison_df.to_dict(orient="records"),
        "confusion_matrices": confusion_matrices,
        "roc_data": roc_data,
        "feature_importances": feature_importances,
        "test_sample_count": len(y_test),
        "test_accepted_count": int(y_test.sum()),
        "test_rejected_count": int((y_test == 0).sum())
    }
    
    os.makedirs("outputs/models", exist_ok=True)
    with open("outputs/models/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    return results


def plot_metrics_comparison(metrics_records: List[Dict[str, Any]]) -> go.Figure:
    """
    Creates a multi-bar chart comparing evaluation metrics across the three models.
    """
    df = pd.DataFrame(metrics_records)
    melted = df.melt(id_vars=["Model"], var_name="Metric", value_name="Score")
    
    fig = px.bar(
        melted,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        title="Predictive Performance Comparison Across Models",
        text=melted["Score"].apply(lambda v: f"{v:.2f}"),
        color_discrete_sequence=["#78909C", "#1E88E5", "#FB8C00", "#43A047"]
    )
    fig.update_layout(
        template="plotly_white",
        yaxis=dict(range=[0.0, 1.05]),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


def plot_confusion_matrix(cm_list: List[List[int]], model_name: str) -> go.Figure:
    """
    Renders an interactive confusion matrix heatmap with TN, FP, FN, TP annotations.
    """
    cm = np.array(cm_list)
    labels = ["Offer Rejected (0)", "Offer Accepted (1)"]
    
    annotations = []
    names = [["True Negative (TN)", "False Positive (FP)"], ["False Negative (FN)", "True Positive (TP)"]]
    for i in range(2):
        for j in range(2):
            annotations.append(
                f"{names[i][j]}<br><b>{cm[i, j]}</b>"
            )
    z_text = np.array(annotations).reshape(2, 2)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=z_text,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=False
    ))
    fig.update_layout(
        title=f"Confusion Matrix – {model_name}",
        xaxis_title="Predicted Decision",
        yaxis_title="Actual Outcome",
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


def plot_combined_roc_curves(roc_data: Dict[str, Any]) -> go.Figure:
    """
    Renders the ROC curves for all three models together with the random guessing baseline.
    """
    fig = go.Figure()
    
    colors = {
        "Majority-Class Baseline": "#78909C",
        "Logistic Regression": "#1E88E5",
        "Decision Tree": "#FB8C00",
        "Random Forest": "#43A047"
    }
    
    for model_name, curve_info in roc_data.items():
        fpr = curve_info["fpr"]
        tpr = curve_info["tpr"]
        auc = curve_info["auc"]
        color = colors.get(model_name, "#333333")
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"{model_name} (AUC = {auc:.3f})",
            line=dict(width=2.5, color=color)
        ))
        
    # Baseline random chance
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random Classifier (AUC = 0.50)",
        line=dict(dash="dash", color="gray", width=1.5)
    ))
    
    fig.update_layout(
        title="Receiver Operating Characteristic (ROC) Curves",
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Sensitivity / Recall)",
        template="plotly_white",
        xaxis=dict(range=[0.0, 1.0]),
        yaxis=dict(range=[0.0, 1.05]),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


def plot_feature_importance(feat_list: List[Dict[str, Any]], model_name: str) -> go.Figure:
    """
    Visualizes top feature importances or coefficients.
    """
    df = pd.DataFrame(feat_list)
    val_col = "importance" if "importance" in df.columns else "abs_impact"
    df = df.sort_values(val_col, ascending=True)
    
    # Clean up feature names for clean presentation
    df["display_feature"] = df["feature"].str.replace("num__", "").str.replace("cat__", "")
    
    fig = px.bar(
        df,
        x=val_col,
        y="display_feature",
        orientation="h",
        title=f"Top Influential Features – {model_name}",
        labels={val_col: "Importance / Impact", "display_feature": "Feature"},
        color_discrete_sequence=["#00897B"]
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig
