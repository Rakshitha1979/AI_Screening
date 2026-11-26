"""
Unit tests for Phase 4: IDA Feature Engineering, Model Training, and Evaluation.
"""

import os
import unittest
import pandas as pd
import numpy as np

from ida_module.feature_engineering import build_offer_prediction_dataset
from ida_module.model_training import train_offer_models, load_trained_models
from ida_module.evaluation import (
    evaluate_all_models,
    plot_metrics_comparison,
    plot_confusion_matrix,
    plot_combined_roc_curves,
    plot_feature_importance
)


class TestPhase4(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cleaned_path = "outputs/cleaned_data/ida_cleaned_recruitment.csv"
        if not os.path.exists(cleaned_path):
            from ida_module.preprocessing import preprocess_recruitment_data
            preprocess_recruitment_data()
        cls.df = pd.read_csv(cleaned_path)
        
    def test_feature_engineering(self):
        X, y, meta = build_offer_prediction_dataset(self.df)
        self.assertEqual(len(X), 938)
        self.assertEqual(len(y), 938)
        self.assertEqual(meta["accepted_count"], 699)
        self.assertEqual(meta["rejected_count"], 239)
        # Check no missing values in X
        self.assertEqual(X.isnull().sum().sum(), 0)
        # Check target contains only 0 and 1
        self.assertEqual(set(y.unique()), {0, 1})
        # Check pass_rate and salary ratios are present
        self.assertIn("pass_rate", X.columns)
        self.assertIn("salary_to_midpoint_ratio", X.columns)
        # Check excluded columns are not in X
        self.assertNotIn("accepted_at", X.columns)
        self.assertNotIn("candidate_name", X.columns)
        self.assertNotIn("application_id", X.columns)
        
    def test_model_training_and_evaluation(self):
        training_output = train_offer_models(self.df, test_size=0.2, random_state=42)
        models = training_output["models"]
        data = training_output["data"]
        feature_names = training_output["all_feature_names"]
        
        self.assertIn("Majority-Class Baseline", models)
        self.assertIn("Logistic Regression", models)
        self.assertIn("Decision Tree", models)
        self.assertIn("Random Forest", models)
        
        # Test split size
        self.assertEqual(len(data["X_test"]), 188)
        self.assertEqual(len(data["X_train"]), 750)
        
        # Verify saved model files exist
        self.assertTrue(os.path.exists("outputs/models/majority_baseline.joblib"))
        self.assertTrue(os.path.exists("outputs/models/logistic_regression.joblib"))
        self.assertTrue(os.path.exists("outputs/models/decision_tree.joblib"))
        self.assertTrue(os.path.exists("outputs/models/random_forest.joblib"))
        
        # Test evaluation
        results = evaluate_all_models(
            models_dict=models,
            X_test=data["X_test"],
            y_test=data["y_test"],
            feature_names=feature_names
        )
        self.assertEqual(len(results["metrics_table"]), 4)
        for row in results["metrics_table"]:
            self.assertGreater(row["Accuracy"], 0.4)
            self.assertGreater(row["ROC-AUC"], 0.4)
            self.assertLessEqual(row["Accuracy"], 1.0)
            
        # Check confusion matrices
        for name in ["Majority-Class Baseline", "Logistic Regression", "Decision Tree", "Random Forest"]:
            self.assertIn(name, results["confusion_matrices"])
            cm = np.array(results["confusion_matrices"][name])
            self.assertEqual(cm.shape, (2, 2))
            self.assertEqual(cm.sum(), 188)
            
        # Test plots generation
        fig_metrics = plot_metrics_comparison(results["metrics_table"])
        self.assertIsNotNone(fig_metrics)
        
        fig_cm = plot_confusion_matrix(results["confusion_matrices"]["Random Forest"], "Random Forest")
        self.assertIsNotNone(fig_cm)
        
        fig_roc = plot_combined_roc_curves(results["roc_data"])
        self.assertIsNotNone(fig_roc)
        
        if "Random Forest" in results["feature_importances"]:
            fig_fi = plot_feature_importance(results["feature_importances"]["Random Forest"], "Random Forest")
            self.assertIsNotNone(fig_fi)
            
        # Test loading saved models
        loaded = load_trained_models()
        self.assertEqual(len(loaded), 4)


if __name__ == "__main__":
    unittest.main()
