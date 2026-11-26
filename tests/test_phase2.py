"""
Unit tests for Phase 2: AI & IDA Preprocessing and EDA.
"""

import os
import unittest
import pandas as pd
import numpy as np

from ai_module.resume_processing import clean_text, parse_skills_field
from ai_module.nlp_processing import tokenize, remove_stopwords, compute_text_stats
from ai_module.skill_extraction import (
    extract_skills_from_text,
    extract_education,
    extract_years_experience,
    compute_skill_gap
)
from ida_module.preprocessing import preprocess_recruitment_data
from ida_module.eda import (
    get_recruitment_summary_metrics,
    get_recruitment_funnel_snapshot,
    get_time_analysis_figures
)


class TestPhase2(unittest.TestCase):
    
    def test_clean_text(self):
        raw = "Hello World! Check https://example.com and email candidate@test.com • Python & C++."
        cleaned = clean_text(raw)
        self.assertNotIn("https://", cleaned)
        self.assertNotIn("@", cleaned)
        self.assertIn("Python", cleaned)
        self.assertIn("C++", cleaned)
        
    def test_parse_skills(self):
        skills = "['python', 'machine learning', 'sql']"
        parsed = parse_skills_field(skills)
        self.assertEqual(parsed, ['python', 'machine learning', 'sql'])
        
    def test_nlp_processing(self):
        text = "Experienced Senior Python and C++ engineer with AI and machine learning background."
        tokens = tokenize(text)
        filtered = remove_stopwords(tokens)
        self.assertIn("python", filtered)
        self.assertIn("c++", filtered)
        self.assertIn("ai", filtered)
        stats = compute_text_stats(text)
        self.assertGreater(stats["token_count"], 5)
        self.assertGreater(stats["lexical_diversity"], 0)
        
    def test_skill_extraction(self):
        text = "Proficient in Python, React, PostgreSQL, Docker, and AWS. Also experienced with Machine Learning."
        extracted = extract_skills_from_text(text)
        for expected in ["python", "react", "postgresql", "docker", "aws", "machine learning"]:
            self.assertIn(expected, extracted)
            
    def test_education_and_experience_extraction(self):
        text = "Holds a Master of Science (M.S.) in Computer Science. Over 6 years of experience in software."
        edu = extract_education(text)
        self.assertTrue(any("M.S." in e or "Master" in e for e in edu))
        exp = extract_years_experience(text)
        self.assertEqual(exp, 6.0)
        
    def test_skill_gap(self):
        cand = ["python", "sql", "docker"]
        req = ["python", "docker", "kubernetes", "aws"]
        gap = compute_skill_gap(cand, req)
        self.assertEqual(set(gap["matching_skills"]), {"python", "docker"})
        self.assertEqual(set(gap["missing_skills"]), {"kubernetes", "aws"})
        self.assertEqual(gap["coverage_percentage"], 50.0)
        
    def test_ida_preprocessing_and_anomalies(self):
        df, audit = preprocess_recruitment_data(
            input_path="data/IDA_Master_Recruitment_Dataset.csv",
            output_path="outputs/cleaned_data/ida_cleaned_recruitment.csv"
        )
        self.assertEqual(len(df), 15000)
        self.assertEqual(audit["total_offers"], 938)
        self.assertEqual(audit["offers_accepted_count"], 699)
        self.assertEqual(audit["offers_rejected_count"], 239)
        self.assertEqual(audit["excluded_anomaly_records"], 217)
        self.assertEqual(audit["valid_acceptance_records"], 699 - 217)
        self.assertTrue(os.path.exists("outputs/cleaned_data/ida_cleaned_recruitment.csv"))
        
        # Test EDA metrics
        metrics = get_recruitment_summary_metrics(df)
        self.assertEqual(metrics["total_offers"], 938)
        self.assertEqual(metrics["accepted_offers"], 699)
        self.assertEqual(metrics["rejected_offers"], 239)
        self.assertGreater(metrics["avg_time_to_acceptance_days"], 0)
        
        time_analysis = get_time_analysis_figures(df)
        self.assertEqual(time_analysis["anomalies_excluded"], 217)
        self.assertEqual(time_analysis["tta_count"], 482)  # 699 - 217


if __name__ == "__main__":
    unittest.main()
