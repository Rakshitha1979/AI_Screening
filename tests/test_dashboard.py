"""
Unit tests for Integrated Dashboard and Application Logic.
Covers all five active dashboard pages (Page 1–5).
"""

import os
import unittest
import pandas as pd
import numpy as np

from app import get_cached_datasets, get_cached_embeddings, get_cached_evaluation
from dashboard.pages.page1_overview import render_page1
from dashboard.pages.page2_screening import render_page2
from dashboard.pages.page3_upload import render_page3
from dashboard.pages.page4_analytics import render_page4
from dashboard.pages.page5_models import render_page5


class TestDashboard(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.df_cv, cls.df_job, cls.df_matches, cls.df_ida, cls.audit = get_cached_datasets()
        cls.job_embs, cls.cv_embs, cls.tfidf = get_cached_embeddings(cls.df_job, cls.df_cv)
        cls.eval_results = get_cached_evaluation(cls.df_ida)
        
    def test_datasets_loaded(self):
        self.assertEqual(len(self.df_cv), 9544)
        self.assertEqual(len(self.df_job), 1167)
        self.assertEqual(len(self.df_matches), 122560)
        self.assertEqual(len(self.df_ida), 15000)
        
    def test_embeddings_and_matcher(self):
        self.assertEqual(self.job_embs.shape, (1167, 384))
        self.assertEqual(self.cv_embs.shape, (9544, 384))
        self.assertTrue(self.tfidf.is_fitted)
        
    def test_evaluation_results(self):
        self.assertIn("metrics_table", self.eval_results)
        self.assertIn("confusion_matrices", self.eval_results)
        self.assertIn("roc_data", self.eval_results)
        self.assertEqual(len(self.eval_results["metrics_table"]), 4)
        
    def test_page_modules_importable(self):
        # Verify all five page render functions are callable
        self.assertTrue(callable(render_page1))
        self.assertTrue(callable(render_page2))
        self.assertTrue(callable(render_page3))
        self.assertTrue(callable(render_page4))
        self.assertTrue(callable(render_page5))

    def test_render_page1_execution(self):
        # Verify render_page1 executes without error
        try:
            render_page1(self.df_cv, self.df_job, self.df_matches, self.df_ida, self.audit)
            executed = True
        except Exception as e:
            executed = False
            self.fail(f"render_page1 raised an unexpected exception: {e}")
        self.assertTrue(executed)

    def test_render_page2_execution(self):
        # Verify render_page2 executes without error including Active Learning queue
        try:
            render_page2(self.df_cv, self.df_job, self.cv_embs, self.job_embs, self.tfidf)
            executed = True
        except Exception as e:
            executed = False
            self.fail(f"render_page2 raised an unexpected exception: {e}")
        self.assertTrue(executed)

    def test_render_page3_execution(self):
        # Verify render_page3 executes without error (without uploaded file)
        try:
            render_page3(self.df_job, self.job_embs, self.df_cv, self.cv_embs)
            executed = True
        except Exception as e:
            executed = False
            self.fail(f"render_page3 raised an unexpected exception: {e}")
        self.assertTrue(executed)

    def test_render_page4_execution(self):
        # Verify render_page4 executes without error
        try:
            render_page4(self.df_ida)
            executed = True
        except Exception as e:
            executed = False
            self.fail(f"render_page4 raised an unexpected exception: {e}")
        self.assertTrue(executed)

    def test_render_page5_execution(self):
        # Verify render_page5 executes without error
        try:
            render_page5(self.eval_results)
            executed = True
        except Exception as e:
            executed = False
            self.fail(f"render_page5 raised an unexpected exception: {e}")
        self.assertTrue(executed)


if __name__ == "__main__":
    unittest.main()

