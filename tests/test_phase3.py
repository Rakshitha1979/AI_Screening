"""
Unit tests for Phase 3: AI Resume Screening, Embeddings, Ranking, and Upload.
"""

import os
import unittest
import io
import pandas as pd
import numpy as np

import docx
try:
    import pymupdf as fitz
except ImportError:
    import fitz

from ai_module.sentence_bert import encode_single_resume, encode_texts
from ai_module.similarity import (
    compute_cosine_similarity,
    TfidfBaselineMatcher,
    compare_matching_models,
    compute_precision_recall_at_k
)
from ai_module.candidate_ranking import (
    rank_candidates_for_job,
    record_recruiter_feedback,
    load_recruiter_feedback,
    identify_uncertain_candidates
)
from ai_module.resume_upload import (
    extract_text_from_file,
    screen_uploaded_resume
)


class TestPhase3(unittest.TestCase):
    
    def test_sentence_bert_encoding_and_similarity(self):
        text_a = "Senior Python engineer with AWS, Docker and Machine Learning experience."
        text_b = "Python Developer specializing in cloud computing and data science algorithms."
        text_c = "Accountant with bookkeeping and financial audit background."
        
        embs = encode_texts([text_a, text_b, text_c])
        self.assertEqual(embs.shape, (3, 384))
        
        # text_a should be much more similar to text_b than to text_c
        sim_ab = compute_cosine_similarity(embs[0], embs[1:2])[0]
        sim_ac = compute_cosine_similarity(embs[0], embs[2:3])[0]
        self.assertGreater(sim_ab, sim_ac)
        
    def test_tfidf_baseline_and_comparison(self):
        corpus = [
            "Python developer with machine learning and sql expertise",
            "Java software engineer building enterprise banking solutions",
            "Data analyst fluent in tableau, powerbi and python scripting"
        ]
        matcher = TfidfBaselineMatcher()
        matcher.fit_corpus(corpus)
        scores = matcher.match("Looking for a Python Machine Learning specialist")
        self.assertEqual(len(scores), 3)
        self.assertEqual(np.argmax(scores), 0)
        
        # Test model comparison
        comparison = compare_matching_models(
            tfidf_scores=np.array([0.8, 0.1, 0.4]),
            sbert_scores=np.array([0.85, 0.2, 0.5]),
            candidate_ids=["c1", "c2", "c3"],
            top_k=2
        )
        self.assertIn("spearman_rank_correlation", comparison)
        self.assertEqual(comparison["top_k_overlap_count"], 2)
        
    def test_precision_recall_at_k(self):
        ranked = ["c1", "c2", "c3", "c4", "c5"]
        ground_truth = {"c1", "c3", "c9"}
        eval_metrics = compute_precision_recall_at_k(ranked, ground_truth, k=3)
        self.assertEqual(eval_metrics["hits"], 2)
        self.assertAlmostEqual(eval_metrics["precision_at_k"], 2/3, places=2)
        self.assertAlmostEqual(eval_metrics["recall_at_k"], 2/3, places=2)
        
    def test_candidate_ranking_and_feedback(self):
        job_row = pd.Series({
            "job_id": 101,
            "job_title": "Python ML Engineer",
            "clean_skills": "['python', 'machine learning', 'aws', 'docker']",
            "years_required": 3.0
        })
        candidates_df = pd.DataFrame([
            {
                "candidate_id": "cand_1",
                "clean_resume_text": "Python engineer with machine learning and docker skills",
                "clean_skills": "['python', 'machine learning', 'docker']",
                "years_experience": 4.0,
                "target_position": "ML Engineer",
                "education_text": "B.Tech Computer Science"
            },
            {
                "candidate_id": "cand_2",
                "clean_resume_text": "Experienced sales and marketing professional",
                "clean_skills": "['sales', 'marketing', 'crm']",
                "years_experience": 5.0,
                "target_position": "Sales Lead",
                "education_text": "BBA"
            }
        ])
        job_emb = encode_single_resume("Python Machine Learning Engineer AWS Docker")
        cand_embs = encode_texts(candidates_df["clean_resume_text"].tolist())
        
        ranked = rank_candidates_for_job(job_row, candidates_df, job_emb, cand_embs, top_n=2)
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked.iloc[0]["candidate_id"], "cand_1")
        self.assertGreater(ranked.iloc[0]["skill_coverage_pct"], 0)
        
        # Test feedback recording
        test_feedback_file = "outputs/test_recruiter_feedback.csv"
        if os.path.exists(test_feedback_file):
            os.remove(test_feedback_file)
        success = record_recruiter_feedback(
            candidate_id="cand_1",
            job_id=101,
            job_title="Python ML Engineer",
            decision="Suitable",
            relevance_score=0.88,
            notes="Strong skill alignment",
            feedback_path=test_feedback_file
        )
        self.assertTrue(success)
        fb_df = load_recruiter_feedback(test_feedback_file)
        self.assertEqual(len(fb_df), 1)
        self.assertEqual(fb_df.iloc[0]["candidate_id"], "cand_1")
        if os.path.exists(test_feedback_file):
            os.remove(test_feedback_file)
            
    def test_identify_uncertain_candidates(self):
        test_candidates = pd.DataFrame([
            {"candidate_id": "c_border", "candidate_name": "Borderline Cand", "clean_resume_text": "python text", "target_position": "Dev"},
            {"candidate_id": "c_clear_high", "candidate_name": "Clear High", "clean_resume_text": "perfect text", "target_position": "Senior Dev"},
            {"candidate_id": "c_clear_low", "candidate_name": "Clear Low", "clean_resume_text": "irrelevant text", "target_position": "Chef"},
            {"candidate_id": "c_divergent", "candidate_name": "Divergent Cand", "clean_resume_text": "cloud specialist", "target_position": "Cloud Eng"}
        ])
        sbert_sims = np.array([0.449, 0.750, 0.120, 0.490])
        tfidf_sims = np.array([0.430, 0.710, 0.050, 0.150])
        
        test_fb_file = "outputs/test_recruiter_feedback_uncertainty.csv"
        if os.path.exists(test_fb_file):
            os.remove(test_fb_file)
            
        res = identify_uncertain_candidates(
            candidates_df=test_candidates,
            sbert_similarities=sbert_sims,
            job_id=999,
            tfidf_scores=tfidf_sims,
            decision_threshold=0.45,
            uncertainty_margin=0.05,
            disagreement_threshold=0.20,
            feedback_path=test_fb_file
        )
        
        pending = res["pending_candidates"]
        self.assertGreater(len(pending), 0)
        pending_ids = set(pending["candidate_id"])
        self.assertIn("c_border", pending_ids)
        self.assertIn("c_divergent", pending_ids)
        self.assertNotIn("c_clear_high", pending_ids)
        self.assertNotIn("c_clear_low", pending_ids)
        self.assertEqual(res["total_reviewed"], 0)
        self.assertEqual(res["total_awaiting_review"], len(pending))
        
        # Record feedback for c_border
        record_recruiter_feedback(
            candidate_id="c_border",
            job_id=999,
            job_title="Dev Role",
            decision="Suitable",
            relevance_score=0.449,
            feedback_path=test_fb_file
        )
        
        res2 = identify_uncertain_candidates(
            candidates_df=test_candidates,
            sbert_similarities=sbert_sims,
            job_id=999,
            tfidf_scores=tfidf_sims,
            decision_threshold=0.45,
            uncertainty_margin=0.05,
            disagreement_threshold=0.20,
            feedback_path=test_fb_file
        )
        
        self.assertNotIn("c_border", set(res2["pending_candidates"]["candidate_id"]))
        self.assertIn("c_border", set(res2["reviewed_candidates"]["candidate_id"]))
        self.assertEqual(res2["total_reviewed"], 1)
        self.assertEqual(res2["reviewed_candidates"].iloc[0]["review_status"], "Suitable")
        
        if os.path.exists(test_fb_file):
            os.remove(test_fb_file)
            
    def test_resume_upload_pdf_and_docx(self):
        # 1. Create a test PDF in memory
        pdf_doc = fitz.open()
        page = pdf_doc.new_page()
        page.insert_text(fitz.Point(50, 50), "Senior Full Stack Engineer. Skills: Python, React, PostgreSQL, Docker. 5 years of experience.")
        pdf_bytes = pdf_doc.write()
        pdf_doc.close()
        
        text_pdf, err_pdf = extract_text_from_file(pdf_bytes, "test_resume.pdf")
        self.assertIsNone(err_pdf)
        self.assertIn("Python", text_pdf)
        self.assertIn("React", text_pdf)
        
        # 2. Create a test DOCX in memory
        docx_doc = docx.Document()
        docx_doc.add_paragraph("Machine Learning Specialist with PyTorch, AWS, and M.S. degree. 4 years experience.")
        docx_buffer = io.BytesIO()
        docx_doc.save(docx_buffer)
        docx_bytes = docx_buffer.getvalue()
        
        text_docx, err_docx = extract_text_from_file(docx_bytes, "test_resume.docx")
        self.assertIsNone(err_docx)
        self.assertIn("Machine Learning", text_docx)
        
        # 3. Test end-to-end screen_uploaded_resume
        job_row = pd.Series({
            "job_id": 202,
            "job_title": "Full Stack Developer",
            "clean_skills": "['python', 'react', 'docker', 'kubernetes']",
            "years_required": 4.0
        })
        job_emb = encode_single_resume("Full Stack Developer Python React Docker Kubernetes")
        
        res = screen_uploaded_resume(pdf_bytes, "test_resume.pdf", job_row, job_emb)
        self.assertTrue(res["success"])
        self.assertIn("python", res["extracted_skills"])
        self.assertIn("react", res["extracted_skills"])
        self.assertGreater(res["similarity_score"], 0.4)
        self.assertIn("kubernetes", res["skill_gap"]["missing_skills"])
        
        # 4. Test invalid file format
        _, err_unsupported = extract_text_from_file(b"data", "resume.txt")
        self.assertIsNotNone(err_unsupported)


if __name__ == "__main__":
    unittest.main()
