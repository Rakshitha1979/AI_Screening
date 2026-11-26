"""
API Integration Tests using FastAPI TestClient.
Verifies all endpoints, response schemas, error handling (404 / 422),
and preserves the existing recruiter_feedback.csv state.
"""

import io
import os
import unittest
import docx
import pymupdf as fitz
from fastapi.testclient import TestClient

from ai_module.candidate_ranking import FEEDBACK_FILE
from backend.main import app


class TestFastAPIBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._feedback_backup = None
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, "rb") as f:
                cls._feedback_backup = f.read()

        cls.client_ctx = TestClient(app)
        cls.client = cls.client_ctx.__enter__()

        jobs_resp = cls.client.get("/api/jobs")
        assert jobs_resp.status_code == 200
        cls.sample_job_id = jobs_resp.json()[0]["job_id"]
        cls.sample_job_title = jobs_resp.json()[0]["job_title"]

    @classmethod
    def tearDownClass(cls):
        cls.client_ctx.__exit__(None, None, None)
        if cls._feedback_backup is not None:
            with open(FEEDBACK_FILE, "wb") as f:
                f.write(cls._feedback_backup)
        elif os.path.exists(FEEDBACK_FILE):
            os.remove(FEEDBACK_FILE)

    def test_health_and_overview(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["resumes_loaded"], 9544)
        self.assertEqual(data["jobs_loaded"], 1167)
        self.assertEqual(data["applications_loaded"], 15000)

        ov = self.client.get("/api/overview")
        self.assertEqual(ov.status_code, 200)
        ov_data = ov.json()
        self.assertEqual(ov_data["counts"]["resumes"], 9544)
        self.assertEqual(ov_data["counts"]["jobs"], 1167)
        self.assertEqual(ov_data["offers"]["accepted"] + ov_data["offers"]["declined"], ov_data["offers"]["total"])

    def test_jobs_endpoints(self):
        cats = self.client.get("/api/jobs/categories")
        self.assertEqual(cats.status_code, 200)
        self.assertGreater(len(cats.json()), 0)

        detail = self.client.get(f"/api/jobs/{self.sample_job_id}")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["job_id"], self.sample_job_id)

        not_found = self.client.get("/api/jobs/nonexistent_999999")
        self.assertEqual(not_found.status_code, 404)

    def test_screening_rank_candidate_uncertain_compare(self):
        rank_r = self.client.get(f"/api/screening/rank?job_id={self.sample_job_id}&top_n=10")
        self.assertEqual(rank_r.status_code, 200)
        rank_data = rank_r.json()
        self.assertEqual(len(rank_data["candidates"]), 10)
        cand_id = rank_data["candidates"][0]["candidate_id"]

        cand_r = self.client.get(f"/api/screening/candidate/{cand_id}?job_id={self.sample_job_id}")
        self.assertEqual(cand_r.status_code, 200)
        self.assertEqual(cand_r.json()["candidate_id"], cand_id)
        self.assertIn("resume_text", cand_r.json())

        unc_r = self.client.get(f"/api/screening/uncertain?job_id={self.sample_job_id}")
        self.assertEqual(unc_r.status_code, 200)
        self.assertIn("pending", unc_r.json())
        self.assertIn("reviewed", unc_r.json())

        cmp_r = self.client.get(f"/api/screening/compare?job_id={self.sample_job_id}&top_k=10")
        self.assertEqual(cmp_r.status_code, 200)
        self.assertEqual(len(cmp_r.json()["scatter_sample"]), 300)

    def test_feedback_post_and_get(self):
        bad = self.client.post("/api/feedback", json={
            "candidate_id": "cand_00001",
            "job_id": self.sample_job_id,
            "job_title": self.sample_job_title,
            "decision": "Invalid Choice",
            "relevance_score": 0.5,
            "notes": "",
        })
        self.assertEqual(bad.status_code, 422)

        ok = self.client.post("/api/feedback", json={
            "candidate_id": "cand_00001",
            "job_id": self.sample_job_id,
            "job_title": self.sample_job_title,
            "decision": "Suitable",
            "relevance_score": 0.72,
            "notes": "API test entry",
        })
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(ok.json()["recruiter_decision"], "Suitable")

        fb_list = self.client.get(f"/api/feedback?job_id={self.sample_job_id}")
        self.assertEqual(fb_list.status_code, 200)
        self.assertGreaterEqual(len(fb_list.json()), 1)

    def test_upload_demo_and_files(self):
        demo = self.client.post("/api/upload/demo", json={"job_id": self.sample_job_id})
        self.assertEqual(demo.status_code, 200)
        self.assertTrue(demo.json()["success"])
        self.assertEqual(len(demo.json()["benchmark"]["histogram"]), 30)

        # Valid DOCX
        doc = docx.Document()
        doc.add_paragraph("Senior Python Developer with 5 years experience in Docker, AWS, SQL, and FastAPI.")
        docx_buf = io.BytesIO()
        doc.save(docx_buf)
        docx_resp = self.client.post(
            "/api/upload/screen",
            data={"job_id": self.sample_job_id},
            files={"file": ("resume.docx", docx_buf.getvalue(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        )
        self.assertEqual(docx_resp.status_code, 200)
        self.assertTrue(docx_resp.json()["success"])

        # Valid PDF
        pdf = fitz.open()
        page = pdf.new_page()
        page.insert_text((72, 72), "Data Scientist with 4 years experience in Python, Scikit-Learn, Pandas, and Machine Learning.")
        pdf_bytes = pdf.tobytes()
        pdf.close()
        pdf_resp = self.client.post(
            "/api/upload/screen",
            data={"job_id": self.sample_job_id},
            files={"file": ("resume.pdf", pdf_bytes, "application/pdf")},
        )
        self.assertEqual(pdf_resp.status_code, 200)
        self.assertTrue(pdf_resp.json()["success"])

        # Scanned / empty PDF -> 422 {error}
        empty_pdf = fitz.open()
        empty_pdf.new_page()
        empty_bytes = empty_pdf.tobytes()
        empty_pdf.close()
        empty_resp = self.client.post(
            "/api/upload/screen",
            data={"job_id": self.sample_job_id},
            files={"file": ("scanned.pdf", empty_bytes, "application/pdf")},
        )
        self.assertEqual(empty_resp.status_code, 422)
        self.assertIn("error", empty_resp.json())

        # Unsupported file type -> 422 {error}
        txt_resp = self.client.post(
            "/api/upload/screen",
            data={"job_id": self.sample_job_id},
            files={"file": ("resume.txt", b"hello world", "text/plain")},
        )
        self.assertEqual(txt_resp.status_code, 422)
        self.assertIn("error", txt_resp.json())

    def test_analytics_endpoints(self):
        for ep in ["summary", "funnel", "monthly", "sources", "interviews", "offers", "roles", "time"]:
            r = self.client.get(f"/api/analytics/{ep}")
            self.assertEqual(r.status_code, 200, f"Failed on /api/analytics/{ep}")

    def test_models_endpoint(self):
        r = self.client.get("/api/models")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(len(data["metrics_table"]), 4)
        self.assertGreater(data["test_sample_count"], 0)
        self.assertEqual(data["test_accepted_count"] + data["test_rejected_count"], data["test_sample_count"])


if __name__ == "__main__":
    unittest.main()
