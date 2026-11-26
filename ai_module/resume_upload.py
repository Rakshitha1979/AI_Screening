"""
Resume Upload and Live Screening Module.
Parses PDF (via PyMuPDF) and DOCX (via python-docx) documents,
validates integrity, handles password encryption and scanned image edge cases,
and executes real-time semantic matching and skill-gap analysis against target jobs.
"""

import io
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np

# Use pymupdf directly for modern API compatibility
try:
    import pymupdf as fitz
except ImportError:
    import fitz

import docx

from ai_module.resume_processing import clean_text, parse_skills_field
from ai_module.skill_extraction import (
    extract_skills_from_text,
    extract_education,
    extract_years_experience,
    compute_skill_gap
)
from ai_module.sentence_bert import encode_single_resume
from ai_module.similarity import compute_cosine_similarity


def extract_text_from_pdf(file_bytes: bytes) -> Tuple[str, Optional[str]]:
    """
    Extracts text from PDF bytes using PyMuPDF.
    Handles encryption, corrupt files, and scanned PDFs without extractable text layers.
    Returns (extracted_text, error_message).
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        return "", f"Corrupt or unreadable PDF file: {str(e)}"
        
    # Check password protection
    if doc.is_encrypted:
        doc.close()
        return "", "The PDF is password-protected. Please upload an unencrypted document."
        
    full_text = []
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_text = page.get_text("text")
        if page_text:
            full_text.append(page_text)
            
    doc.close()
    combined_text = "\n".join(full_text).strip()
    
    if not combined_text or len(combined_text) < 20:
        return "", "No extractable text found. The document appears to be a scanned image or blank PDF without an embedded text layer."
        
    return combined_text, None


def extract_text_from_docx(file_bytes: bytes) -> Tuple[str, Optional[str]]:
    """
    Extracts text from DOCX bytes using python-docx.
    Iterates through both paragraphs and tables.
    Returns (extracted_text, error_message).
    """
    try:
        buffer = io.BytesIO(file_bytes)
        doc = docx.Document(buffer)
    except Exception as e:
        return "", f"Corrupt or unreadable DOCX file: {str(e)}"
        
    paragraphs_text = [p.text for p in doc.paragraphs if p.text.strip()]
    
    # Also extract any text in tables
    table_text = []
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    table_text.append(cell.text.strip())
                    
    combined = "\n".join(paragraphs_text + table_text).strip()
    if not combined or len(combined) < 20:
        return "", "The uploaded DOCX document contains no readable text."
        
    return combined, None


def extract_text_from_file(file_bytes: bytes, filename: str) -> Tuple[str, Optional[str]]:
    """
    Routes document extraction based on file extension.
    """
    fname = filename.lower()
    if fname.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif fname.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    else:
        return "", f"Unsupported file format '{filename}'. Please upload a .pdf or .docx file."


def screen_uploaded_resume(
    file_bytes: bytes,
    filename: str,
    job_row: pd.Series,
    job_embedding: np.ndarray,
    existing_ranked_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Full end-to-end processing pipeline for an uploaded resume:
    1. Extracts raw text from PDF/DOCX
    2. Cleans and normalizes text
    3. Extracts skills, degrees, and stated experience
    4. Encodes text into Sentence-BERT vector
    5. Calculates cosine similarity against selected job embedding
    6. Performs skill-gap analysis
    7. Evaluates candidate placement relative to existing dataset candidates
    """
    raw_text, error = extract_text_from_file(file_bytes, filename)
    if error:
        return {
            "success": False,
            "filename": filename,
            "error": error
        }
        
    cleaned_resume_text = clean_text(raw_text)
    
    # Skill & Entity Extraction
    extracted_skills = extract_skills_from_text(cleaned_resume_text)
    extracted_education = extract_education(cleaned_resume_text)
    extracted_experience_years = extract_years_experience(cleaned_resume_text)
    
    # Job requirements
    job_skills = parse_skills_field(job_row.get("clean_skills", []))
    
    # Skill-Gap Analysis
    skill_gap = compute_skill_gap(extracted_skills, job_skills)
    
    # Sentence-BERT Embedding & Cosine Similarity
    cand_embedding = encode_single_resume(cleaned_resume_text)
    similarity = float(compute_cosine_similarity(job_embedding, cand_embedding.reshape(1, -1))[0])
    similarity = round(similarity, 4)
    
    # Comparison against existing candidates if provided
    comparison_info: Dict[str, Any] = {}
    if existing_ranked_df is not None and not existing_ranked_df.empty and "similarity_score" in existing_ranked_df.columns:
        scores = existing_ranked_df["similarity_score"].values
        better_count = int((scores < similarity).sum())
        total_compared = len(scores)
        percentile = round((better_count / total_compared * 100.0), 1)
        
        # Placement rank
        placement_rank = int((scores >= similarity).sum()) + 1
        comparison_info = {
            "placement_rank": placement_rank,
            "total_candidates_compared": total_compared,
            "percentile": percentile,
            "top_existing_score": round(float(np.max(scores)), 4),
            "median_existing_score": round(float(np.median(scores)), 4)
        }
        
    return {
        "success": True,
        "filename": filename,
        "is_uploaded_candidate": True,
        "raw_text_length": len(raw_text),
        "cleaned_text": cleaned_resume_text,
        "extracted_skills": extracted_skills,
        "extracted_education": extracted_education,
        "extracted_years_experience": extracted_experience_years,
        "similarity_score": similarity,
        "skill_gap": skill_gap,
        "comparison": comparison_info
    }
