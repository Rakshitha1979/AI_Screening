"""
Resume and Job Text Preprocessing Module.
Cleans and normalizes unstructured text from resumes and job descriptions,
preserving domain-specific technical terms, skill mentions, and section markers.
"""

import re
import html
from typing import Optional, Dict, Any
import pandas as pd


def clean_text(raw_text: Optional[str]) -> str:
    """
    Cleans raw resume or job description text.
    - Decodes HTML entities
    - Removes non-printable or noisy formatting characters
    - Normalizes excessive whitespace and line breaks
    - Preserves programming terms like C++, C#, .NET, Node.js
    """
    if raw_text is None or pd.isna(raw_text):
        return ""
    
    text = str(raw_text)
    # Decode HTML entities if present
    text = html.unescape(text)
    
    # Replace URLs and email addresses with neutral tokens or remove them
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' ', text)
    
    # Replace bullet characters and irregular separators with standard spaces
    text = re.sub(r'[\u2022\u2023\u25E6\u2043\u2219\u00B7\uF0A7\u25AA\u25CF\t\r\n]+', ' ', text)
    
    # Keep characters useful for skills: letters, numbers, +, #, ., -, /
    # Remove control characters and decorative symbols
    text = re.sub(r'[^\w\s\+\#\.\-\/]', ' ', text)
    
    # Collapse multiple dots that are not part of known extensions or numbers
    text = re.sub(r'\.{2,}', ' ', text)
    
    # Collapse repeated whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def parse_skills_field(skills_val: Any) -> list:
    """
    Safely parses clean_skills or clean_skills_json column into a list of strings.
    Handles JSON strings, Python string representations, or comma-separated lists.
    """
    if skills_val is None or pd.isna(skills_val):
        return []
    
    if isinstance(skills_val, list):
        return [str(s).strip().lower() for s in skills_val if str(s).strip()]
    
    val_str = str(skills_val).strip()
    
    # Check if JSON or python list syntax
    if (val_str.startswith('[') and val_str.endswith(']')) or (val_str.startswith('(') and val_str.endswith(')')):
        try:
            import json
            parsed = json.loads(val_str.replace("'", '"'))
            if isinstance(parsed, list):
                return [str(s).strip().lower() for s in parsed if str(s).strip()]
        except Exception:
            # Fallback regex extraction of quoted words
            items = re.findall(r'["\'](.*?)["\']', val_str)
            if items:
                return [item.strip().lower() for item in items if item.strip()]
    
    # Otherwise treat as comma or semicolon separated
    items = re.split(r'[,;]+', val_str)
    return [item.strip().lower() for item in items if item.strip()]


def load_and_preprocess_cvs(filepath: str = "data/cv.csv") -> pd.DataFrame:
    """
    Loads cv.csv, applies text normalization, and prepares structured fields.
    """
    df = pd.read_csv(filepath)
    df["clean_resume_text"] = df["resume_text"].fillna("").apply(clean_text)
    df["parsed_skills"] = df["clean_skills"].apply(parse_skills_field)
    df["years_experience"] = pd.to_numeric(df["years_experience"], errors="coerce").fillna(0.0)
    return df


def load_and_preprocess_jobs(filepath: str = "data/job.csv") -> pd.DataFrame:
    """
    Loads job.csv, applies text normalization, and prepares structured fields.
    """
    df = pd.read_csv(filepath)
    # Combine job_title, job_description, and job_text for rich context
    df["combined_job_text"] = (
        df["job_title"].fillna("") + " " +
        df["category"].fillna("") + " " +
        df["job_description"].fillna("")
    ).apply(clean_text)
    
    df["parsed_skills"] = df["clean_skills"].apply(parse_skills_field)
    df["years_required"] = pd.to_numeric(df["years_required"], errors="coerce").fillna(0.0)
    return df
