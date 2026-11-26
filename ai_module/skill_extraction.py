"""
Skill and Entity Extraction Module.
Uses a curated, multi-domain technical and functional skills taxonomy,
coupled with phrase matching, regex entity detection for education and experience,
without falsely claiming a custom-trained NER model.
"""

import re
from typing import List, Dict, Any, Set, Tuple
from ai_module.resume_processing import parse_skills_field

# Comprehensive, multi-domain skills taxonomy
SKILLS_TAXONOMY: Dict[str, List[str]] = {
    # Programming Languages
    "python": ["python", "python3", "py"],
    "java": ["java", "core java", "j2ee"],
    "javascript": ["javascript", "js", "ecmascript"],
    "typescript": ["typescript", "ts"],
    "c++": ["c++", "cpp"],
    "c#": ["c#", "csharp", "c sharp"],
    "c": ["c language", "ansi c"],
    "go": ["golang", "go language"],
    "rust": ["rust"],
    "ruby": ["ruby", "ruby on rails"],
    "php": ["php"],
    "scala": ["scala"],
    "kotlin": ["kotlin"],
    "swift": ["swift"],
    "r": ["r language", "r programming"],
    "sql": ["sql", "structured query language", "tsql", "pl/sql", "plsql"],
    "bash": ["bash", "shell scripting", "powershell"],
    
    # Web & Full Stack Frameworks
    "react": ["react", "react.js", "reactjs"],
    "angular": ["angular", "angularjs", "angular 2+"],
    "vue": ["vue", "vue.js", "vuejs"],
    "node.js": ["node.js", "nodejs", "node"],
    "express": ["express", "express.js", "expressjs"],
    "django": ["django"],
    "flask": ["flask"],
    "fastapi": ["fastapi"],
    "spring": ["spring", "spring boot", "spring framework"],
    "asp.net": ["asp.net", ".net", ".net core", "dotnet"],
    "html/css": ["html", "html5", "css", "css3", "sass", "bootstrap", "tailwind"],
    "graphql": ["graphql"],
    "rest api": ["rest api", "restful api", "rest", "web services", "apis"],
    
    # Data Science & Machine Learning
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "neural networks", "dl"],
    "natural language processing": ["natural language processing", "nlp", "text mining"],
    "computer vision": ["computer vision", "cv", "object detection", "image processing"],
    "tensorflow": ["tensorflow", "tf"],
    "pytorch": ["pytorch", "torch"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "keras": ["keras"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "scipy": ["scipy"],
    "spark": ["spark", "apache spark", "pyspark"],
    "hadoop": ["hadoop", "hdfs", "mapreduce", "hive", "yarn"],
    "data science": ["data science"],
    "large language models": ["llm", "large language models", "generative ai", "transformers", "huggingface"],
    
    # Databases & Big Data
    "postgresql": ["postgresql", "postgres"],
    "mysql": ["mysql"],
    "mongodb": ["mongodb", "mongo"],
    "redis": ["redis"],
    "cassandra": ["cassandra"],
    "elasticsearch": ["elasticsearch", "elastic search"],
    "oracle": ["oracle db", "oracle database"],
    "sql server": ["sql server", "ms sql", "mssql"],
    "snowflake": ["snowflake"],
    "bigquery": ["bigquery"],
    "dynamodb": ["dynamodb"],
    
    # Cloud, DevOps & Tools
    "aws": ["aws", "amazon web services", "ec2", "s3", "lambda"],
    "azure": ["azure", "microsoft azure"],
    "gcp": ["gcp", "google cloud", "google cloud platform"],
    "docker": ["docker", "containerization"],
    "kubernetes": ["kubernetes", "k8s"],
    "terraform": ["terraform", "iac", "infrastructure as code"],
    "ci/cd": ["ci/cd", "continuous integration", "continuous deployment", "jenkins", "gitlab ci", "github actions"],
    "git": ["git", "github", "gitlab", "bitbucket"],
    "linux": ["linux", "unix", "ubuntu", "centos", "redhat"],
    "ansible": ["ansible"],
    
    # Analytics & Business Intelligence
    "data analysis": ["data analysis", "data analytics"],
    "powerbi": ["powerbi", "power bi"],
    "tableau": ["tableau"],
    "excel": ["excel", "advanced excel", "vba"],
    "business intelligence": ["business intelligence", "bi"],
    "data visualization": ["data visualization"],
    "sas": ["sas"],
    "etl": ["etl", "data pipeline", "data warehousing"],
    
    # Management, QA & Soft Skills
    "agile": ["agile", "scrum", "kanban", "sprint planning"],
    "jira": ["jira", "confluence"],
    "software development": ["software development", "software engineering", "sdlc"],
    "unit testing": ["unit testing", "test automation", "qa", "selenium", "pytest", "junit"],
    "system design": ["system design", "microservices", "software architecture"],
    "project management": ["project management", "pmp", "stakeholder management"],
    "communication": ["communication skills", "leadership", "team leadership", "problem solving"]
}

# Compile canonical lookup dictionary: alias -> canonical skill name
CANONICAL_SKILL_LOOKUP: Dict[str, str] = {}
for canonical, aliases in SKILLS_TAXONOMY.items():
    CANONICAL_SKILL_LOOKUP[canonical] = canonical
    for alias in aliases:
        CANONICAL_SKILL_LOOKUP[alias.lower()] = canonical

# Sort aliases by word length descending so multi-word skills (e.g. "machine learning") match first
SORTED_SKILL_PATTERNS = sorted(CANONICAL_SKILL_LOOKUP.keys(), key=lambda s: (-len(s.split()), -len(s)))


def extract_skills_from_text(text: str) -> List[str]:
    """
    Extracts canonical skills from free-form text using dictionary-guided phrase matching
    with boundary validation.
    """
    if not text:
        return []
    
    text_lower = " " + text.lower() + " "
    # Normalize punctuation separators to spaces for clean token-boundary matching
    normalized = re.sub(r'[/,;()]+', ' ', text_lower)
    
    matched_canonical: Set[str] = set()
    
    for alias in SORTED_SKILL_PATTERNS:
        # Construct pattern respecting non-word boundaries or specific tech symbols
        if alias in ["c++", "c#", ".net"]:
            escaped = re.escape(alias)
            pattern = rf'(?:^|[\s,;]){escaped}(?:[\s,;]|$)'
        elif len(alias) <= 2:
            # Short acronyms: require strict boundary
            escaped = re.escape(alias)
            pattern = rf'\b{escaped}\b'
        else:
            escaped = re.escape(alias)
            pattern = rf'\b{escaped}\b'
            
        if re.search(pattern, normalized):
            matched_canonical.add(CANONICAL_SKILL_LOOKUP[alias])
            
    return sorted(list(matched_canonical))


def extract_education(text: str) -> List[str]:
    """
    Extracts recognized education degrees and credentials using regex pattern matching.
    """
    if not text:
        return []
        
    degree_patterns = [
        (r'\b(ph\.?d|doctor\s+of\s+philosophy|doctorate)\b', "Ph.D. / Doctorate"),
        (r'\b(m\.?tech|master\s+of\s+technology)\b', "M.Tech"),
        (r'\b(m\.?s|master\s+of\s+science)\b', "M.S."),
        (r'\b(m\.?b\.?a|master\s+of\s+business\s+administration)\b', "MBA"),
        (r'\b(master[\'s]*\s+degree|masters)\b', "Master's Degree"),
        (r'\b(b\.?tech|bachelor\s+of\s+technology)\b', "B.Tech"),
        (r'\b(b\.?s|b\.?sc|bachelor\s+of\s+science)\b', "B.S. / B.Sc"),
        (r'\b(b\.?e|bachelor\s+of\s+engineering)\b', "B.E."),
        (r'\b(b\.?a|bachelor\s+of\s+arts)\b', "B.A."),
        (r'\b(bachelor[\'s]*\s+degree|bachelors)\b', "Bachelor's Degree"),
        (r'\b(diploma|associate[\'s]*\s+degree)\b', "Diploma / Associate")
    ]
    
    found_degrees: List[str] = []
    text_lower = text.lower()
    for pattern, label in degree_patterns:
        if re.search(pattern, text_lower):
            if label not in found_degrees:
                found_degrees.append(label)
                
    return found_degrees


def extract_years_experience(text: str) -> float:
    """
    Heuristically extracts stated years of experience from text (e.g. '5+ years of experience').
    """
    if not text:
        return 0.0
    
    patterns = [
        r'(\d+(?:\.\d+)?)\s*(?:\+)?\s*(?:years?|yrs?)(?:\s+of)?\s+experience\b',
        r'experience\s*(?:of|:)?\s*(\d+(?:\.\d+)?)\s*(?:\+)?\s*(?:years?|yrs?)\b',
        r'over\s+(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\b'
    ]
    
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            try:
                val = float(match.group(1))
                if 0.0 <= val <= 50.0:
                    return val
            except (ValueError, IndexError):
                pass
    return 0.0


def compute_skill_gap(candidate_skills: List[str], required_skills: List[str]) -> Dict[str, Any]:
    """
    Compares candidate skills against job required skills.
    Calculates matching skills, missing skills, and coverage percentage.
    """
    cand_set = set([s.lower() for s in candidate_skills])
    req_set = set([s.lower() for s in required_skills])
    
    matching = sorted(list(cand_set.intersection(req_set)))
    missing = sorted(list(req_set - cand_set))
    extra = sorted(list(cand_set - req_set))
    
    coverage = (len(matching) / len(req_set) * 100.0) if req_set else 0.0
    
    return {
        "required_skills": sorted(list(req_set)),
        "candidate_skills": sorted(list(cand_set)),
        "matching_skills": matching,
        "missing_skills": missing,
        "additional_skills": extra,
        "match_count": len(matching),
        "required_count": len(req_set),
        "coverage_percentage": round(coverage, 1)
    }
