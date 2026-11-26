"""
Natural Language Processing (NLP) Module for Resumes and Job Descriptions.
Provides tokenization, stopword filtering, vocabulary statistics,
and pipeline step explanations for dashboard display.
"""

import re
from typing import List, Dict, Any, Set
from collections import Counter

# Standard English stopwords with technical exceptions preserved
STANDARD_STOPWORDS: Set[str] = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
    'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
    'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll',
    'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
    'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
    "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

# Technical terms that might otherwise look like single letters or stopwords
PROTECTED_TECH_TERMS: Set[str] = {
    'c', 'r', 'go', 'net', 'ai', 'ml', 'bi', 'ui', 'ux', 'qa', 'os', 'ip', 'db',
    'sql', 'c++', 'c#', '.net', 'node.js', 'vue.js', 'react.js'
}


def tokenize(text: str) -> List[str]:
    """
    Tokenizes text into words, preserving programming languages and compound terms.
    """
    if not text:
        return []
    # Match words, including terms with ++ or # or .
    pattern = r'[a-zA-Z0-9\+\#\.\-]+'
    tokens = re.findall(pattern, text.lower())
    # Strip leading/trailing dots or hyphens from tokens unless it's a known identifier
    cleaned_tokens = []
    for t in tokens:
        stripped = t.strip('.-')
        if stripped in PROTECTED_TECH_TERMS or len(stripped) > 1:
            cleaned_tokens.append(stripped)
    return cleaned_tokens


def remove_stopwords(tokens: List[str]) -> List[str]:
    """
    Filters out common stopwords while keeping technical keywords intact.
    """
    return [
        t for t in tokens 
        if t in PROTECTED_TECH_TERMS or (t not in STANDARD_STOPWORDS and len(t) > 1)
    ]


def compute_text_stats(text: str) -> Dict[str, Any]:
    """
    Computes linguistic metrics for resume or job text.
    """
    raw_words = text.split()
    tokens = tokenize(text)
    filtered = remove_stopwords(tokens)
    vocab = set(filtered)
    
    lexical_diversity = round(len(vocab) / len(filtered), 3) if filtered else 0.0
    
    return {
        "raw_word_count": len(raw_words),
        "token_count": len(tokens),
        "filtered_keyword_count": len(filtered),
        "unique_vocabulary": len(vocab),
        "lexical_diversity": lexical_diversity,
        "top_keywords": [word for word, count in Counter(filtered).most_common(10)]
    }


def explain_nlp_pipeline() -> List[Dict[str, str]]:
    """
    Returns an educational explanation of each step in the NLP pipeline
    suitable for display in the recruiter dashboard.
    """
    return [
        {
            "step": "1. Text Sanitization",
            "description": "Removes markup, HTML artifacts, bullet symbols, contact emails, and URLs while preserving technical characters (+, #, .)."
        },
        {
            "step": "2. Domain-Aware Tokenization",
            "description": "Splits text into coherent tokens, safely protecting acronyms and programming names like C++, C#, .NET, and R."
        },
        {
            "step": "3. Selective Stopword Filtering",
            "description": "Strips non-informative grammatical glue words while retaining short domain acronyms (e.g. AI, ML, UI, OS)."
        },
        {
            "step": "4. Vector Representation",
            "description": "Encodes normalized text into dense semantic vector representations for high-dimensional relevance calculation."
        }
    ]
