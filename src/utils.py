"""
Utility functions for the extraction pipeline.
"""

import re
from typing import Optional, List, Dict, Any
from pathlib import Path
import hashlib


def clean_doi(doi: str) -> str:
    """Clean and normalize DOI format."""
    if not doi:
        return ""
    
    # Remove common prefixes
    doi = doi.strip()
    doi = re.sub(r'^https?://doi\.org/', '', doi)
    doi = re.sub(r'^doi:', '', doi, flags=re.IGNORECASE)
    
    # Remove spaces
    doi = doi.replace(' ', '')
    
    return doi


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for filesystem compatibility."""
    # Replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Limit length
    if len(filename) > 200:
        filename = filename[:200]
    
    return filename


def extract_author_names(author_string: str) -> List[str]:
    """Extract individual author names from author string."""
    if not author_string:
        return []
    
    # Common separators
    if ';' in author_string:
        authors = author_string.split(';')
    elif ',' in author_string and author_string.count(',') > 2:
        # Likely comma-separated list
        authors = re.split(r',\s*(?=[A-Z])', author_string)
    else:
        # Assume single author
        authors = [author_string]
    
    # Clean each author name
    cleaned = []
    for author in authors:
        author = author.strip()
        if author and len(author) > 2:
            cleaned.append(author)
    
    return cleaned


def guess_gender_from_name(first_name: str) -> str:
    """
    Simple heuristic to guess gender from first name.
    Returns 'Male', 'Female', or 'Unknown'.
    
    Note: This is a basic implementation and should be replaced
    with a proper gender detection library for production use.
    """
    if not first_name:
        return "Unknown"
    
    first_name = first_name.lower().strip()
    
    # Common endings (very basic heuristic)
    female_endings = ['a', 'e', 'ie', 'y', 'ine', 'elle']
    male_endings = ['o', 'n', 'k', 'r', 'd']
    
    # Common names (small sample)
    common_female = {
        'mary', 'patricia', 'jennifer', 'linda', 'elizabeth',
        'barbara', 'susan', 'jessica', 'sarah', 'karen',
        'nancy', 'betty', 'helen', 'sandra', 'donna',
        'carol', 'ruth', 'sharon', 'michelle', 'laura'
    }
    
    common_male = {
        'james', 'john', 'robert', 'michael', 'william',
        'david', 'richard', 'joseph', 'thomas', 'charles',
        'christopher', 'daniel', 'matthew', 'anthony', 'donald',
        'mark', 'kenneth', 'steven', 'paul', 'andrew'
    }
    
    # Check against known names
    if first_name in common_female:
        return "Female"
    if first_name in common_male:
        return "Male"
    
    # Check endings (very unreliable)
    for ending in female_endings:
        if first_name.endswith(ending):
            return "Female"
    
    for ending in male_endings:
        if first_name.endswith(ending):
            return "Male"
    
    return "Unknown"


def extract_country_from_affiliation(affiliation: str) -> Optional[str]:
    """Extract country from affiliation string."""
    if not affiliation:
        return None
    
    # Common country patterns
    countries = {
        'USA', 'United States', 'U.S.A', 'U.S.', 'US',
        'UK', 'United Kingdom', 'England', 'Scotland', 'Wales',
        'Canada', 'Australia', 'Germany', 'France', 'Italy',
        'Spain', 'Netherlands', 'Belgium', 'Switzerland',
        'Sweden', 'Norway', 'Denmark', 'Finland',
        'Japan', 'China', 'Korea', 'India', 'Singapore',
        'Brazil', 'Mexico', 'Argentina', 'Chile',
        'Israel', 'Turkey', 'Egypt', 'South Africa'
    }
    
    affiliation_lower = affiliation.lower()
    
    for country in countries:
        if country.lower() in affiliation_lower:
            # Normalize country names
            if country in ['U.S.A', 'U.S.', 'US', 'United States']:
                return 'USA'
            elif country in ['UK', 'England', 'Scotland', 'Wales']:
                return 'United Kingdom'
            else:
                return country
    
    # Check for email domains that might indicate country
    if '.edu' in affiliation:
        return 'USA'
    elif '.ac.uk' in affiliation:
        return 'United Kingdom'
    elif '.ca' in affiliation:
        return 'Canada'
    elif '.au' in affiliation:
        return 'Australia'
    
    return None


def calculate_text_hash(text: str) -> str:
    """Calculate hash of text for deduplication."""
    return hashlib.md5(text.encode()).hexdigest()


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def format_doi_url(doi: str) -> str:
    """Format DOI as a URL."""
    clean = clean_doi(doi)
    if clean:
        return f"https://doi.org/{clean}"
    return ""


def estimate_tokens(text: str) -> int:
    """
    Rough estimate of token count for text.
    Approximation: ~1 token per 4 characters.
    """
    return len(text) // 4


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """Truncate text to maximum length with suffix."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix