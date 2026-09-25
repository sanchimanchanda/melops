"""High-speed TSV data loader and universal normalizer."""

import os
import re
import unicodedata
import polars as pl
from typing import Dict, Any, List

def normalize_text(text: str) -> str:
    """Universal string cleaner (unicode accent normalization, lowercase, URL/punctuation stripping)."""
    if text is None:
        return ""
    text = str(text).strip()
    if text == "" or text.lower() in ("none", "null", "nan"):
        return ""
    # Normalize unicode accents
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    text = text.lower()
    # Strip URL patterns
    text = re.sub(r'https?://|www\.|\.com|\.org|\.in|\.net|\.fr|\.io', ' ', text)
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-z0-9]', ' ', text)
    return ' '.join(text.split())

def load_tsv(filepath: str) -> pl.DataFrame:
    """Loads TSV with strict tab separator and null handling."""
    return pl.read_csv(filepath, separator="\t", null_values=["", "null", "None", "NULL", "NONE"])
