"""Phase 3: High-Performance Vectorized Feature Engineering Engine."""

import re
import numpy as np
from rapidfuzz import fuzz, distance
from typing import Dict, List, Tuple, Any

def extract_numeric_tokens(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r'\b\d+\b', str(text))

def compute_pairwise_features(
    s1_name: str,
    s1_addr: str,
    s1_norm_name: str,
    s1_norm_addr: str,
    cand_name: str,
    cand_addr: str,
    cand_norm_name: str,
    cand_norm_addr: str,
    s1_country: str = "",
    cand_country: str = ""
) -> List[float]:
    s1_norm_name = s1_norm_name or ""
    cand_norm_name = cand_norm_name or ""
    s1_norm_addr = s1_norm_addr or ""
    cand_norm_addr = cand_norm_addr or ""
    
    # 1. Name Similarities
    name_lev_ratio = distance.Levenshtein.normalized_similarity(s1_norm_name, cand_norm_name)
    name_jaro_winkler = distance.JaroWinkler.similarity(s1_norm_name, cand_norm_name)
    name_token_sort = fuzz.token_sort_ratio(s1_norm_name, cand_norm_name) / 100.0
    name_token_set = fuzz.token_set_ratio(s1_norm_name, cand_norm_name) / 100.0
    name_partial_ratio = fuzz.partial_ratio(s1_norm_name, cand_norm_name) / 100.0
    
    # Prefix / Substring match
    min_len = min(len(s1_norm_name), len(cand_norm_name))
    is_prefix = 1.0 if min_len >= 4 and (s1_norm_name.startswith(cand_norm_name[:min_len]) or cand_norm_name.startswith(s1_norm_name[:min_len])) else 0.0
    
    # 2. Address Similarities
    has_both_addr = 1.0 if (s1_norm_addr and cand_norm_addr) else 0.0
    if has_both_addr:
        addr_lev_ratio = distance.Levenshtein.normalized_similarity(s1_norm_addr, cand_norm_addr)
        addr_jaro_winkler = distance.JaroWinkler.similarity(s1_norm_addr, cand_norm_addr)
        addr_token_sort = fuzz.token_sort_ratio(s1_norm_addr, cand_norm_addr) / 100.0
        addr_token_set = fuzz.token_set_ratio(s1_norm_addr, cand_norm_addr) / 100.0
        addr_partial_ratio = fuzz.partial_ratio(s1_norm_addr, cand_norm_addr) / 100.0
    else:
        addr_lev_ratio = 0.0
        addr_jaro_winkler = 0.0
        addr_token_sort = 0.0
        addr_token_set = 0.0
        addr_partial_ratio = 0.0
        
    # 3. Numeric / PIN / House Number Matches
    s1_nums = set(extract_numeric_tokens(s1_norm_addr + " " + s1_norm_name))
    cand_nums = set(extract_numeric_tokens(cand_norm_addr + " " + cand_norm_name))
    
    common_nums = s1_nums.intersection(cand_nums)
    num_common_digits = len(common_nums)
    num_jaccard = (len(common_nums) / len(s1_nums.union(cand_nums))) if (s1_nums or cand_nums) else 0.0
    has_exact_pin_match = 1.0 if any(len(n) in (5, 6) for n in common_nums) else 0.0
    
    # 4. Token Overlap Jaccard
    s1_tokens = set(s1_norm_name.split())
    cand_tokens = set(cand_norm_name.split())
    token_overlap_count = len(s1_tokens.intersection(cand_tokens))
    token_jaccard = (token_overlap_count / len(s1_tokens.union(cand_tokens))) if (s1_tokens or cand_tokens) else 0.0
    
    # 5. Length Differentials
    len_ratio_name = (min_len / max(len(s1_norm_name), len(cand_norm_name))) if max(len(s1_norm_name), len(cand_norm_name)) > 0 else 1.0
    
    # 6. Combined Full-Text Match
    full_s1 = s1_norm_name + " " + s1_norm_addr
    full_cand = cand_norm_name + " " + cand_norm_addr
    full_token_set = fuzz.token_set_ratio(full_s1, full_cand) / 100.0
    
    # 7. Source ID indicator
    is_s2 = 1.0 if str(cand_name).startswith("S2-") else 0.0
    
    return [
        name_lev_ratio,
        name_jaro_winkler,
        name_token_sort,
        name_token_set,
        name_partial_ratio,
        is_prefix,
        has_both_addr,
        addr_lev_ratio,
        addr_jaro_winkler,
        addr_token_sort,
        addr_token_set,
        addr_partial_ratio,
        float(num_common_digits),
        num_jaccard,
        has_exact_pin_match,
        float(token_overlap_count),
        token_jaccard,
        len_ratio_name,
        full_token_set,
        is_s2
    ]

FEATURE_NAMES = [
    "name_lev_ratio",
    "name_jaro_winkler",
    "name_token_sort",
    "name_token_set",
    "name_partial_ratio",
    "is_prefix",
    "has_both_addr",
    "addr_lev_ratio",
    "addr_jaro_winkler",
    "addr_token_sort",
    "addr_token_set",
    "addr_partial_ratio",
    "num_common_digits",
    "num_jaccard",
    "has_exact_pin_match",
    "token_overlap_count",
    "token_jaccard",
    "len_ratio_name",
    "full_token_set",
    "is_s2"
]
