"""Candidate Blocking & Inverted Index Generation."""

import re
import unicodedata
import polars as pl
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import time

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    if text == "" or text.lower() in ("none", "null", "nan"):
        return ""
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    text = text.lower()
    text = re.sub(r'https?://|www\.|\.com|\.org|\.in|\.net|\.fr|\.io', ' ', text)
    text = re.sub(r'[^a-z0-9]', ' ', text)
    return ' '.join(text.split())

def get_name_blocking_keys(norm_name: str) -> List[str]:
    if not norm_name:
        return []
    tokens = norm_name.split()
    keys = []
    if len(tokens) >= 2:
        keys.append(f"n2:{tokens[0]}_{tokens[1]}")
    elif len(tokens) == 1:
        keys.append(f"n1:{tokens[0]}")
    if tokens and len(tokens[0]) >= 4:
        keys.append(f"nt0:{tokens[0]}")
    compact = norm_name.replace(" ", "")
    if len(compact) >= 5:
        keys.append(f"c5:{compact[:5]}")
    return keys

def get_address_blocking_keys(norm_addr: str) -> List[str]:
    if not norm_addr:
        return []
    tokens = norm_addr.split()
    nums = [t for t in tokens if t.isdigit()]
    words = [t for t in tokens if not t.isdigit() and len(t) >= 4]
    
    keys = []
    if nums and words:
        keys.append(f"a_nw:{nums[0]}_{words[0]}")
    if nums and len(words) >= 2:
        keys.append(f"a_nw2:{nums[0]}_{words[1]}")
    for num in nums:
        if len(num) in (5, 6):
            keys.append(f"pin:{num}")
    return keys

class FastMultiKeyBlocker:
    def __init__(self, max_candidates_per_s1: int = 12):
        self.max_candidates = max_candidates_per_s1
        self.corpus_records = {}
        self.inverted_index = defaultdict(list)
        
    def fit_corpus(self, corpus_df: pl.DataFrame):
        t0 = time.time()
        for row in corpus_df.iter_rows(named=True):
            cid = row["entity_id"]
            name = str(row.get("business_name") or "")
            addr = str(row.get("business_address") or "")
            
            norm_name = normalize_text(name)
            norm_addr = normalize_text(addr)
            
            self.corpus_records[cid] = {
                "entity_id": cid,
                "name": name,
                "addr": addr,
                "norm_name": norm_name,
                "norm_addr": norm_addr,
                "country": row.get("country")
            }
            
            for k in get_name_blocking_keys(norm_name):
                self.inverted_index[k].append(cid)
            for k in get_address_blocking_keys(norm_addr):
                self.inverted_index[k].append(cid)
                
    def query(self, s1_df: pl.DataFrame) -> Dict[str, List[str]]:
        candidate_map = {}
        for row in s1_df.iter_rows(named=True):
            s1_id = row["entity_id"]
            name = str(row.get("business_name") or "")
            addr = str(row.get("business_address") or "")
            
            norm_name = normalize_text(name)
            norm_addr = normalize_text(addr)
            
            candidate_counts = defaultdict(int)
            for k in get_name_blocking_keys(norm_name):
                matched = self.inverted_index.get(k)
                if matched:
                    for cid in matched:
                        candidate_counts[cid] += 3
            for k in get_address_blocking_keys(norm_addr):
                matched = self.inverted_index.get(k)
                if matched:
                    for cid in matched:
                        candidate_counts[cid] += 2
                        
            if candidate_counts:
                sorted_cands = sorted(candidate_counts.items(), key=lambda x: x[1], reverse=True)
                candidate_map[s1_id] = [cid for cid, _ in sorted_cands[:self.max_candidates]]
            else:
                candidate_map[s1_id] = []
        return candidate_map
