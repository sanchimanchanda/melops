"""Benchmark FastOptimizedBlocker recall and throughput."""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
import polars as pl
from src.fast_blocker import FastOptimizedBlocker

def run_fast_benchmark():
    print("=" * 60)
    print("🚀 Benchmarking FastOptimizedBlocker on Validation Set")
    print("=" * 60)
    
    t0 = time.time()
    val_s1 = pl.read_csv("dataset/validation/val_source1.tsv", separator="\t").head(10000)
    val_gt = pl.read_csv("dataset/validation/val_ground_truth.tsv", separator="\t")
    
    val_s1_ids = set(val_s1["entity_id"].to_list())
    val_gt = val_gt.filter(pl.col("source1_entity_id").is_in(val_s1_ids))
    
    gt_map = {}
    for row in val_gt.iter_rows(named=True):
        m_str = row["matched_entity_ids"]
        if m_str and str(m_str).strip():
            gt_map[row["source1_entity_id"]] = set(x.strip() for x in str(m_str).split(",") if x.strip())
        else:
            gt_map[row["source1_entity_id"]] = set()
            
    print(f"Loaded {len(val_s1):,} validation queries.")
    
    # Load corpus
    print("Loading Corpus (Train S2 + S3)...")
    df_s2 = pl.read_csv("student_resource/dataset/train/train_source2.tsv", separator="\t")
    df_s3 = pl.read_csv("student_resource/dataset/train/train_source3.tsv", separator="\t")
    df_corpus = pl.concat([df_s2, df_s3])
    print(f"Corpus loaded: {len(df_corpus):,} records in {time.time() - t0:.2f}s.")
    
    # Fit blocker
    blocker = FastOptimizedBlocker(max_candidates=15)
    blocker.fit_corpus(df_corpus)
    
    # Query blocker
    candidate_map = blocker.query(val_s1)
    
    # Evaluate recall
    total_true_links = 0
    captured_links = 0
    total_candidates = 0
    
    for s1_id, true_set in gt_map.items():
        cands = set(candidate_map.get(s1_id, []))
        total_candidates += len(cands)
        if true_set:
            total_true_links += len(true_set)
            captured_links += len(true_set.intersection(cands))
            
    recall_ceiling = captured_links / total_true_links if total_true_links > 0 else 0.0
    avg_cands = total_candidates / len(val_s1)
    
    print("\n" + "=" * 60)
    print("🎯 Fast Blocker Results:")
    print(f"   Evaluated Queries: {len(val_s1):,}")
    print(f"   Total Ground Truth Links: {total_true_links:,}")
    print(f"   Captured Links: {captured_links:,}")
    print(f"   🎯 Candidate Recall Ceiling: {recall_ceiling * 100:.2f}%")
    print(f"   Average Candidates / Entity: {avg_cands:.2f}")
    print(f"   Total Execution Time: {time.time() - t0:.2f}s")
    print("=" * 60)

if __name__ == "__main__":
    run_fast_benchmark()
