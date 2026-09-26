"""Benchmark blocking recall and reduction ratio on validation split."""

import sys
import os
sys.path.insert(0, os.path.abspath("."))
import time
import polars as pl
from src.fast_blocker import FastOptimizedBlocker

def benchmark_blocking():
    print("=" * 60)
    print("🔍 Benchmarking Candidate Blocking Engine")
    print("=" * 60)
    
    # 1. Load validation S1 and ground truth
    val_s1 = pl.read_csv("dataset/validation/val_source1.tsv", separator="\t").head(5000)
    val_gt = pl.read_csv("dataset/validation/val_ground_truth.tsv", separator="\t")
    
    val_s1_ids = set(val_s1["entity_id"].to_list())
    val_gt = val_gt.filter(pl.col("source1_entity_id").is_in(val_s1_ids))
    
    # Build GT mapping
    gt_map = {}
    for row in val_gt.iter_rows(named=True):
        m_str = row["matched_entity_ids"]
        if m_str and str(m_str).strip():
            gt_map[row["source1_entity_id"]] = set(x.strip() for x in str(m_str).split(",") if x.strip())
        else:
            gt_map[row["source1_entity_id"]] = set()
            
    # 2. Load Train S2 and S3 for corpus (filtered by countries present in val_s1)
    print("Loading Corpus (Train S2 + S3)...")
    t0 = time.time()
    df_s2 = pl.read_csv("student_resource/dataset/train/train_source2.tsv", separator="\t")
    df_s3 = pl.read_csv("student_resource/dataset/train/train_source3.tsv", separator="\t")
    df_corpus = pl.concat([df_s2, df_s3])
    print(f"Loaded {len(df_corpus):,} corpus records in {time.time() - t0:.2f}s.")
    
    # Partition by country
    blocker = FastOptimizedBlocker(max_candidates=60)
    blocker.fit_corpus(df_corpus)
    
    candidate_map = blocker.query(val_s1)
    
    # Measure Recall Ceiling
    total_true_matches = 0
    captured_matches = 0
    entities_with_gt = 0
    entities_with_candidates = 0
    total_candidates_generated = 0
    
    for s1_id, true_set in gt_map.items():
        cands = set(candidate_map.get(s1_id, []))
        total_candidates_generated += len(cands)
        if len(cands) > 0:
            entities_with_candidates += 1
            
        if len(true_set) > 0:
            entities_with_gt += 1
            total_true_matches += len(true_set)
            captured_matches += len(true_set.intersection(cands))
            
    recall_ceiling = captured_matches / total_true_matches if total_true_matches > 0 else 0.0
    avg_cands_per_entity = total_candidates_generated / len(val_s1)
    
    print("\n" + "=" * 60)
    print("📊 Blocking Benchmark Results:")
    print(f"   Evaluated S1 Entities: {len(val_s1):,}")
    print(f"   Total True Links: {total_true_matches:,}")
    print(f"   Captured Links in Candidates: {captured_matches:,}")
    print(f"   🎯 Candidate Recall Ceiling: {recall_ceiling * 100:.2f}%")
    print(f"   Average Candidates per Entity: {avg_cands_per_entity:.2f}")
    print(f"   Entities with Candidates: {entities_with_candidates:,} / {len(val_s1):,} ({entities_with_candidates/len(val_s1)*100:.2f}%)")
    print("=" * 60)

if __name__ == "__main__":
    benchmark_blocking()
