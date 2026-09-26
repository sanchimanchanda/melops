"""End-to-End Scalable Pipeline for Amazon ML Challenge 2026: Business Entity Resolution."""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
import subprocess
import numpy as np
import polars as pl
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from tqdm import tqdm

from src.fast_blocker import FastOptimizedBlocker, fast_normalize
from src.feature_engine import compute_pairwise_features, FEATURE_NAMES
from src.model import EntityMatchingModel
from src.calibrate import score_predictions_f05, calibrate_threshold

def build_training_dataset_country(
    train_s1_df: pl.DataFrame,
    df_corpus: pl.DataFrame,
    gt_map: Dict[str, Set[str]],
    country: str,
    max_negatives: int = 3
) -> Tuple[np.ndarray, np.ndarray, FastOptimizedBlocker]:
    """
    Builds training dataset for a specific country partition.
    """
    part_s1 = train_s1_df.filter(pl.col("country") == country)
    part_corpus = df_corpus.filter(pl.col("country") == country)
    print(f"   Building training data for [{country}]: {len(part_s1):,} S1 queries, {len(part_corpus):,} corpus rows...")
    
    blocker = FastOptimizedBlocker(max_candidates=100)
    blocker.fit_corpus(part_corpus)
    
    ids = part_s1["entity_id"].to_list()
    names = part_s1["business_name"].to_list()
    addrs = part_s1["business_address"].to_list()
    n_rows = len(ids)
    
    s1_records = {}
    for i in range(n_rows):
        s1_records[ids[i]] = (
            str(names[i] or ""),
            str(addrs[i] or ""),
            fast_normalize(names[i]),
            fast_normalize(addrs[i])
        )
        
    cand_map = blocker.query(part_s1)
    corpus_records = blocker.records
    
    X_list = []
    y_list = []
    
    for s1_id, s1_data in s1_records.items():
        s1_name, s1_addr, s1_nname, s1_naddr = s1_data
        true_matches = gt_map.get(s1_id, set())
        candidates = cand_map.get(s1_id, [])
        
        for mid in true_matches:
            if mid in corpus_records:
                c_nname, c_naddr = corpus_records[mid]
                feats = compute_pairwise_features(s1_name, s1_addr, s1_nname, s1_naddr, mid, "", c_nname, c_naddr)
                X_list.append(feats)
                y_list.append(1)
                
        # Hard Negatives
        neg_c = 0
        for cid in candidates:
            if cid not in true_matches and cid in corpus_records:
                c_nname, c_naddr = corpus_records[cid]
                feats = compute_pairwise_features(s1_name, s1_addr, s1_nname, s1_naddr, cid, "", c_nname, c_naddr)
                X_list.append(feats)
                y_list.append(0)
                neg_c += 1
                if neg_c >= max_negatives:
                    break
                    
    X = np.array(X_list, dtype=np.float32) if X_list else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    y = np.array(y_list, dtype=np.int32) if y_list else np.empty((0,), dtype=np.int32)
    print(f"   [{country}] Dataset: {X.shape[0]:,} pairs ({np.sum(y==1):,} pos, {np.sum(y==0):,} neg)")
    return X, y, blocker

def run_end_to_end_pipeline():
    print("=" * 70)
    print("🚀 Amazon ML Challenge 2026: End-to-End Business Entity Resolution Pipeline")
    print("=" * 70)
    
    start_time = time.time()
    os.makedirs("output", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # ---------------------------------------------------------
    # STAGE 1: Load Training & Validation Data
    # ---------------------------------------------------------
    print("\n📦 1. Loading Training & Validation Data...")
    train_s1_full = pl.read_csv("student_resource/dataset/train/train_source1.tsv", separator="\t")
    train_s2_full = pl.read_csv("student_resource/dataset/train/train_source2.tsv", separator="\t")
    train_s3_full = pl.read_csv("student_resource/dataset/train/train_source3.tsv", separator="\t")
    train_gt_full = pl.read_csv("student_resource/dataset/train/train_ground_truth.tsv", separator="\t")
    
    # Properly sample a stratified validation set to prevent threshold bias
    val_s1 = train_s1_full.sample(n=30000, seed=42)
    val_s1_ids = set(val_s1["entity_id"].to_list())
    val_gt = train_gt_full.filter(pl.col("source1_entity_id").is_in(val_s1_ids))
    
    val_gt_map = {}
    for row in val_gt.iter_rows(named=True):
        m_str = row["matched_entity_ids"]
        if m_str and str(m_str).strip():
            val_gt_map[row["source1_entity_id"]] = set(x.strip() for x in str(m_str).split(",") if x.strip())
        else:
            val_gt_map[row["source1_entity_id"]] = set()
            
    # Sample training S1 entities (excluding validation)
    train_s1_sample = train_s1_full.filter(~pl.col("entity_id").is_in(val_s1_ids)).head(1000000)
    sample_ids = set(train_s1_sample["entity_id"].to_list())
    train_gt_map = {}
    for row in train_gt_full.filter(pl.col("source1_entity_id").is_in(sample_ids)).iter_rows(named=True):
        m_str = row["matched_entity_ids"]
        if m_str and str(m_str).strip():
            train_gt_map[row["source1_entity_id"]] = set(x.strip() for x in str(m_str).split(",") if x.strip())
        else:
            train_gt_map[row["source1_entity_id"]] = set()
            
    train_corpus = pl.concat([train_s2_full, train_s3_full])
    print(f"Loaded Train S1 sample: {len(train_s1_sample):,}, Validation S1: {len(val_s1):,}, Corpus: {len(train_corpus):,}")
    
    # ---------------------------------------------------------
    # STAGE 2: Country-Partitioned Feature Extraction & Assembly
    # ---------------------------------------------------------
    print("\n🧠 2. Country-Partitioned Feature Extraction...")
    X_train_us, y_train_us, blocker_us = build_training_dataset_country(train_s1_sample, train_corpus, train_gt_map, "US", max_negatives=6)
    X_train_in, y_train_in, blocker_in = build_training_dataset_country(train_s1_sample, train_corpus, train_gt_map, "India", max_negatives=6)
    
    X_val_us, y_val_us, _ = build_training_dataset_country(val_s1, train_corpus, val_gt_map, "US", max_negatives=6)
    X_val_in, y_val_in, _ = build_training_dataset_country(val_s1, train_corpus, val_gt_map, "India", max_negatives=6)
    
    X_train = np.vstack([X_train_us, X_train_in])
    y_train = np.concatenate([y_train_us, y_train_in])
    X_val = np.vstack([X_val_us, X_val_in])
    y_val = np.concatenate([y_val_us, y_val_in])
    
    print(f"Combined Training Set: {X_train.shape[0]:,} pairs ({np.sum(y_train==1):,} pos, {np.sum(y_train==0):,} neg)")
    print(f"Combined Validation Set: {X_val.shape[0]:,} pairs ({np.sum(y_val==1):,} pos, {np.sum(y_val==0):,} neg)")
    
    # ---------------------------------------------------------
    # STAGE 3: Train LightGBM Matching Model
    # ---------------------------------------------------------
    print("\n📈 3. Training LightGBM Pairwise Classifier...")
    model = EntityMatchingModel()
    model.fit(X_train, y_train, X_val, y_val)
    model.save("models/lightgbm_matcher.joblib")
    
    # ---------------------------------------------------------
    # STAGE 4: Threshold Calibration on Validation Set
    # ---------------------------------------------------------
    print("\n🎯 4. Calibrating Probability Threshold on Validation Split...")
    scored_val_candidates = {}
    
    for country, country_blocker in [("US", blocker_us), ("India", blocker_in)]:
        part_val = val_s1.filter(pl.col("country") == country)
        cand_map = country_blocker.query(part_val)
        
        v_ids = part_val["entity_id"].to_list()
        v_names = part_val["business_name"].to_list()
        v_addrs = part_val["business_address"].to_list()
        
        for i in range(len(v_ids)):
            s1_id = v_ids[i]
            s1_name = str(v_names[i] or "")
            s1_addr = str(v_addrs[i] or "")
            s1_nname = fast_normalize(s1_name)
            s1_naddr = fast_normalize(s1_addr)
            
            candidates = cand_map.get(s1_id, [])
            cand_scores = []
            if candidates:
                cand_feats = []
                valid_cids = []
                for cid in candidates:
                    if cid in country_blocker.records:
                        c_nname, c_naddr = country_blocker.records[cid]
                        feats = compute_pairwise_features(s1_name, s1_addr, s1_nname, s1_naddr, cid, "", c_nname, c_naddr)
                        cand_feats.append(feats)
                        valid_cids.append(cid)
                if cand_feats:
                    probs = model.predict_proba(np.array(cand_feats, dtype=np.float32))
                    for cid, p in zip(valid_cids, probs):
                        cand_scores.append((cid, float(p)))
            scored_val_candidates[s1_id] = cand_scores
            
    best_threshold, best_val_f05 = calibrate_threshold(val_gt_map, scored_val_candidates)
    print(f"   🏆 Optimal Decision Threshold: {best_threshold:.3f}")
    print(f"   🎯 Validation Macro F0.5 Score: {best_val_f05:.4f}")
    
    # ---------------------------------------------------------
    # STAGE 5: Test Inference Across Country Partitions
    # ---------------------------------------------------------
    print("\n🚀 5. Running Test Inference (Partitioned by Country: US, India, France)...")
    test_s1_full = pl.read_csv("student_resource/dataset/test/test_source1.tsv", separator="\t")
    test_s2_full = pl.read_csv("student_resource/dataset/test/test_source2.tsv", separator="\t")
    test_s3_full = pl.read_csv("student_resource/dataset/test/test_source3.tsv", separator="\t")
    test_corpus_full = pl.concat([test_s2_full, test_s3_full])
    
    all_test_s1_ids = test_s1_full["entity_id"].to_list()
    final_matching_results = {s1_id: [] for s1_id in all_test_s1_ids}
    final_candidate_pairs = {s1_id: [] for s1_id in all_test_s1_ids}
    
    countries = test_s1_full["country"].unique().to_list()
    print(f"   Test Country Partitions: {countries}")
    
    for country in countries:
        print(f"\n   -> Processing Country Partition: [{country}]")
        part_s1 = test_s1_full.filter(pl.col("country") == country)
        part_corpus = test_corpus_full.filter(pl.col("country") == country)
        print(f"      S1: {len(part_s1):,} queries | Corpus (S2+S3): {len(part_corpus):,} records")
        
        country_blocker = FastOptimizedBlocker(max_candidates=100)
        country_blocker.fit_corpus(part_corpus)
        cand_map = country_blocker.query(part_s1)
        
        p_ids = part_s1["entity_id"].to_list()
        p_names = part_s1["business_name"].to_list()
        p_addrs = part_s1["business_address"].to_list()
        n_part = len(p_ids)
        
        print(f"      Scoring candidate pairs for [{country}]...")
        for i in tqdm(range(n_part), desc=f"Scoring {country}"):
            s1_id = p_ids[i]
            s1_name = str(p_names[i] or "")
            s1_addr = str(p_addrs[i] or "")
            s1_nname = fast_normalize(s1_name)
            s1_naddr = fast_normalize(s1_addr)
            
            candidates = cand_map.get(s1_id, [])
            final_candidate_pairs[s1_id] = candidates
            
            if candidates:
                cand_feats = []
                valid_cids = []
                for cid in candidates:
                    if cid in country_blocker.records:
                        c_nname, c_naddr = country_blocker.records[cid]
                        feats = compute_pairwise_features(s1_name, s1_addr, s1_nname, s1_naddr, cid, "", c_nname, c_naddr)
                        cand_feats.append(feats)
                        valid_cids.append(cid)
                if cand_feats:
                    probs = model.predict_proba(np.array(cand_feats, dtype=np.float32))
                    matched_ids = [cid for cid, p in zip(valid_cids, probs) if p >= best_threshold]
                    final_matching_results[s1_id] = matched_ids
                    
    # ---------------------------------------------------------
    # STAGE 6: Write Output TSV Files & Pre-Flight Validation
    # ---------------------------------------------------------
    print("\n💾 6. Writing Output TSV Files...")
    matching_tsv_path = "output/matching_results.tsv"
    candidate_tsv_path = "output/candidate_pairs.tsv"
    
    with open(matching_tsv_path, "w", encoding="utf-8") as f_m, open(candidate_tsv_path, "w", encoding="utf-8") as f_c:
        f_m.write("source1_entity_id\tmatched_entity_ids\n")
        f_c.write("source1_entity_id\tcandidate_entity_ids\n")
        
        for s1_id in all_test_s1_ids:
            m_list = final_matching_results.get(s1_id, [])
            c_list = final_candidate_pairs.get(s1_id, [])
            
            f_m.write(f"{s1_id}\t{','.join(m_list)}\n")
            f_c.write(f"{s1_id}\t{','.join(c_list)}\n")
            
    print(f"   Successfully written: {matching_tsv_path} and {candidate_tsv_path}")
    
    # ---------------------------------------------------------
    # STAGE 7: Run validate_submission.py
    # ---------------------------------------------------------
    print("\n🔍 7. Running Submission Validation Check...")
    val_cmd = [
        "python3",
        "student_resource/utils/validate_submission.py",
        "--matching", matching_tsv_path,
        "--candidate", candidate_tsv_path,
        "--test-dir", "student_resource/dataset/test"
    ]
    res = subprocess.run(val_cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)
        
    print(f"\n🎉 Pipeline Finished Successfully in {time.time() - start_time:.2f}s!")

if __name__ == "__main__":
    run_end_to_end_pipeline()
