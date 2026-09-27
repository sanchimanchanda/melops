import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
import subprocess
import numpy as np
import polars as pl
from tqdm import tqdm
import joblib

from src.fast_blocker import FastOptimizedBlocker, fast_normalize
from src.feature_engine import compute_pairwise_features

def resume_inference():
    print("=" * 70)
    print("🚀 Resuming Test Inference from Saved Model")
    print("=" * 70)
    
    start_time = time.time()
    
    print("\n📦 Loading Model and Setting Threshold...")
    model = joblib.load("models/lightgbm_matcher.joblib")
    best_threshold = 0.850
    print(f"Loaded LightGBM model. Using Threshold: {best_threshold}")
    
    print("\n🚀 Running Test Inference (Partitioned by Country: US, India, France)...")
    test_s1_full = pl.read_csv("student_resource/dataset/test/test_source1.tsv", separator="\t")
    test_s2_full = pl.read_csv("student_resource/dataset/test/test_source2.tsv", separator="\t")
    test_s3_full = pl.read_csv("student_resource/dataset/test/test_source3.tsv", separator="\t")
    test_corpus_full = pl.concat([test_s2_full, test_s3_full])
    
    all_test_s1_ids = test_s1_full["entity_id"].to_list()
    final_matching_results = {s1_id: [] for s1_id in all_test_s1_ids}
    final_candidate_pairs = {s1_id: [] for s1_id in all_test_s1_ids}
    
    countries = test_s1_full["country"].unique().to_list()
    
    # We will prioritize US first, then France, then India.
    for country in ["US", "France", "India"]:
        if country not in countries:
            continue
            
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
    print("\n💾 Writing Output TSV Files...")
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
    print(f"\n🎉 Resume Pipeline Finished Successfully in {time.time() - start_time:.2f}s!")

if __name__ == "__main__":
    resume_inference()
