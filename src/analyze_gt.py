"""Phase 1.5: Detailed Analysis of Ground Truth Matching Structure."""

import polars as pl
import numpy as np

def analyze_matching_structure():
    df_s1 = pl.read_csv("student_resource/dataset/train/train_source1.tsv", separator="\t")
    df_gt = pl.read_csv("student_resource/dataset/train/train_ground_truth.tsv", separator="\t")
    
    s1_countries = dict(zip(df_s1["entity_id"].to_list(), df_s1["country"].to_list()))
    
    s2_count = 0
    s3_count = 0
    both_count = 0
    total_non_singleton = 0
    
    for row in df_gt.iter_rows(named=True):
        m_str = row["matched_entity_ids"]
        if m_str is not None and str(m_str).strip() != "":
            ids = [x.strip() for x in str(m_str).split(",") if x.strip()]
            has_s2 = any(x.startswith("S2-") for x in ids)
            has_s3 = any(x.startswith("S3-") for x in ids)
            total_non_singleton += 1
            if has_s2 and has_s3:
                both_count += 1
            elif has_s2:
                s2_count += 1
            elif has_s3:
                s3_count += 1
                
    print(f"Total non-singletons: {total_non_singleton:,}")
    print(f"Entities matched with BOTH S2 and S3: {both_count:,} ({both_count/total_non_singleton*100:.2f}%)")
    print(f"Entities matched with S2 ONLY: {s2_count:,} ({s2_count/total_non_singleton*100:.2f}%)")
    print(f"Entities matched with S3 ONLY: {s3_count:,} ({s3_count/total_non_singleton*100:.2f}%)")

if __name__ == "__main__":
    analyze_matching_structure()
