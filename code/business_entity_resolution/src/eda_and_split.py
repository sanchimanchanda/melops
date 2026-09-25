"""Phase 1: Exploratory Data Analysis, Distribution Profiling & Validation Split."""

import os
import sys
import time
import json
import polars as pl
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple

def compute_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]]
) -> Tuple[float, float, float, Dict[str, float]]:
    """
    Computes exact competition Macro F0.5 score across all reference S1 entities.
    
    Formula per entity:
        F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    
    Singletons:
        - If GT is empty and Pred is empty -> F0.5 = 1.0 (correct singleton)
        - If GT is empty and Pred is non-empty -> F0.5 = 0.0 (false merge)
        - If GT is non-empty and Pred is empty -> F0.5 = 0.0 (missed match)
    """
    f05_scores = []
    precisions = []
    recalls = []
    
    for s1_id, gt_matches in ground_truth.items():
        pred_matches = predictions.get(s1_id, set())
        
        # Singleton case (GT has no matches)
        if len(gt_matches) == 0:
            if len(pred_matches) == 0:
                f05 = 1.0
                p, r = 1.0, 1.0
            else:
                f05 = 0.0
                p, r = 0.0, 1.0
        else:
            if len(pred_matches) == 0:
                f05 = 0.0
                p, r = 1.0, 0.0
            else:
                tp = len(gt_matches.intersection(pred_matches))
                fp = len(pred_matches - gt_matches)
                fn = len(gt_matches - pred_matches)
                
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                
                if (0.25 * p + r) > 0:
                    f05 = (1.25 * p * r) / (0.25 * p + r)
                else:
                    f05 = 0.0
        
        f05_scores.append(f05)
        precisions.append(p)
        recalls.append(r)
        
    macro_f05 = float(np.mean(f05_scores))
    macro_p = float(np.mean(precisions))
    macro_r = float(np.mean(recalls))
    
    return macro_f05, macro_p, macro_r, {
        "macro_f05": macro_f05,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "num_entities": len(ground_truth)
    }

def run_eda(data_dir: str):
    print("=" * 60)
    print("🚀 Running EDA on Train & Test Datasets")
    print("=" * 60)
    
    t0 = time.time()
    
    train_s1_path = os.path.join(data_dir, "train", "train_source1.tsv")
    train_s2_path = os.path.join(data_dir, "train", "train_source2.tsv")
    train_s3_path = os.path.join(data_dir, "train", "train_source3.tsv")
    train_gt_path = os.path.join(data_dir, "train", "train_ground_truth.tsv")
    
    test_s1_path = os.path.join(data_dir, "test", "test_source1.tsv")
    test_s2_path = os.path.join(data_dir, "test", "test_source2.tsv")
    test_s3_path = os.path.join(data_dir, "test", "test_source3.tsv")
    
    print("1. Loading Train Source 1...")
    df_s1 = pl.read_csv(train_s1_path, separator="\t", infer_schema_length=10000, null_values=[""])
    print(f"   Train S1 shape: {df_s1.shape}")
    print(f"   Country breakdown in Train S1:\n{df_s1['country'].value_counts()}")
    
    print("2. Loading Ground Truth...")
    df_gt = pl.read_csv(train_gt_path, separator="\t", infer_schema_length=10000, null_values=[""])
    print(f"   Train GT shape: {df_gt.shape}")
    
    # Analyze match counts
    match_lengths = []
    num_singletons = 0
    total_matched_links = 0
    
    for row in df_gt.iter_rows(named=True):
        m_str = row.get("matched_entity_ids")
        if m_str is None or str(m_str).strip() == "":
            match_lengths.append(0)
            num_singletons += 1
        else:
            ids = [x.strip() for x in str(m_str).split(",") if x.strip()]
            match_lengths.append(len(ids))
            total_matched_links += len(ids)
            
    match_lengths = np.array(match_lengths)
    print(f"   Total S1 entities in GT: {len(match_lengths):,}")
    print(f"   Singletons (0 matches): {num_singletons:,} ({num_singletons / len(match_lengths) * 100:.2f}%)")
    print(f"   Entities with >= 1 match: {(len(match_lengths) - num_singletons):,} ({(len(match_lengths) - num_singletons) / len(match_lengths) * 100:.2f}%)")
    print(f"   Total S2/S3 linked records: {total_matched_links:,}")
    print(f"   Max matches for an entity: {np.max(match_lengths)}")
    print(f"   Mean matches (among non-singletons): {total_matched_links / (len(match_lengths) - num_singletons):.2f}")
    
    print("3. Loading Test Source 1...")
    df_test_s1 = pl.read_csv(test_s1_path, separator="\t", infer_schema_length=10000, null_values=[""])
    print(f"   Test S1 shape: {df_test_s1.shape}")
    print(f"   Country breakdown in Test S1:\n{df_test_s1['country'].value_counts()}")
    
    print(f"EDA Completed in {time.time() - t0:.2f}s")
    print("=" * 60)

def create_validation_split(
    data_dir: str,
    val_size: int = 50000,
    random_state: int = 42,
    output_dir: str = "dataset/validation"
):
    """
    Creates a stratified validation split of S1 entities and extracts corresponding S2/S3 candidate sets.
    """
    print(f"Creating validation split (val_size={val_size:,})...")
    os.makedirs(output_dir, exist_ok=True)
    
    train_s1_path = os.path.join(data_dir, "train", "train_source1.tsv")
    train_gt_path = os.path.join(data_dir, "train", "train_ground_truth.tsv")
    
    df_s1 = pl.read_csv(train_s1_path, separator="\t", null_values=[""])
    df_gt = pl.read_csv(train_gt_path, separator="\t", null_values=[""])
    
    # Merge to stratify by country and singleton status
    df_merged = df_s1.join(df_gt, left_on="entity_id", right_on="source1_entity_id", how="left")
    
    df_merged = df_merged.with_columns(
        is_singleton=pl.col("matched_entity_ids").is_null() | (pl.col("matched_entity_ids").str.strip_chars() == "")
    )
    
    # Stratified sample
    np.random.seed(random_state)
    sample_indices = []
    
    # Group by (country, is_singleton) and sample proportionally
    groups = df_merged.group_by(["country", "is_singleton"])
    total_rows = len(df_merged)
    
    sampled_dfs = []
    for (country, is_singleton), group_df in groups:
        n_group = len(group_df)
        group_sample_size = int(round(val_size * (n_group / total_rows)))
        sampled_group = group_df.sample(n=min(group_sample_size, n_group), seed=random_state)
        sampled_dfs.append(sampled_group)
        
    df_val = pl.concat(sampled_dfs)
    print(f"Validation split created: {len(df_val):,} entities.")
    print(f"Validation Country Breakdown:\n{df_val['country'].value_counts()}")
    print(f"Validation Singleton Breakdown:\n{df_val['is_singleton'].value_counts()}")
    
    # Save validation files
    val_s1 = df_val.select(["entity_id", "business_name", "business_address", "country"])
    val_gt = df_val.select([pl.col("entity_id").alias("source1_entity_id"), "matched_entity_ids"])
    
    val_s1.write_csv(os.path.join(output_dir, "val_source1.tsv"), separator="\t")
    val_gt.write_csv(os.path.join(output_dir, "val_ground_truth.tsv"), separator="\t")
    
    print(f"Saved validation set to {output_dir}/")
    return output_dir

if __name__ == "__main__":
    data_directory = "student_resource/dataset"
    run_eda(data_directory)
    create_validation_split(data_directory, val_size=50000, output_dir="dataset/validation")
