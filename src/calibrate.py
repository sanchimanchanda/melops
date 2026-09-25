"""Macro F0.5 Optimizer, Dynamic Thresholding, and Singleton Filter."""

import numpy as np
from typing import Dict, List, Set, Tuple

def score_predictions_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]]
) -> float:
    """Computes competition Macro F0.5 across all S1 entities."""
    f05_scores = []
    
    for s1_id, gt_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        
        if len(gt_set) == 0:
            f05 = 1.0 if len(pred_set) == 0 else 0.0
        else:
            if len(pred_set) == 0:
                f05 = 0.0
            else:
                tp = len(gt_set.intersection(pred_set))
                fp = len(pred_set - gt_set)
                fn = len(gt_set - pred_set)
                
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                
                denom = 0.25 * p + r
                f05 = (1.25 * p * r) / denom if denom > 0 else 0.0
                
        f05_scores.append(f05)
    return float(np.mean(f05_scores))

def calibrate_threshold(
    ground_truth: Dict[str, Set[str]],
    scored_candidates: Dict[str, List[Tuple[str, float]]]
) -> Tuple[float, float, float]:
    """
    Finds the optimal probability threshold T maximizing Macro F0.5.
    Returns (best_threshold, best_f05, baseline_f05).
    """
    thresholds = np.linspace(0.15, 0.85, 41)
    best_t = 0.50
    best_f05 = -1.0
    
    for t in thresholds:
        preds = {}
        for s1_id, cand_scores in scored_candidates.items():
            matched = set(cid for cid, score in cand_scores if score >= t)
            preds[s1_id] = matched
            
        score = score_predictions_f05(ground_truth, preds)
        if score > best_f05:
            best_f05 = score
            best_t = t
            
    return float(best_t), float(best_f05)
