# Business Entity Resolution Pipeline
**Amazon ML Challenge 2026**

## 🏗️ Architecture & Reproduction Guide

This directory contains the self-contained, reproducible pipeline for Business Entity Resolution across three noisy sources (Source 1, Source 2, Source 3).

### Pipeline Stages
1. **Candidate Blocking (`src/fast_blocker.py`):** Multi-key inverted index & token alignment partitioned by country to prune $O(N \times M)$ search space to top-$K$ candidates ($K \le 12$).
2. **Feature Extraction (`src/feature_engine.py`):** C-accelerated `RapidFuzz` string metrics (Levenshtein, Jaro-Winkler, Token Sort/Set ratios) and numeric postal/address alignment.
3. **ML Matching Model (`src/model.py`):** LightGBM gradient boosted decision tree classifier trained on balanced positive and hard negative pairs.
4. **Calibration & Post-Processing (`src/calibrate.py`):** Dynamic threshold optimization maximizing Macro $F_{0.5}$ with high singleton confidence cutoff.
5. **Validation (`src/pipeline.py`):** Generates `output/matching_results.tsv` and `output/candidate_pairs.tsv` and validates against `utils/validate_submission.py`.

---

## 🚀 Execution Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run End-to-End Pipeline
```bash
PYTHONPATH=. python3 src/pipeline.py
```

### 3. Validate Submission Outputs
```bash
python3 ../../student_resource/utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../student_resource/dataset/test
```
