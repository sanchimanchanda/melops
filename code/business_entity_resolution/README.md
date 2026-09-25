# Entity Resolution Pipeline Reproduction

## Setup
1. Ensure your data directory matches the challenge structure, specifically placing the train and test TSV files inside `student_resource/dataset/`.
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Execution
Run the end-to-end pipeline from the root directory:
```bash
python3 src/pipeline.py
```
This script will:
- Load the datasets and sample 400,000 S1 queries.
- Build training sets for each country with max_negatives=6.
- Train the LightGBM classifier.
- Run validation and calibrate the probability threshold.
- Run inference on the Test set and save outputs to `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
