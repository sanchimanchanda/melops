# Project Requirements Specification: Business Entity Resolution

## 📋 1. Functional Requirements

### 1.1 Data Ingestion & Schema Compliance
- **File Parsing:** All datasets (`train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, `train_ground_truth.tsv`, `test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`) MUST be read with explicit tab delimiter `sep="\t"`.
- **Field Schema:**
  - `entity_id`: String (`S1-xxxxx`, `S2-xxxxx`, `S3-xxxxx`).
  - `business_name`: String (noisy, legal suffixes, abbreviations, typos).
  - `business_address`: String (landmarks, street names, municipal numbers, missing pin codes).
  - `country`: String. Must support open-set labels (US, India, and test set France) without hardcoded filtering.
- **Reference Standard:** Source 1 is deduplicated reference. Each S1 entity matches 0, 1, or multiple records from S2 and S3.

### 1.2 Text Normalization & Preprocessing
- Language- and locale-agnostic lowercasing, punctuation handling, whitespace normalization.
- Domain-aware expansion of business abbreviations (*Corp/Corporation*, *Pvt/Private*, *Ltd/Limited*, *St/Street*, *Rd/Road*).
- Preservation of numeric identifiers (house/street numbers, pin codes, unit numbers).

### 1.3 High-Throughput Candidate Generation (Blocking)
- Search space reduction from $O(|S1| \times (|S2| + |S3|))$ to top-$K$ plausible candidates per S1 entity.
- Hybrid candidate pooling combining:
  1. Country partition matching (with fallback).
  2. Character n-gram MinHash LSH / Sparse TF-IDF nearest neighbors.
  3. Inverted index BM25 lexical retrieval.
  4. Bi-encoder dense semantic similarity search (using FAISS/HNSW).
- Must record candidate list for every S1 entity in `candidate_pairs.tsv`.

### 1.4 Pairwise Feature Engineering
- **Fuzzy String Metrics:** RapidFuzz Levenshtein, Damerau-Levenshtein, Jaro-Winkler, Token Sort Ratio, Token Set Ratio, Partial Ratio.
- **Lexical Overlaps:** Jaccard similarity of word/character tokens, Overlap coefficient, Dice coefficient.
- **Statistical TF-IDF Similarities:** Word (1-2 gram) and Character (2-5 gram) TF-IDF cosine distances.
- **Semantic Embeddings:** Cosine similarity from open-source SentenceTransformers (MIT/Apache 2.0).
- **Address Tokens:** Numeric sequence equality, street/locality token overlap.

### 1.5 Classification & Matching Model
- GBDT pairwise classifier (LightGBM / XGBoost / CatBoost) and/or Cross-Encoder Transformer.
- Models must strictly have MIT/Apache 2.0 licenses and parameter count $\le 8\text{B}$.
- Training pair construction: Ground truth positives + hard negatives sampled from blocking candidates + random negatives.

### 1.6 Threshold Optimization & Singleton Handling
- Optimize decision threshold to directly maximize **Macro $F_{0.5}$**:
  $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
- Precision is weighted 2× over recall.
- Singleton confidence threshold: Entities with low maximum candidate probability must predict empty list `""` to secure 1.0 credit per singleton.

### 1.7 Submission & Verification
- Generate `matching_results.tsv` and `candidate_pairs.tsv` in `output/`.
- Ensure exact TSV formatting (no quoting, correct column headers).
- Automated validation via `student_resource/utils/validate_submission.py`.
- Final zip archive packaging matching competition specifications.

---

## 💻 2. Non-Functional & System Requirements

- **Zero External Lookup:** No external APIs, geocoding services, web scraping, or database queries.
- **Performance & Scalability:** Vectorized operations (`RapidFuzz`, `Polars`, `Numba`) to process candidate pairs within minutes without out-of-memory (OOM) errors.
- **Reproducibility:** Pinned seeds (`random_state=42`), modular codebase under `src/`, deterministic pipeline execution.
- **Platform Compatibility:** macOS / Linux, Python 3.10+.

---

## 📦 3. Deliverables & Submission Layout

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv        # Scored leaderboard matches
│   └── candidate_pairs.tsv         # Blocking candidate set
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # Complete modular source code
│       ├── README.md               # End-to-end execution guide
│       └── requirements.txt        # Pinned dependencies
└── Documentation_template.md       # Technical methodology writeup
```
