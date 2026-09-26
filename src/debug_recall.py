import polars as pl
from src.fast_blocker import FastOptimizedBlocker

def run():
    print("Loading data...")
    val_s1 = pl.read_csv("dataset/validation/val_source1.tsv", separator="\t").head(5000)
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
            
    df_s2 = pl.read_csv("student_resource/dataset/train/train_source2.tsv", separator="\t")
    df_s3 = pl.read_csv("student_resource/dataset/train/train_source3.tsv", separator="\t")
    df_corpus = pl.concat([df_s2, df_s3])
    
    print("Fitting blocker...")
    blocker = FastOptimizedBlocker(max_candidates=60)
    blocker.fit_corpus(df_corpus)
    
    print("Querying...")
    candidate_map = blocker.query(val_s1)
    
    # Find a missed match
    missed = []
    for s1_id, true_set in gt_map.items():
        cands = set(candidate_map.get(s1_id, []))
        for t_id in true_set:
            if t_id not in cands:
                missed.append((s1_id, t_id))
                if len(missed) >= 10:
                    break
        if len(missed) >= 10:
            break
            
    print("\n--- MISSED MATCHES ---")
    for s1_id, t_id in missed:
        print(f"\nS1: {s1_id}")
        s1_row = val_s1.filter(pl.col("entity_id") == s1_id).row(0)
        print(s1_row)
        
        print(f"True Match (Missed): {t_id}")
        t_row = df_corpus.filter(pl.col("entity_id") == t_id)
        if len(t_row) > 0:
            print(t_row.row(0))
        else:
            print("NOT FOUND IN CORPUS?!")

if __name__ == "__main__":
    run()
