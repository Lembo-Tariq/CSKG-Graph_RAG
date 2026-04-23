"""
evaluate_baseline_rag_v2.py
===========================
Evaluates the Baseline RAG (abstracts) using:
- Hit Rate@K: did the correct paper appear in top K results?
- MRR: how high up was the correct paper?

One query per paper = clean binary evaluation (0 or 1 per query)
"""

import os
import json
import csv
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# STEP 1: Load evaluation dataset
# ─────────────────────────────────────────────

eval_file = os.path.join(current_dir, "eval_dataset_single.json")
with open(eval_file, "r", encoding="utf-8") as f:
    eval_dataset = json.load(f)

print("Loaded " + str(len(eval_dataset)) + " queries")

# ─────────────────────────────────────────────
# STEP 2: Load Baseline Chroma DB
# ─────────────────────────────────────────────

db_dir = os.path.join(current_dir, "db", "chroma_db_baseline")
print("Loading Baseline Chroma DB...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=db_dir, embedding_function=embeddings)
print("DB loaded")

# ─────────────────────────────────────────────
# STEP 3: Define K and retriever
# ─────────────────────────────────────────────

K = 20
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": K, "fetch_k": 50}
)

# ─────────────────────────────────────────────
# STEP 4: Metric functions
# ─────────────────────────────────────────────

def get_rank(retrieved_chunks, correct_paper_id):
    """
    Returns the rank (position) of the first chunk
    belonging to the correct paper.
    Returns None if not found.

    Example:
    chunks = [chunk from W111, chunk from W222, chunk from W333]
    correct = W222
    rank = 2 (found at position 2)
    """
    for i, chunk in enumerate(retrieved_chunks):
        if correct_paper_id in chunk:
            return i + 1  # rank starts at 1 not 0
    return None


def hit_rate(rank):
    """
    Hit Rate = 1 if correct paper found in top K, 0 if not.
    Binary — exactly what Dr. Danilo wants.
    """
    return 1 if rank is not None else 0


def reciprocal_rank(rank):
    """
    RR = 1 / rank of first correct result
    If correct paper at rank 1 -> RR = 1.0
    If correct paper at rank 2 -> RR = 0.5
    If correct paper at rank 5 -> RR = 0.2
    If not found -> RR = 0.0
    """
    if rank is None:
        return 0.0
    return 1.0 / rank

# ─────────────────────────────────────────────
# STEP 5: Run evaluation
# ─────────────────────────────────────────────

print("\nRunning Baseline RAG evaluation with K=" + str(K) + "...")

all_results = []
summary_rows = []
all_hits = []
all_rrs = []

for sample in eval_dataset:
    sample_id = sample["sample_id"]
    query = sample["query"]
    correct_paper_id = sample["paper_id"]
    title = sample["title"]

    print("\nSample " + str(sample_id) + ": " + query[:60] + "...")
    print("Correct paper: " + correct_paper_id + " (" + title[:40] + "...)")

    # Retrieve chunks
    retrieved_docs = retriever.invoke(query)
    retrieved_chunks = [doc.page_content for doc in retrieved_docs]

    # Calculate metrics
    rank = get_rank(retrieved_chunks, correct_paper_id)
    hr = hit_rate(rank)
    rr = reciprocal_rank(rank)

    all_hits.append(hr)
    all_rrs.append(rr)

    # Show result
    if hr == 1:
        print("HIT! Correct paper found at rank " + str(rank))
    else:
        print("MISS - correct paper not in top " + str(K))

    print("Hit=" + str(hr) + " | RR=" + str(round(rr, 3)))

    all_results.append({
        "sample_id": sample_id,
        "query": query,
        "correct_paper_id": correct_paper_id,
        "title": title,
        "rank": rank,
        "hit_rate": hr,
        "reciprocal_rank": rr
    })

    summary_rows.append({
        "sample_id": sample_id,
        "query": query[:50],
        "correct_paper_id": correct_paper_id,
        "rank": rank if rank else "not found",
        "hit_rate": hr,
        "reciprocal_rank": round(rr, 3)
    })

# ─────────────────────────────────────────────
# STEP 6: Overall metrics
# ─────────────────────────────────────────────

mean_hit_rate = sum(all_hits) / len(all_hits)
mrr = sum(all_rrs) / len(all_rrs)

print("\n" + "=" * 50)
print("BASELINE RAG RESULTS (K=" + str(K) + ")")
print("=" * 50)
print("Hit Rate@" + str(K) + ": " + str(round(mean_hit_rate, 3)))
print("MRR:        " + str(round(mrr, 3)))
print("Hits:       " + str(sum(all_hits)) + "/" + str(len(all_hits)) + " queries")
print("=" * 50)

# ─────────────────────────────────────────────
# STEP 7: Save results
# ─────────────────────────────────────────────

output = {
    "k": K,
    "evaluation_type": "baseline-rag-single-paper",
    "overall_metrics": {
        "hit_rate_at_k": round(mean_hit_rate, 3),
        "mrr": round(mrr, 3),
        "total_hits": sum(all_hits),
        "total_queries": len(all_hits)
    },
    "per_sample_results": all_results
}

json_file = os.path.join(current_dir, "baseline_eval_v2_results.json")
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

csv_file = os.path.join(current_dir, "baseline_eval_v2_results.csv")
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "sample_id", "query", "correct_paper_id",
        "rank", "hit_rate", "reciprocal_rank"
    ])
    writer.writeheader()
    writer.writerows(summary_rows)
    writer.writerow({
        "sample_id": "OVERALL",
        "query": "MEAN",
        "correct_paper_id": "-",
        "rank": "-",
        "hit_rate": round(mean_hit_rate, 3),
        "reciprocal_rank": round(mrr, 3)
    })

print("\nResults saved to baseline_eval_v2_results.json and baseline_eval_v2_results.csv")