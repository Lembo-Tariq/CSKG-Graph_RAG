"""
evaluate_baseline_full.py
=========================
Evaluates Baseline RAG on all 734 papers at multiple K values.
K = 1, 3, 5, 10, 20
Metrics: Hit Rate@K and MRR
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
# Load evaluation dataset
# ─────────────────────────────────────────────

eval_file = os.path.join(current_dir, "eval_dataset_full.json")
with open(eval_file, "r", encoding="utf-8") as f:
    eval_dataset = json.load(f)

print("Loaded " + str(len(eval_dataset)) + " queries")

# ─────────────────────────────────────────────
# Load Baseline Chroma DB
# ─────────────────────────────────────────────

db_dir = os.path.join(current_dir, "db", "chroma_db_baseline")
print("Loading Baseline Chroma DB...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=db_dir, embedding_function=embeddings)
print("DB loaded with 887 papers")

# ─────────────────────────────────────────────
# Metric functions
# ─────────────────────────────────────────────

def get_rank(retrieved_chunks, correct_paper_id):
    for i, chunk in enumerate(retrieved_chunks):
        if correct_paper_id in chunk:
            return i + 1
    return None

def hit_rate(rank):
    return 1 if rank is not None else 0

def reciprocal_rank(rank):
    if rank is None:
        return 0.0
    return 1.0 / rank

# ─────────────────────────────────────────────
# Run evaluation at multiple K values
# ─────────────────────────────────────────────

K_VALUES = [1, 3, 5, 10, 20]
MAX_K = max(K_VALUES)

# We retrieve MAX_K chunks once per query
# then calculate metrics for each K value
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": MAX_K, "fetch_k": 50}
)

print("Running evaluation for K values: " + str(K_VALUES))
print("Total queries: " + str(len(eval_dataset)))
print("This may take a few minutes...\n")

# Store results per K
results_per_k = {k: {"hits": [], "rrs": []} for k in K_VALUES}
per_query_results = []

for i, sample in enumerate(eval_dataset):
    query = sample["query"]
    correct_paper_id = sample["paper_id"]

    # Retrieve MAX_K chunks once
    retrieved_docs = retriever.invoke(query)
    retrieved_chunks = [doc.page_content for doc in retrieved_docs]

    # Get rank once (position of correct paper in full list)
    full_rank = get_rank(retrieved_chunks, correct_paper_id)

    # Calculate metrics for each K value
    query_result = {
        "sample_id": sample["sample_id"],
        "paper_id": correct_paper_id,
        "title": sample["title"],
        "query": query,
        "full_rank": full_rank
    }

    for k in K_VALUES:
        # For this K, the paper is a hit only if rank <= k
        rank_at_k = full_rank if (full_rank is not None and full_rank <= k) else None
        hr = hit_rate(rank_at_k)
        rr = reciprocal_rank(rank_at_k)

        results_per_k[k]["hits"].append(hr)
        results_per_k[k]["rrs"].append(rr)

        query_result["hit@" + str(k)] = hr
        query_result["rr@" + str(k)] = round(rr, 3)

    per_query_results.append(query_result)

    # Progress update every 100 queries
    if (i + 1) % 100 == 0:
        print("Progress: " + str(i+1) + "/734 queries evaluated...")

# ─────────────────────────────────────────────
# Calculate overall metrics per K
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("BASELINE RAG RESULTS - FULL EVALUATION (734 papers)")
print("=" * 60)
print("{:<6} {:<15} {:<10}".format("K", "Hit Rate@K", "MRR@K"))
print("-" * 35)

summary = {}
for k in K_VALUES:
    hits = results_per_k[k]["hits"]
    rrs = results_per_k[k]["rrs"]
    mean_hr = sum(hits) / len(hits)
    mean_mrr = sum(rrs) / len(rrs)
    total_hits = sum(hits)
    summary[k] = {
        "hit_rate": round(mean_hr, 3),
        "mrr": round(mean_mrr, 3),
        "total_hits": total_hits
    }
    print("{:<6} {:<15} {:<10}".format(
        k,
        str(round(mean_hr, 3)) + " (" + str(total_hits) + "/" + str(len(hits)) + ")",
        str(round(mean_mrr, 3))
    ))

print("=" * 60)

# ─────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────

output = {
    "evaluation_type": "baseline-rag-full-734-papers",
    "total_queries": len(eval_dataset),
    "db_size": 887,
    "k_values": K_VALUES,
    "results_per_k": summary,
    "per_query_results": per_query_results
}

json_file = os.path.join(current_dir, "baseline_eval_full_results.json")
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

# CSV summary
csv_file = os.path.join(current_dir, "baseline_eval_full_results.csv")
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["K", "Hit_Rate", "MRR", "Total_Hits", "Total_Queries"])
    writer.writeheader()
    for k in K_VALUES:
        writer.writerow({
            "K": k,
            "Hit_Rate": summary[k]["hit_rate"],
            "MRR": summary[k]["mrr"],
            "Total_Hits": summary[k]["total_hits"],
            "Total_Queries": len(eval_dataset)
        })

print("\nResults saved to baseline_eval_full_results.json and .csv")