"""
step7_evaluate.py
=================
Evaluates both Triple RAG and Baseline RAG.
Metrics: Hit Rate@K and MRR at K=1,3,5,10,20
"""

import os
import json
import csv
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

NUM_SAMPLES = 2000  # must match step2

current_dir = os.path.dirname(os.path.abspath(__file__))

eval_file       = os.path.join(current_dir, "Step4Out-eval_dataset_" + str(NUM_SAMPLES) + ".json")
triple_db_dir   = os.path.join(current_dir, "db", "triple_db_" + str(NUM_SAMPLES))
baseline_db_dir = os.path.join(current_dir, "db", "baseline_db_" + str(NUM_SAMPLES))
json_out        = os.path.join(current_dir, "Step7Out-eval_" + str(NUM_SAMPLES) + "_results.json")
csv_out         = os.path.join(current_dir, "Step7Out-eval_" + str(NUM_SAMPLES) + "_results.csv")

with open(eval_file, "r", encoding="utf-8") as f:
    eval_dataset = json.load(f)

print("Loaded " + str(len(eval_dataset)) + " queries")

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Loading Triple RAG DB...")
triple_db = Chroma(persist_directory=triple_db_dir, embedding_function=embeddings)

print("Loading Baseline RAG DB...")
baseline_db = Chroma(persist_directory=baseline_db_dir, embedding_function=embeddings)

print("Both DBs loaded!")

K_VALUES = [1, 3, 5, 10, 20]
MAX_K = max(K_VALUES)

triple_retriever = triple_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": MAX_K, "fetch_k": 50}
)
baseline_retriever = baseline_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": MAX_K, "fetch_k": 50}
)

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

def evaluate_system(retriever, system_name, eval_dataset, k_values):
    print("\nEvaluating " + system_name + "...")
    results_per_k = {k: {"hits": [], "rrs": []} for k in k_values}
    per_query = []

    for i, sample in enumerate(eval_dataset):
        query = sample["query"]
        correct_paper_id = sample["paper_id"]

        retrieved_docs = retriever.invoke(query)
        retrieved_chunks = [doc.page_content for doc in retrieved_docs]
        full_rank = get_rank(retrieved_chunks, correct_paper_id)

        query_result = {
            "sample_id": sample["sample_id"],
            "paper_id": correct_paper_id,
            "title": sample["title"],
            "query": query,
            "full_rank": full_rank
        }

        for k in k_values:
            rank_at_k = full_rank if (full_rank is not None and full_rank <= k) else None
            hr = hit_rate(rank_at_k)
            rr = reciprocal_rank(rank_at_k)
            results_per_k[k]["hits"].append(hr)
            results_per_k[k]["rrs"].append(rr)
            query_result["hit@" + str(k)] = hr
            query_result["rr@" + str(k)] = round(rr, 3)

        per_query.append(query_result)

        if (i + 1) % 100 == 0:
            print("  Progress: " + str(i+1) + "/" + str(len(eval_dataset)) + " queries...")

    summary = {}
    for k in k_values:
        hits = results_per_k[k]["hits"]
        rrs  = results_per_k[k]["rrs"]
        summary[k] = {
            "hit_rate": round(sum(hits) / len(hits), 3),
            "mrr": round(sum(rrs) / len(rrs), 3),
            "total_hits": sum(hits),
            "total_queries": len(hits)
        }

    return summary, per_query

triple_summary, triple_per_query = evaluate_system(
    triple_retriever, "Triple RAG", eval_dataset, K_VALUES
)
baseline_summary, baseline_per_query = evaluate_system(
    baseline_retriever, "Baseline RAG", eval_dataset, K_VALUES
)

total_q = len(eval_dataset)
print("\n" + "=" * 70)
print("EVALUATION RESULTS (" + str(total_q) + " papers, NUM_SAMPLES=" + str(NUM_SAMPLES) + ")")
print("=" * 70)
print("{:<6} {:<25} {:<25}".format("K", "Baseline RAG (Abstracts)", "Triple RAG (CSKG)"))
print("-" * 70)
print("{:<6} {:<12} {:<12} {:<12} {:<12}".format("", "Hit Rate", "MRR", "Hit Rate", "MRR"))
print("-" * 70)

for k in K_VALUES:
    b = baseline_summary[k]
    t = triple_summary[k]
    print("{:<6} {:<12} {:<12} {:<12} {:<12}".format(
        k,
        str(b["hit_rate"]) + " (" + str(b["total_hits"]) + "/" + str(total_q) + ")",
        str(b["mrr"]),
        str(t["hit_rate"]) + " (" + str(t["total_hits"]) + "/" + str(total_q) + ")",
        str(t["mrr"])
    ))

print("=" * 70)

output = {
    "num_samples": NUM_SAMPLES,
    "total_queries": total_q,
    "k_values": K_VALUES,
    "baseline_rag": {"results_per_k": baseline_summary, "per_query_results": baseline_per_query},
    "triple_rag":   {"results_per_k": triple_summary,   "per_query_results": triple_per_query}
}

with open(json_out, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

with open(csv_out, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "K", "Baseline_HitRate", "Baseline_MRR",
        "Triple_HitRate", "Triple_MRR"
    ])
    writer.writeheader()
    for k in K_VALUES:
        writer.writerow({
            "K": k,
            "Baseline_HitRate": baseline_summary[k]["hit_rate"],
            "Baseline_MRR":     baseline_summary[k]["mrr"],
            "Triple_HitRate":   triple_summary[k]["hit_rate"],
            "Triple_MRR":       triple_summary[k]["mrr"]
        })

print("\nResults saved to " + json_out)
print("Results saved to " + csv_out)