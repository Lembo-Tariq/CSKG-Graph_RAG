"""
evaluate_rag_v2.py
==================
Step 3 of the new evaluation pipeline.
Runs 20 synthetic queries through the RAG pipeline and
calculates Precision@K, Recall@K, F1, and MRR.

Key improvements from v1:
- Each query has 5-10 correct papers (not just 1)
- Uses paper ID matching (W numbers) instead of keyword matching
- Much more precise evaluation
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

eval_file = os.path.join(current_dir, "eval_dataset_v2.json")
with open(eval_file, "r", encoding="utf-8") as f:
    eval_dataset = json.load(f)

print(f"✅ Loaded {len(eval_dataset)} samples")

# ─────────────────────────────────────────────
# STEP 2: Load Chroma DB
# ─────────────────────────────────────────────

print("📂 Loading Chroma DB...")
db_dir = os.path.join(current_dir, "..", "Normal_RAG", "db", "chroma_db")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=db_dir, embedding_function=embeddings)
print("✅ DB loaded")

# ─────────────────────────────────────────────
# STEP 3: Define K and retriever
# ─────────────────────────────────────────────
# K=20 as Dr. Danilo suggested
# "if we retrieve 20 papers we expect a good portion
#  of the correct papers to appear"

K = 20
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": K, "fetch_k": 50}
)

# ─────────────────────────────────────────────
# STEP 4: Metric functions
# ─────────────────────────────────────────────

def is_relevant(fact: str, correct_paper_ids: list) -> bool:
    """
    A fact is relevant if it contains ANY of the correct paper W IDs.
    Each fact ends with (paper: W1010415138) so we check directly.

    Example:
    fact = "sentiment analysis uses the method cnn (paper: W102190306)"
    correct_paper_ids = ["W102190306", "W1010415138"]
    → "W102190306" found in fact → True
    """
    for paper_id in correct_paper_ids:
        if paper_id in fact:
            return True
    return False


def precision_at_k(retrieved_facts: list, correct_paper_ids: list, k: int) -> float:
    """
    Precision@K = relevant facts in top K / K
    How many of the retrieved facts are from correct papers?
    """
    retrieved_k = retrieved_facts[:k]
    relevant = sum(1 for fact in retrieved_k if is_relevant(fact, correct_paper_ids))
    return relevant / k if k > 0 else 0.0


def recall_at_k(retrieved_facts: list, correct_paper_ids: list, k: int) -> float:
    """
    Recall@K = distinct correct papers found in top K / total correct papers
    How many of the correct papers have at least one fact retrieved?
    """
    total = len(correct_paper_ids)
    if total == 0:
        return 0.0
    retrieved_k = retrieved_facts[:k]

    papers_found = set()
    for fact in retrieved_k:
        for paper_id in correct_paper_ids:
            if paper_id in fact:
                papers_found.add(paper_id)

    return len(papers_found) / total


def f1_score(precision: float, recall: float) -> float:
    """
    F1 = harmonic mean of Precision and Recall
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def reciprocal_rank(retrieved_facts: list, correct_paper_ids: list) -> float:
    """
    RR = 1 / position of first relevant fact
    How quickly does the first correct paper appear?
    """
    for i, fact in enumerate(retrieved_facts):
        if is_relevant(fact, correct_paper_ids):
            return 1.0 / (i + 1)
    return 0.0

# ─────────────────────────────────────────────
# STEP 5: Run evaluation
# ─────────────────────────────────────────────

print(f"\n🔍 Running evaluation with K={K}...")
print(f"   Total queries: {len(eval_dataset)}\n")

all_results = []
summary_rows = []
all_precisions = []
all_recalls = []
all_f1s = []
all_rrs = []

for sample in eval_dataset:
    sample_id = sample["sample_id"]
    query = sample["query"]
    correct_paper_ids = sample["paper_ids"]
    correct_papers = sample["papers"]

    print(f"📦 Sample {sample_id}: {query[:60]}...")
    print(f"   Correct paper IDs: {correct_paper_ids}")

    # Retrieve facts from RAG
    retrieved_docs = retriever.invoke(query)
    retrieved_facts = [doc.page_content for doc in retrieved_docs]

    # Calculate metrics
    p = precision_at_k(retrieved_facts, correct_paper_ids, K)
    r = recall_at_k(retrieved_facts, correct_paper_ids, K)
    f1 = f1_score(p, r)
    rr = reciprocal_rank(retrieved_facts, correct_paper_ids)

    all_precisions.append(p)
    all_recalls.append(r)
    all_f1s.append(f1)
    all_rrs.append(rr)

    # Show first 5 retrieved facts
    print(f"   Retrieved facts (first 5):")
    for i, fact in enumerate(retrieved_facts[:5]):
        flag = "✅" if is_relevant(fact, correct_paper_ids) else "❌"
        print(f"   {flag} {fact[:80]}...")

    print(f"   P@{K}={p:.3f} | R@{K}={r:.3f} | F1={f1:.3f} | RR={rr:.3f}\n")

    all_results.append({
        "sample_id": sample_id,
        "query": query,
        "correct_paper_ids": correct_paper_ids,
        "retrieved_facts": retrieved_facts,
        "precision_at_k": p,
        "recall_at_k": r,
        "f1": f1,
        "reciprocal_rank": rr
    })

    summary_rows.append({
        "sample_id": sample_id,
        "query": query[:50],
        "num_correct_papers": len(correct_paper_ids),
        "precision_at_k": round(p, 3),
        "recall_at_k": round(r, 3),
        "f1": round(f1, 3),
        "reciprocal_rank": round(rr, 3)
    })

# ─────────────────────────────────────────────
# STEP 6: Overall metrics
# ─────────────────────────────────────────────

mean_precision = sum(all_precisions) / len(all_precisions)
mean_recall = sum(all_recalls) / len(all_recalls)
mean_f1 = sum(all_f1s) / len(all_f1s)
mrr = sum(all_rrs) / len(all_rrs)

print("=" * 50)
print(f"📊 OVERALL RESULTS (K={K}, Paper ID Matching)")
print("=" * 50)
print(f"   Mean Precision@{K}: {mean_precision:.3f}")
print(f"   Mean Recall@{K}:    {mean_recall:.3f}")
print(f"   Mean F1@{K}:        {mean_f1:.3f}")
print(f"   MRR:                {mrr:.3f}")
print("=" * 50)

# ─────────────────────────────────────────────
# STEP 7: Save results
# ─────────────────────────────────────────────

output = {
    "k": K,
    "evaluation_type": "paper-id-matching",
    "overall_metrics": {
        "mean_precision_at_k": round(mean_precision, 3),
        "mean_recall_at_k": round(mean_recall, 3),
        "mean_f1_at_k": round(mean_f1, 3),
        "mrr": round(mrr, 3)
    },
    "per_sample_results": all_results
}

json_file = os.path.join(current_dir, "rag_eval_results_v3.json")
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

csv_file = os.path.join(current_dir, "rag_eval_results_v3.csv")
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "sample_id", "query", "num_correct_papers",
        "precision_at_k", "recall_at_k", "f1", "reciprocal_rank"
    ])
    writer.writeheader()
    writer.writerows(summary_rows)
    writer.writerow({
        "sample_id": "OVERALL",
        "query": "MEAN",
        "num_correct_papers": "-",
        "precision_at_k": round(mean_precision, 3),
        "recall_at_k": round(mean_recall, 3),
        "f1": round(mean_f1, 3),
        "reciprocal_rank": round(mrr, 3)
    })

print(f"\n✨ Results saved to rag_eval_results_v3.json and rag_eval_results_v3.csv")