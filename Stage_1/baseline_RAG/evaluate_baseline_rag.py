import os
import json
import csv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))

# Load evaluation dataset (same 20 queries used for triple-based RAG)
eval_file = os.path.join(current_dir, "..", "OpenAlex_Enrichment", "eval_dataset_v2.json")
with open(eval_file, "r", encoding="utf-8") as f:
    eval_dataset = json.load(f)

print("Loaded " + str(len(eval_dataset)) + " samples")

# Load the BASELINE Chroma DB (abstracts based)
db_dir = os.path.join(current_dir, "db", "chroma_db_baseline")
print("Loading Baseline Chroma DB...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=db_dir, embedding_function=embeddings)
print("DB loaded")

K = 20
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": K, "fetch_k": 50}
)

# Metric functions

def is_relevant(chunk, correct_paper_ids):
    for paper_id in correct_paper_ids:
        if paper_id in chunk:
            return True
    return False


def precision_at_k(retrieved_chunks, correct_paper_ids, k):
    retrieved_k = retrieved_chunks[:k]
    relevant = sum(1 for chunk in retrieved_k if is_relevant(chunk, correct_paper_ids))
    return relevant / k if k > 0 else 0.0


def recall_at_k(retrieved_chunks, correct_paper_ids, k):
    total = len(correct_paper_ids)
    if total == 0:
        return 0.0
    retrieved_k = retrieved_chunks[:k]
    papers_found = set()
    for chunk in retrieved_k:
        for paper_id in correct_paper_ids:
            if paper_id in chunk:
                papers_found.add(paper_id)
    return len(papers_found) / total


def f1_score(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def reciprocal_rank(retrieved_chunks, correct_paper_ids):
    for i, chunk in enumerate(retrieved_chunks):
        if is_relevant(chunk, correct_paper_ids):
            return 1.0 / (i + 1)
    return 0.0


print("Running baseline evaluation with K=" + str(K) + "...")

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

    print("Sample " + str(sample_id) + ": " + query[:60] + "...")
    print("Correct papers: " + str(len(correct_paper_ids)))

    retrieved_docs = retriever.invoke(query)
    retrieved_chunks = [doc.page_content for doc in retrieved_docs]

    p = precision_at_k(retrieved_chunks, correct_paper_ids, K)
    r = recall_at_k(retrieved_chunks, correct_paper_ids, K)
    f1 = f1_score(p, r)
    rr = reciprocal_rank(retrieved_chunks, correct_paper_ids)

    all_precisions.append(p)
    all_recalls.append(r)
    all_f1s.append(f1)
    all_rrs.append(rr)

    print("First 3 retrieved chunks:")
    for i, chunk in enumerate(retrieved_chunks[:3]):
        flag = "HIT" if is_relevant(chunk, correct_paper_ids) else "MISS"
        print("  " + flag + ": " + chunk[:80] + "...")

    print("P@" + str(K) + "=" + str(round(p, 3)) +
          " | R@" + str(K) + "=" + str(round(r, 3)) +
          " | F1=" + str(round(f1, 3)) +
          " | RR=" + str(round(rr, 3)))
    print("")

    all_results.append({
        "sample_id": sample_id,
        "query": query,
        "correct_paper_ids": correct_paper_ids,
        "retrieved_chunks": retrieved_chunks,
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

mean_precision = sum(all_precisions) / len(all_precisions)
mean_recall = sum(all_recalls) / len(all_recalls)
mean_f1 = sum(all_f1s) / len(all_f1s)
mrr = sum(all_rrs) / len(all_rrs)

print("=" * 50)
print("OVERALL RESULTS - BASELINE RAG (K=" + str(K) + ")")
print("=" * 50)
print("Mean Precision@" + str(K) + ": " + str(round(mean_precision, 3)))
print("Mean Recall@" + str(K) + ":    " + str(round(mean_recall, 3)))
print("Mean F1@" + str(K) + ":        " + str(round(mean_f1, 3)))
print("MRR:                " + str(round(mrr, 3)))
print("=" * 50)

output = {
    "k": K,
    "evaluation_type": "baseline-rag-abstracts",
    "overall_metrics": {
        "mean_precision_at_k": round(mean_precision, 3),
        "mean_recall_at_k": round(mean_recall, 3),
        "mean_f1_at_k": round(mean_f1, 3),
        "mrr": round(mrr, 3)
    },
    "per_sample_results": all_results
}

json_file = os.path.join(current_dir, "baseline_eval_results.json")
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

csv_file = os.path.join(current_dir, "baseline_eval_results.csv")
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

print("Results saved to baseline_eval_results.json and baseline_eval_results.csv")
