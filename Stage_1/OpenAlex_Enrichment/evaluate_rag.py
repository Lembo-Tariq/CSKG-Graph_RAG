"""
evaluate_rag.py
===============
Step 3 of the evaluation pipeline — IMPROVED VERSION.
Evaluates RAG at concept level instead of paper/DOI level.
This is fairer to Normal RAG since CSKG facts are generic.

Changes from v1:
- Concept-level retrieval instead of DOI matching
- K increased to 5
- Extracts key concepts from paper titles as ground truth
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

eval_file = os.path.join(current_dir, "eval_dataset.json")
with open(eval_file, "r", encoding="utf-8") as f:
    eval_dataset = json.load(f)

print(f"✅ Loaded {len(eval_dataset)} papers with queries")

# ─────────────────────────────────────────────
# STEP 2: Load Chroma DB
# ─────────────────────────────────────────────

print("📂 Loading Chroma DB...")
db_dir = os.path.join(current_dir, "..", "Normal_RAG", "db", "chroma_db")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=db_dir, embedding_function=embeddings)
print("✅ DB loaded")

# ─────────────────────────────────────────────
# STEP 3: Define concepts per paper
# ─────────────────────────────────────────────
# Instead of matching by DOI, we define key concepts
# that should appear in retrieved facts for each paper.
# A retrieved fact is "relevant" if it contains ANY
# of the paper's key concepts.
#
# This is fairer because CSKG facts are generic —
# many papers share the same relationships but we
# can at least check if the RIGHT CONCEPTS were retrieved.

PAPER_CONCEPTS = {
    "Transfer learning for image classification using VGG19: Caltech-101 image data set": [
        "transfer learning", "vgg", "caltech", "image classification"
    ],
    "Robust Image Sentiment Analysis Using Progressively Trained and Domain Transferred Deep Networks": [
        "sentiment", "domain transfer", "progressive", "image sentiment"
    ],
    "Small-Sample Image Classification Method of Combining Prototype and Margin Learning": [
        "small sample", "prototype", "margin", "few shot"
    ],
    "Genetic Programming With a New Representation to Automatically Learn Features and Evolve Ensembles for Image Classification": [
        "genetic programming", "ensemble", "evolutionary", "feature learning"
    ],
    "Image Classification Using Transfer Learning and Deep Learning": [
        "transfer learning", "deep learning", "image classification", "vgg16"
    ],
    "Art Classification with Pytorch Using Transfer Learning": [
        "art", "pytorch", "transfer learning", "artwork"
    ],
    "Effects of Image Degradation and Degradation Removal to CNN-Based Image Classification": [
        "degradation", "noise", "cnn", "image quality"
    ],
    "Evolutionary Deep Learning: A Genetic Programming Approach to Image Classification": [
        "evolutionary", "genetic programming", "deep learning", "convolutional"
    ]
}

# ─────────────────────────────────────────────
# STEP 4: Define K and retriever
# ─────────────────────────────────────────────

K = 5
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": K, "fetch_k": 20}
)

# ─────────────────────────────────────────────
# STEP 5: Define metric functions
# ─────────────────────────────────────────────

def is_relevant(fact: str, concepts: list) -> bool:
    """
    A fact is relevant if it contains ANY of the paper's key concepts.
    We check case-insensitively.
    
    Example:
    fact = "image classification uses the method transfer learning..."
    concepts = ["transfer learning", "vgg", "caltech"]
    → "transfer learning" found in fact → relevant = True
    """
    fact_lower = fact.lower()
    return any(concept.lower() in fact_lower for concept in concepts)


def precision_at_k(retrieved_facts: list, concepts: list, k: int) -> float:
    """
    Precision@K = relevant facts in top K / K
    """
    retrieved_k = retrieved_facts[:k]
    relevant = sum(1 for fact in retrieved_k if is_relevant(fact, concepts))
    return relevant / k


def recall_at_k(retrieved_facts: list, concepts: list, k: int, total_relevant: int) -> float:
    """
    Recall@K = relevant facts in top K / total relevant facts in DB
    """
    if total_relevant == 0:
        return 0.0
    retrieved_k = retrieved_facts[:k]
    relevant = sum(1 for fact in retrieved_k if is_relevant(fact, concepts))
    return relevant / total_relevant


def f1_score(precision: float, recall: float) -> float:
    """
    F1 = harmonic mean of Precision and Recall
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def reciprocal_rank(retrieved_facts: list, concepts: list) -> float:
    """
    RR = 1 / position of first relevant fact
    """
    for i, fact in enumerate(retrieved_facts):
        if is_relevant(fact, concepts):
            return 1.0 / (i + 1)
    return 0.0

# ─────────────────────────────────────────────
# STEP 6: Count total relevant facts in DB
# ─────────────────────────────────────────────
# We need to know how many facts in the ENTIRE DB
# are relevant to each paper (for recall calculation)
# We do this by fetching all facts and checking concepts

print("\n📊 Counting relevant facts in DB for each paper...")

# Get all documents from Chroma
all_docs = db.get()
all_facts = all_docs["documents"]
print(f"   Total facts in DB: {len(all_facts)}")

# Count relevant facts per paper
total_relevant_per_paper = {}
for paper in eval_dataset:
    title = paper["title"]
    concepts = PAPER_CONCEPTS.get(title, [])
    count = sum(1 for fact in all_facts if is_relevant(fact, concepts))
    total_relevant_per_paper[title] = count
    print(f"   {title[:45]}... → {count} relevant facts")

# ─────────────────────────────────────────────
# STEP 7: Run evaluation
# ─────────────────────────────────────────────

print(f"\n🔍 Running concept-level evaluation with K={K}...")
print(f"   Total queries: {sum(len(p['generated_queries']) for p in eval_dataset)}\n")

all_results = []
summary_rows = []

all_precisions = []
all_recalls = []
all_f1s = []
all_rrs = []

for paper in eval_dataset:
    title = paper["title"]
    concepts = PAPER_CONCEPTS.get(title, [])
    total_relevant = total_relevant_per_paper[title]

    print(f"📄 Paper: {title[:50]}...")
    print(f"   Concepts: {concepts}")
    print(f"   Total relevant facts in DB: {total_relevant}")

    paper_results = []

    for query in paper["generated_queries"]:
        retrieved_docs = retriever.invoke(query)
        retrieved_facts = [doc.page_content for doc in retrieved_docs]

        p = precision_at_k(retrieved_facts, concepts, K)
        r = recall_at_k(retrieved_facts, concepts, K, total_relevant)
        f1 = f1_score(p, r)
        rr = reciprocal_rank(retrieved_facts, concepts)

        all_precisions.append(p)
        all_recalls.append(r)
        all_f1s.append(f1)
        all_rrs.append(rr)

        print(f"\n   Query: {query}")
        for i, fact in enumerate(retrieved_facts):
            relevant_flag = "✅" if is_relevant(fact, concepts) else "❌"
            print(f"   {relevant_flag} {fact[:80]}...")
        print(f"   P@{K}={p:.3f} | R@{K}={r:.3f} | F1={f1:.3f} | RR={rr:.3f}")

        paper_results.append({
            "query": query,
            "concepts": concepts,
            "retrieved_facts": retrieved_facts,
            "precision_at_k": p,
            "recall_at_k": r,
            "f1": f1,
            "reciprocal_rank": rr
        })

        summary_rows.append({
            "paper_title": title[:50],
            "query": query,
            "precision_at_k": round(p, 3),
            "recall_at_k": round(r, 3),
            "f1": round(f1, 3),
            "reciprocal_rank": round(rr, 3)
        })

    all_results.append({
        "paper": title,
        "concepts": concepts,
        "total_relevant_in_db": total_relevant,
        "queries": paper_results
    })
    print()

# ─────────────────────────────────────────────
# STEP 8: Overall metrics
# ─────────────────────────────────────────────

mean_precision = sum(all_precisions) / len(all_precisions)
mean_recall = sum(all_recalls) / len(all_recalls)
mean_f1 = sum(all_f1s) / len(all_f1s)
mrr = sum(all_rrs) / len(all_rrs)

print("=" * 50)
print(f"📊 OVERALL RESULTS (K={K}, Concept-Level)")
print("=" * 50)
print(f"   Mean Precision@{K}: {mean_precision:.3f}")
print(f"   Mean Recall@{K}:    {mean_recall:.3f}")
print(f"   Mean F1@{K}:        {mean_f1:.3f}")
print(f"   MRR:               {mrr:.3f}")
print("=" * 50)

# ─────────────────────────────────────────────
# STEP 9: Save results
# ─────────────────────────────────────────────

output = {
    "k": K,
    "evaluation_type": "concept-level",
    "overall_metrics": {
        "mean_precision_at_k": round(mean_precision, 3),
        "mean_recall_at_k": round(mean_recall, 3),
        "mean_f1_at_k": round(mean_f1, 3),
        "mrr": round(mrr, 3)
    },
    "per_paper_results": all_results
}

json_file = os.path.join(current_dir, "rag_eval_results_v2.json")
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

csv_file = os.path.join(current_dir, "rag_eval_results_v2.csv")
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["paper_title", "query", "precision_at_k", "recall_at_k", "f1", "reciprocal_rank"])
    writer.writeheader()
    writer.writerows(summary_rows)
    writer.writerow({
        "paper_title": "OVERALL",
        "query": "MEAN",
        "precision_at_k": round(mean_precision, 3),
        "recall_at_k": round(mean_recall, 3),
        "f1": round(mean_f1, 3),
        "reciprocal_rank": round(mrr, 3)
    })

print(f"\n✨ Results saved to rag_eval_results_v2.json and rag_eval_results_v2.csv")