"""
create_results_table.py
=======================
Compiles all experiment results into a master comparison table.
Saves as both CSV and JSON for the research paper.
"""

import os
import json
import csv

current_dir = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# All experiment results hardcoded
# ─────────────────────────────────────────────

experiments = [
    {
        "id": 1,
        "name": "Triple RAG (DOI matching)",
        "db_content": "CSKG triples with DOIs",
        "db_size": "10,000 facts",
        "query_type": "24 synthetic (5-10 papers combined)",
        "num_queries": 24,
        "k": 20,
        "hit_rate": 0.030,
        "mrr": 0.090,
        "notes": "First evaluation attempt - DOI level matching"
    },
    {
        "id": 2,
        "name": "Triple RAG (Paper ID matching)",
        "db_content": "CSKG triples with W IDs",
        "db_size": "10,000 facts",
        "query_type": "24 synthetic (5-10 papers combined)",
        "num_queries": 24,
        "k": 20,
        "hit_rate": 0.030,
        "mrr": 0.090,
        "notes": "Same queries, switched to W ID matching"
    },
    {
        "id": 3,
        "name": "Baseline RAG (abstracts, combined queries)",
        "db_content": "887 paper abstracts + titles",
        "db_size": "887 papers",
        "query_type": "24 synthetic (5-10 papers combined)",
        "num_queries": 24,
        "k": 20,
        "hit_rate": 0.030,
        "mrr": 0.138,
        "notes": "Old query approach - too generic"
    },
    {
        "id": 4,
        "name": "Baseline RAG (abstracts, single paper queries)",
        "db_content": "887 paper abstracts + titles",
        "db_size": "887 papers",
        "query_type": "20 single paper specific queries",
        "num_queries": 20,
        "k": 20,
        "hit_rate": 0.950,
        "mrr": 0.900,
        "notes": "Switched to single paper queries - major improvement"
    },
    {
        "id": 5,
        "name": "Baseline RAG Full (K=1)",
        "db_content": "887 paper abstracts + titles",
        "db_size": "887 papers",
        "query_type": "734 single paper specific queries",
        "num_queries": 734,
        "k": 1,
        "hit_rate": 0.907,
        "mrr": 0.907,
        "notes": "Full evaluation"
    },
    {
        "id": 6,
        "name": "Baseline RAG Full (K=3)",
        "db_content": "887 paper abstracts + titles",
        "db_size": "887 papers",
        "query_type": "734 single paper specific queries",
        "num_queries": 734,
        "k": 3,
        "hit_rate": 0.962,
        "mrr": 0.933,
        "notes": "Full evaluation"
    },
    {
        "id": 7,
        "name": "Baseline RAG Full (K=5)",
        "db_content": "887 paper abstracts + titles",
        "db_size": "887 papers",
        "query_type": "734 single paper specific queries",
        "num_queries": 734,
        "k": 5,
        "hit_rate": 0.970,
        "mrr": 0.935,
        "notes": "Full evaluation"
    },
    {
        "id": 8,
        "name": "Baseline RAG Full (K=10)",
        "db_content": "887 paper abstracts + titles",
        "db_size": "887 papers",
        "query_type": "734 single paper specific queries",
        "num_queries": 734,
        "k": 10,
        "hit_rate": 0.971,
        "mrr": 0.935,
        "notes": "Full evaluation"
    },
    {
        "id": 9,
        "name": "Baseline RAG Full (K=20)",
        "db_content": "887 paper abstracts + titles",
        "db_size": "887 papers",
        "query_type": "734 single paper specific queries",
        "num_queries": 734,
        "k": 20,
        "hit_rate": 0.974,
        "mrr": 0.935,
        "notes": "Full evaluation"
    },
    {
        "id": 10,
        "name": "Triple RAG Full (K=1)",
        "db_content": "CSKG triples with W IDs",
        "db_size": "10,000 facts / 909 papers",
        "query_type": "734 single paper specific queries",
        "num_queries": 734,
        "k": 1,
        "hit_rate": 0.470,
        "mrr": 0.470,
        "notes": "Full evaluation - same queries as baseline"
    },
    {
        "id": 11,
        "name": "Triple RAG Full (K=3)",
        "db_content": "CSKG triples with W IDs",
        "db_size": "10,000 facts / 909 papers",
        "query_type": "734 single paper specific queries",
        "num_queries": 734,
        "k": 3,
        "hit_rate": 0.597,
        "mrr": 0.526,
        "notes": "Full evaluation"
    },
    {
        "id": 12,
        "name": "Triple RAG Full (K=5)",
        "db_content": "CSKG triples with W IDs",
        "db_size": "10,000 facts / 909 papers",
        "query_type": "734 single paper specific queries",
        "num_queries": 734,
        "k": 5,
        "hit_rate": 0.640,
        "mrr": 0.536,
        "notes": "Full evaluation"
    },
    {
        "id": 13,
        "name": "Triple RAG Full (K=10)",
        "db_content": "CSKG triples with W IDs",
        "db_size": "10,000 facts / 909 papers",
        "query_type": "734 single paper specific queries",
        "num_queries": 734,
        "k": 10,
        "hit_rate": 0.683,
        "mrr": 0.541,
        "notes": "Full evaluation"
    },
    {
        "id": 14,
        "name": "Triple RAG Full (K=20)",
        "db_content": "CSKG triples with W IDs",
        "db_size": "10,000 facts / 909 papers",
        "query_type": "734 single paper specific queries",
        "num_queries": 734,
        "k": 20,
        "hit_rate": 0.725,
        "mrr": 0.544,
        "notes": "Full evaluation"
    }
]

# ─────────────────────────────────────────────
# Print master table
# ─────────────────────────────────────────────

print("=" * 100)
print("MASTER RESULTS TABLE - ALL EXPERIMENTS")
print("=" * 100)
print("{:<4} {:<45} {:<6} {:<12} {:<8} {}".format(
    "ID", "Experiment", "K", "Hit Rate@K", "MRR", "Notes"
))
print("-" * 100)

for exp in experiments:
    print("{:<4} {:<45} {:<6} {:<12} {:<8} {}".format(
        exp["id"],
        exp["name"][:44],
        exp["k"],
        str(exp["hit_rate"]),
        str(exp["mrr"]),
        exp["notes"][:40]
    ))

print("=" * 100)

# ─────────────────────────────────────────────
# Save as CSV
# ─────────────────────────────────────────────

csv_file = os.path.join(current_dir, "master_results_table.csv")
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "id", "name", "db_content", "db_size",
        "query_type", "num_queries", "k",
        "hit_rate", "mrr", "notes"
    ])
    writer.writeheader()
    writer.writerows(experiments)

# Save as JSON
json_file = os.path.join(current_dir, "master_results_table.json")
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(experiments, f, indent=2)

print("\nSaved to master_results_table.csv and master_results_table.json")

# ─────────────────────────────────────────────
# Print clean comparison (Baseline vs Triple)
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("KEY COMPARISON: Baseline RAG vs Triple RAG (Full Eval)")
print("=" * 60)
print("{:<6} {:<20} {:<20} {:<12}".format(
    "K", "Baseline (Abstracts)", "Triple (CSKG)", "Difference"
))
print("-" * 60)

baseline = {1: 0.907, 3: 0.962, 5: 0.970, 10: 0.971, 20: 0.974}
triple = {1: 0.470, 3: 0.597, 5: 0.640, 10: 0.683, 20: 0.725}

for k in [1, 3, 5, 10, 20]:
    diff = round(baseline[k] - triple[k], 3)
    print("{:<6} {:<20} {:<20} {:<12}".format(
        k,
        str(baseline[k]),
        str(triple[k]),
        "+" + str(diff)
    ))

print("=" * 60)