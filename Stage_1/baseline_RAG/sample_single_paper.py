"""
sample_single_papers.py
=======================
Samples 20 individual papers from papers_with_abstracts.json
One paper per query - gives specific queries and clean evaluation.
Each query has exactly 1 correct answer (Hit Rate = 0 or 1).
"""

import os
import json
import random

random.seed(42)

current_dir = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# STEP 1: Load papers with abstracts
# ─────────────────────────────────────────────
# We already have all 887 papers with their IDs
# No need to go back to raw SPARQL file

input_file = os.path.join(current_dir, "papers_with_abstracts.json")

print("Loading papers...")
with open(input_file, "r", encoding="utf-8") as f:
    papers = json.load(f)

# Filter out papers with no abstract or failed fetches
valid_papers = [
    p for p in papers
    if p["abstract"] != "Abstract not available"
    and p["title"] != "Error fetching"
]

print("Total papers: " + str(len(papers)))
print("Valid papers with abstracts: " + str(len(valid_papers)))

# ─────────────────────────────────────────────
# STEP 2: Randomly sample 20 papers
# ─────────────────────────────────────────────

NUM_SAMPLES = 20
sampled = random.sample(valid_papers, NUM_SAMPLES)

samples = []
for i, paper in enumerate(sampled):
    samples.append({
        "sample_id": i + 1,
        "paper_id": paper["openalex_id"],
        "title": paper["title"],
        "authors": paper["authors"],
        "year": paper["year"],
        "abstract": paper["abstract"]
    })
    print("Sample " + str(i+1) + ": " + paper["title"][:60] + "...")

# ─────────────────────────────────────────────
# STEP 3: Save samples
# ─────────────────────────────────────────────

output_file = os.path.join(current_dir, "single_paper_samples.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(samples, f, indent=2, ensure_ascii=False)

print("\nDone! Saved " + str(len(samples)) + " samples to: " + output_file)