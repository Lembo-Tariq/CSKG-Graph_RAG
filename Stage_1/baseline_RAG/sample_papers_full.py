"""
sample_papers_full.py
=====================
Uses ALL valid papers (with abstracts) instead of just 20.
734 papers = 734 queries = more reliable evaluation.
"""

import os
import json

current_dir = os.path.dirname(os.path.abspath(__file__))

print("Loading papers...")
input_file = os.path.join(current_dir, "papers_with_abstracts.json")
with open(input_file, "r", encoding="utf-8") as f:
    papers = json.load(f)

# Filter out papers with no abstract or failed fetches
valid_papers = [
    p for p in papers
    if p["abstract"] != "Abstract not available"
    and p["title"] != "Error fetching"
    and p["title"] != "No title"
]

print("Total papers: " + str(len(papers)))
print("Valid papers with abstracts: " + str(len(valid_papers)))

# Use ALL valid papers — no random sampling
samples = []
for i, paper in enumerate(valid_papers):
    samples.append({
        "sample_id": i + 1,
        "paper_id": paper["openalex_id"],
        "title": paper["title"],
        "authors": paper["authors"],
        "year": paper["year"],
        "abstract": paper["abstract"]
    })

output_file = os.path.join(current_dir, "all_paper_samples.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(samples, f, indent=2, ensure_ascii=False)

print("Saved " + str(len(samples)) + " samples to: " + output_file)
print("Preview (first 3):")
for s in samples[:3]:
    print("  " + str(s["sample_id"]) + ": " + s["title"][:60] + "...")