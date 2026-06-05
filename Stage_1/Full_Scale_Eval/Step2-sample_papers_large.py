"""
sample_papers_large.py
======================
Samples N diverse papers from the large dataset.
Change NUM_SAMPLES to run different experiments.
All output filenames are dynamic based on NUM_SAMPLES.
"""

import os
import json
import random
from collections import defaultdict
import ast
import pandas as pd

random.seed(42)

current_dir = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# CONFIGURATION — change this number only
# ─────────────────────────────────────────────

NUM_SAMPLES = 2000  # change to 500, 1000, 2000 etc.

# All output files automatically named after NUM_SAMPLES
samples_file = os.path.join(current_dir, "sampled_" + str(NUM_SAMPLES) + "_papers.json")
output_text  = os.path.join(current_dir, "cskg_text_" + str(NUM_SAMPLES) + ".txt")

# ─────────────────────────────────────────────
# STEP 1: Load CSV
# ─────────────────────────────────────────────

input_file = os.path.join(current_dir, "Step1In-cskg_large.csv")
print("Loading CSV...")
df = pd.read_csv(input_file)
print("Loaded " + str(len(df)) + " rows")

# ─────────────────────────────────────────────
# STEP 2: Build paper triple counts
# ─────────────────────────────────────────────

print("Building paper triple counts...")
paper_triple_count = defaultdict(int)

def parse_paper_ids(raw):
    try:
        parsed = ast.literal_eval(raw)
        return list(parsed)
    except:
        return []

for i, row in df.iterrows():
    paper_ids = parse_paper_ids(row["wasDerivedFrom"])
    for paper_id in paper_ids:
        paper_triple_count[paper_id] += 1

    if (i + 1) % 100000 == 0:
        print("Progress: " + str(i+1) + "/" + str(len(df)) + " rows...")

print("Total unique papers: " + str(len(paper_triple_count)))

# ─────────────────────────────────────────────
# STEP 3: Filter papers with minimum triples
# ─────────────────────────────────────────────

MIN_TRIPLES = 3
valid_papers = [
    paper_id for paper_id, count in paper_triple_count.items()
    if count >= MIN_TRIPLES
]

print("Papers with at least " + str(MIN_TRIPLES) + " triples: " + str(len(valid_papers)))

# ─────────────────────────────────────────────
# STEP 4: Sample N papers
# ─────────────────────────────────────────────

sampled_papers = random.sample(valid_papers, NUM_SAMPLES)
sampled_set = set(sampled_papers)

print("Sampled " + str(len(sampled_papers)) + " papers")

with open(samples_file, "w") as f:
    json.dump({
        "total_valid_papers": len(valid_papers),
        "sampled_count": len(sampled_papers),
        "paper_ids": sampled_papers,
        "triple_counts": {p: paper_triple_count[p] for p in sampled_papers}
    }, f, indent=2)

print("Saved sampled papers to: " + samples_file)

# ─────────────────────────────────────────────
# STEP 5: Filter sentences for sampled papers
# ─────────────────────────────────────────────

print("Filtering sentences for sampled papers...")
input_text = os.path.join(current_dir, "Step1Out-cskg_text_large.txt")

kept = 0
total = 0

with open(input_text, "r", encoding="utf-8") as fin, \
     open(output_text, "w", encoding="utf-8") as fout:
    for line in fin:
        total += 1
        for paper_id in sampled_set:
            if paper_id in line:
                fout.write(line)
                kept += 1
                break

        if total % 500000 == 0:
            print("Processed " + str(total) + " lines, kept " + str(kept) + "...")

print("Done!")
print("Total lines processed: " + str(total))
print("Lines kept: " + str(kept))
print("Saved to: " + output_text)
print("Preview (first 5 lines):")

with open(output_text, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        print("  " + line.strip())