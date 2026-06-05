"""
parse_csv.py
============
Parses the new large-scale CSKG CSV dataset.
Key difference from old format:
- One triple can belong to MULTIPLE papers (wasDerivedFrom column)
- 729,164 triples from 109,786 unique papers
- Output: one line per triple per paper (exploded format)
"""

import os
import ast
import pandas as pd
from collections import defaultdict

current_dir = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# STEP 1: Load the CSV
# ─────────────────────────────────────────────

input_file = os.path.join(current_dir, "filtered_triples.csv")
output_file = os.path.join(current_dir, "cskg_text_large.txt")

print("Loading CSV dataset...")
df = pd.read_csv(input_file)
print("Shape: " + str(df.shape))
print("Columns: " + str(df.columns.tolist()))

# ─────────────────────────────────────────────
# STEP 2: Predicate mapping
# ─────────────────────────────────────────────

PREDICATE_MAP = {
    "usesMethod":          "uses the method",
    "usesTask":            "uses the task of",
    "usesMaterial":        "uses the material",
    "usesMetric":          "uses the metric",
    "usesOtherEntity":     "uses the entity",
    "analyzesTask":        "analyzes the task of",
    "analyzesMethod":      "analyzes the method",
    "identifiesMaterial":  "identifies the material",
    "identifiesMetric":    "identifies the metric",
    "identifiesMethod":    "identifies the method",
    "producesMethod":      "produces the method",
    "producesOtherEntity": "produces the entity",
    "executesMethod":      "executes the method",
    "matchesMethod":       "matches the method",
    "based-onMethod":      "is based on the method",
    "based-onTask":        "is based on the task of",
    "broader":             "is a broader concept than",
    "narrower":            "is a narrower concept than",
    "related":             "is related to",
    "exactMatch":          "exactly matches",
    "closeMatch":          "closely matches",
    "hasMethod":           "has the method",
    "hasTask":             "has the task of",
    "hasMaterial":         "has the material",
    "includesMethod":      "includes the method",
    "proposesMethod":      "proposes the method",
    "improvesMethod":      "improves the method",
    "improvesMetric":      "improves the metric",
    "predictsTask":        "predicts the task of",
    "acquiresMethod":      "acquires the method",
    "learnsMethod":        "learns the method",
    "solvesMethod":        "solves using the method",
    "affectsMethod":       "affects the method",
    "providesMethod":      "provides the method",
    "improvesOtherEntity": "improves the entity",
}

def predicate_to_text(pred):
    return PREDICATE_MAP.get(pred, pred)

def clean_label(label):
    return str(label).replace("_", " ").strip()

# ─────────────────────────────────────────────
# STEP 3: Parse wasDerivedFrom column
# ─────────────────────────────────────────────
# wasDerivedFrom looks like: {'W3046846102', 'W4287703121'}
# We need to parse this string into a Python set

def parse_paper_ids(raw):
    """
    Parses the wasDerivedFrom column into a list of W IDs.
    The column is stored as a string representation of a set.
    """
    try:
        # ast.literal_eval safely converts string to Python object
        parsed = ast.literal_eval(raw)
        return list(parsed)
    except:
        return []

# ─────────────────────────────────────────────
# STEP 4: Convert triples to sentences
# ─────────────────────────────────────────────
# For each triple we create ONE sentence per paper
# that contributed to it.
# This way each fact is linked to its specific paper.

print("Converting triples to sentences...")

sentences = []
paper_triple_count = defaultdict(int)
skipped = 0

for i, row in df.iterrows():
    try:
        subject   = clean_label(row["subj"])
        predicate = predicate_to_text(str(row["obj_prop"]).strip())
        obj       = clean_label(row["obj"])
        paper_ids = parse_paper_ids(row["wasDerivedFrom"])

        if not paper_ids:
            skipped += 1
            continue

        # Create one sentence per paper
        for paper_id in paper_ids:
            sentence = subject + " " + predicate + " " + obj + " (paper: " + paper_id + ")"
            sentences.append(sentence)
            paper_triple_count[paper_id] += 1

    except Exception as e:
        skipped += 1
        continue

    # Progress update every 100k rows
    if (i + 1) % 100000 == 0:
        print("Progress: " + str(i+1) + "/" + str(len(df)) + " rows processed...")

print("Done converting!")
print("Total sentences: " + str(len(sentences)))
print("Unique papers: " + str(len(paper_triple_count)))
print("Skipped rows: " + str(skipped))

# ─────────────────────────────────────────────
# STEP 5: Save to text file
# ─────────────────────────────────────────────

print("Saving to text file...")
with open(output_file, "w", encoding="utf-8") as f:
    for sentence in sentences:
        f.write(sentence + "\n")

print("Saved to: " + output_file)
print("Preview (first 5 lines):")
for s in sentences[:5]:
    print("  " + s)

# ─────────────────────────────────────────────
# STEP 6: Save paper statistics
# ─────────────────────────────────────────────
# Save which papers have the most triples
# useful for sampling later

import json
stats = {
    "total_sentences": len(sentences),
    "unique_papers": len(paper_triple_count),
    "skipped_rows": skipped,
    "top_20_papers": sorted(
        paper_triple_count.items(),
        key=lambda x: x[1],
        reverse=True
    )[:20]
}

stats_file = os.path.join(current_dir, "dataset_stats.json")
with open(stats_file, "w") as f:
    json.dump(stats, f, indent=2)

print("Dataset stats saved to: " + stats_file)