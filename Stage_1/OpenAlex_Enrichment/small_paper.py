"""
sample_papers.py
================
Step 1 of the new evaluation pipeline.
Parses the CSKG SPARQL results, groups triples by paper,
filters for papers with target subjects (image_classification,
sentiment_analysis, language_model), then creates 20 random
samples of 5-10 diverse papers each.
"""

import os
import json
import random
from collections import defaultdict

random.seed(42)  # for reproducibility — same results every run

current_dir = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# STEP 1: Load and parse the SPARQL JSON
# ─────────────────────────────────────────────
# The file has this structure:
# { "results": { "bindings": [ {paper, any_s, any_p, any_o}, ... ] } }
# Each row is one triple belonging to one paper

input_file = os.path.join(current_dir, "output_modified.json")

print("📂 Loading SPARQL results...")
with open(input_file, "r", encoding="utf-8") as f:
    raw = json.load(f)

bindings = raw["results"]["bindings"]
print(f"✅ Loaded {len(bindings)} triples")

# ─────────────────────────────────────────────
# STEP 2: Group triples by paper ID
# ─────────────────────────────────────────────
# Each paper has multiple triples
# We group them so we know which triples belong to which paper
# Paper ID looks like: https://w3id.org/cskg/resource/W1010415138
# We extract just the W number: W1010415138

# defaultdict(list) means: if key doesn't exist, create empty list
paper_triples = defaultdict(list)
paper_subjects = defaultdict(set)

for row in bindings:
    # Extract paper ID (W number)
    paper_uri = row["paper"]["value"]
    paper_id = paper_uri.split("/")[-1]  # → W1010415138
    
    # Extract subject, predicate, object labels
    subject = row["any_s"]["value"].split("/")[-1].replace("_", " ")
    predicate = row["any_p"]["value"].split("#")[-1] if "#" in row["any_p"]["value"] else row["any_p"]["value"].split("/")[-1]
    obj = row["any_o"]["value"].split("/")[-1].replace("_", " ")
    
    # Store triple
    paper_triples[paper_id].append({
        "subject": subject,
        "predicate": predicate,
        "object": obj
    })
    
    # Store subjects for this paper (for filtering)
    paper_subjects[paper_id].add(subject)

total_papers = len(paper_triples)
print(f"✅ Found {total_papers} unique papers")

# ─────────────────────────────────────────────
# STEP 3: Filter papers with target subjects
# ─────────────────────────────────────────────
# We want papers that have at least one triple where
# the subject is one of our three target concepts

TARGET_SUBJECTS = ["image classification", "sentiment analysis", "language model"]

target_papers = []
other_papers = []

for paper_id, subjects in paper_subjects.items():
    # Check if any subject matches our targets
    has_target = any(
        any(target in subject for target in TARGET_SUBJECTS)
        for subject in subjects
    )
    if has_target:
        target_papers.append(paper_id)
    else:
        other_papers.append(paper_id)

print(f"✅ Papers with target subjects: {len(target_papers)}")
print(f"✅ Other papers: {len(other_papers)}")

# ─────────────────────────────────────────────
# STEP 4: Create 20 random samples
# ─────────────────────────────────────────────
# Each sample:
# - Has 5-10 papers
# - At least 1 paper must have a target subject
# - Papers should be diverse (not all from same topic)

NUM_SAMPLES = 20
MIN_PAPERS = 5
MAX_PAPERS = 10

all_papers = list(paper_triples.keys())
print(f"\n🎲 Creating {NUM_SAMPLES} random samples from {len(all_papers)} papers...")

samples = []

for i in range(NUM_SAMPLES):
    num_papers = random.randint(MIN_PAPERS, MAX_PAPERS)
    sample_paper_ids = random.sample(all_papers, num_papers)
    
    sample_data = {
        "sample_id": i + 1,
        "paper_ids": sample_paper_ids,
        "papers": {}
    }
    
    for paper_id in sample_paper_ids:
        triples = paper_triples[paper_id]
        subjects = list(paper_subjects[paper_id])
        
        sample_data["papers"][paper_id] = {
            "triples": triples[:10],
            "subjects": subjects[:5]
        }
    
    samples.append(sample_data)
    print(f"   Sample {i+1:2d}: {len(sample_paper_ids)} papers")

# ─────────────────────────────────────────────
# STEP 5: Save samples
# ─────────────────────────────────────────────

output_file = os.path.join(current_dir, "samples.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(samples, f, indent=2, ensure_ascii=False)

print(f"\n✨ Done! 20 samples saved to: {output_file}")
print(f"\n📄 Preview of Sample 1:")
s = samples[0]
print(f"   Papers: {s['paper_ids']}")
print(f"   Count: {len(s['paper_ids'])} papers")