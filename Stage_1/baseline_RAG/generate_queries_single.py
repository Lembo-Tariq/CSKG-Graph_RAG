"""
generate_queries_single.py
==========================
Generates ONE specific query per paper using its abstract.
More specific than combining 5-10 papers together.
Result: 20 queries, each with exactly 1 correct paper.
"""

import os
import json
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# STEP 1: Load single paper samples
# ─────────────────────────────────────────────

samples_file = os.path.join(current_dir, "single_paper_samples.json")
with open(samples_file, "r", encoding="utf-8") as f:
    samples = json.load(f)

print("Loaded " + str(len(samples)) + " paper samples")

# ─────────────────────────────────────────────
# STEP 2: Generate one specific query per paper
# ─────────────────────────────────────────────
# Key difference from before:
# - One paper at a time (not 5-10 combined)
# - Query must be specific enough to find THIS paper
# - Not a question — a search query

client = Groq()
eval_dataset = []

print("\nGenerating queries with Groq LLaMA...")

for sample in samples:
    sample_id = sample["sample_id"]
    paper_id = sample["paper_id"]
    title = sample["title"]
    abstract = sample["abstract"]

    print("\nSample " + str(sample_id) + ": " + title[:50] + "...")

    prompt = """You are simulating a researcher searching for a specific academic paper.

Given this paper:
Title: """ + title + """
Abstract: """ + abstract[:500] + """

Generate exactly ONE specific search query that would help find THIS specific paper.
The query should:
- Be 8-15 words long
- Include specific technical terms from this paper
- Be specific enough to distinguish this paper from others on similar topics
- Sound like a real search query a researcher would type
- NOT be a question

Return ONLY the query as plain text, nothing else."""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        query = response.choices[0].message.content.strip()
        query = query.replace('"', '').replace("'", '').strip()

        print("Query: " + query)

        eval_dataset.append({
            "sample_id": sample_id,
            "paper_id": paper_id,
            "title": title,
            "query": query,
            "abstract": abstract[:300]
        })

    except Exception as e:
        print("Error: " + str(e))

    time.sleep(1)

# ─────────────────────────────────────────────
# STEP 3: Save evaluation dataset
# ─────────────────────────────────────────────

output_file = os.path.join(current_dir, "eval_dataset_single.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(eval_dataset, f, indent=2, ensure_ascii=False)

print("\nDone! Saved to: " + output_file)
print("Total queries: " + str(len(eval_dataset)))
print("\nPreview:")
for e in eval_dataset[:3]:
    print("\n  Paper: " + e["title"][:50] + "...")
    print("  Query: " + e["query"])