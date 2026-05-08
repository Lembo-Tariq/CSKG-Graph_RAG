"""
generate_queries_full.py
========================
Generates ONE specific query per paper for ALL 734 valid papers.
Same approach as generate_queries_single.py but for full dataset.
"""

import os
import json
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))

# Load all 734 papers
samples_file = os.path.join(current_dir, "all_paper_samples.json")
with open(samples_file, "r", encoding="utf-8") as f:
    samples = json.load(f)

print("Loaded " + str(len(samples)) + " papers")

client = Groq()
eval_dataset = []
failed = []

print("Generating queries... (this will take ~15 minutes)")

for sample in samples:
    sample_id = sample["sample_id"]
    paper_id = sample["paper_id"]
    title = sample["title"]
    abstract = sample["abstract"]

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

        eval_dataset.append({
            "sample_id": sample_id,
            "paper_id": paper_id,
            "title": title,
            "query": query,
            "abstract": abstract[:300]
        })

        # Progress update every 50 papers
        if sample_id % 50 == 0:
            print("Progress: " + str(sample_id) + "/734 done...")

    except Exception as e:
        failed.append(paper_id)
        print("Failed sample " + str(sample_id) + ": " + str(e))

    time.sleep(1)

# Save
output_file = os.path.join(current_dir, "eval_dataset_full.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(eval_dataset, f, indent=2, ensure_ascii=False)

print("Done! " + str(len(eval_dataset)) + " queries saved")
print("Failed: " + str(len(failed)))
print("Preview (first 3):")
for e in eval_dataset[:3]:
    print("  Paper: " + e["title"][:50] + "...")
    print("  Query: " + e["query"])