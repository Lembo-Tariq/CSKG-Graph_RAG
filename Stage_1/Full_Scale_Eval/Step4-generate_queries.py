"""
step4_generate_queries.py
=========================
Generates ONE specific query per paper for all valid papers.
Has resume capability - picks up where it left off if crashed.
"""

import os
import json
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

NUM_SAMPLES = 2000  # must match step2

current_dir = os.path.dirname(os.path.abspath(__file__))

abstracts_file = os.path.join(current_dir, "Step3Out-papers_" + str(NUM_SAMPLES) + "_abstracts.json")
output_file    = os.path.join(current_dir, "Step4Out-eval_dataset_" + str(NUM_SAMPLES) + ".json")

with open(abstracts_file, "r", encoding="utf-8") as f:
    papers = json.load(f)

valid_papers = [
    p for p in papers
    if p.get("abstract") and p["abstract"] != "Abstract not available"
    and p.get("title") and p["title"] != "Error fetching"
    and p["title"] != "No title"
]

print("Total papers: " + str(len(papers)))
print("Valid papers with abstracts: " + str(len(valid_papers)))

# Resume capability
if os.path.exists(output_file):
    with open(output_file, "r", encoding="utf-8") as f:
        eval_dataset = json.load(f)
    done_ids = {e["paper_id"] for e in eval_dataset}
    print("Resuming from " + str(len(eval_dataset)) + " existing queries...")
else:
    eval_dataset = []
    done_ids = set()

failed = []
client = Groq()

print("Generating queries...")

for i, paper in enumerate(valid_papers):
    paper_id = paper["openalex_id"]
    title = str(paper["title"])
    abstract = str(paper["abstract"])

    if paper_id in done_ids:
        continue

    prompt = (
        "You are simulating a researcher searching for a specific academic paper.\n\n"
        "Given this paper:\n"
        "Title: " + title + "\n"
        "Abstract: " + abstract[:500] + "\n\n"
        "Generate exactly ONE specific search query that would help find THIS specific paper.\n"
        "The query should:\n"
        "- Be 8-15 words long\n"
        "- Include specific technical terms from this paper\n"
        "- Be specific enough to distinguish this paper from others on similar topics\n"
        "- Sound like a real search query a researcher would type\n"
        "- NOT be a question\n\n"
        "Return ONLY the query as plain text, nothing else."
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        query = response.choices[0].message.content.strip()
        query = query.replace('"', '').replace("'", '').strip()

        eval_dataset.append({
            "sample_id": i + 1,
            "paper_id": paper_id,
            "title": title,
            "query": query,
            "abstract": abstract[:300]
        })

        # Save after every query
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(eval_dataset, f, indent=2, ensure_ascii=False)

        if len(eval_dataset) % 100 == 0:
            print("Progress: " + str(len(eval_dataset)) + "/" + str(len(valid_papers)) + " queries...")

    except Exception as e:
        failed.append(paper_id)
        print("Failed: " + paper_id + " - " + str(e))

    time.sleep(1)

print("Done! " + str(len(eval_dataset)) + " queries saved")
print("Failed: " + str(len(failed)))
print("Preview (first 3):")
for e in eval_dataset[:3]:
    print("  Paper: " + e["title"][:50] + "...")
    print("  Query: " + e["query"])