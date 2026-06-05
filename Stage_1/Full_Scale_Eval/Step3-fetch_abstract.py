"""
step3_fetch_abstracts.py
========================
Fetches abstracts for sampled papers from OpenAlex.
Reads from sampled_{NUM_SAMPLES}_papers.json
"""

import os
import json
import time
import requests

NUM_SAMPLES = 2000  # must match step2

current_dir = os.path.dirname(os.path.abspath(__file__))
POLITE_EMAIL = "tariq.ribhi.yaseen@gmail.com"

samples_file   = os.path.join(current_dir, "Step2Out-sampled_" + str(NUM_SAMPLES) + "_papers.json")
output_file    = os.path.join(current_dir, "Step3Out-papers_" + str(NUM_SAMPLES) + "_abstracts.json")

with open(samples_file, "r") as f:
    samples_data = json.load(f)

paper_ids = samples_data["paper_ids"]
print("Fetching abstracts for " + str(len(paper_ids)) + " papers...")

def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return ""
    words = []
    for word, positions in inverted_index.items():
        for pos in positions:
            words.append((pos, word))
    words.sort(key=lambda x: x[0])
    return " ".join(word for _, word in words)

papers = []
failed = []

for i, paper_id in enumerate(paper_ids):
    url = "https://api.openalex.org/works/" + paper_id + "?mailto=" + POLITE_EMAIL

    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        title = data.get("display_name", "No title")
        year = data.get("publication_year", "N/A")
        abstract_index = data.get("abstract_inverted_index", {})
        abstract = reconstruct_abstract(abstract_index)
        authors = [
            a["author"]["display_name"]
            for a in data.get("authorships", [])[:3]
        ]

        papers.append({
            "openalex_id": paper_id,
            "title": title,
            "year": year,
            "authors": authors,
            "abstract": abstract if abstract else "Abstract not available"
        })

        if (i + 1) % 100 == 0:
            print("Fetched " + str(i+1) + "/" + str(len(paper_ids)) + " papers...")

    except Exception as e:
        failed.append(paper_id)
        papers.append({
            "openalex_id": paper_id,
            "title": "Error fetching",
            "year": "N/A",
            "authors": [],
            "abstract": "Abstract not available"
        })

    time.sleep(0.5)

print("Successfully fetched: " + str(len(papers) - len(failed)))
print("Failed: " + str(len(failed)))

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(papers, f, indent=2, ensure_ascii=False)

print("Saved to: " + output_file)