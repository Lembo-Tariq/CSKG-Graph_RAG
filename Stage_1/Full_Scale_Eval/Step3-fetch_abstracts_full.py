"""
Step3-fetch_abstracts_full.py
=============================
Fetches abstracts for ALL 807,912 unique papers from OpenAlex.
- Reads paper IDs from unique_paper_ids.txt
- Saves progress every 1000 papers (resume capability)
- If interrupted, run again and it picks up where it left off
Estimated time: ~67 hours at 0.3s per request
"""

import os
import json
import time
import requests

current_dir = os.path.dirname(os.path.abspath(__file__))
POLITE_EMAIL = "tariq.ribhi.yaseen@gmail.com"

ids_file    = os.path.join(current_dir, "unique_paper_ids.txt")
output_file = os.path.join(current_dir, "papers_full_abstracts.json")

# Load all paper IDs
with open(ids_file, "r") as f:
    all_paper_ids = [line.strip() for line in f if line.strip()]

print("Total papers to fetch: " + str(len(all_paper_ids)))

# Resume capability
if os.path.exists(output_file):
    with open(output_file, "r", encoding="utf-8") as f:
        papers = json.load(f)
    done_ids = {p["openalex_id"] for p in papers}
    print("Resuming from " + str(len(done_ids)) + " already fetched...")
else:
    papers = []
    done_ids = set()

def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return ""
    words = []
    for word, positions in inverted_index.items():
        for pos in positions:
            words.append((pos, word))
    words.sort(key=lambda x: x[0])
    return " ".join(word for _, word in words)

failed = 0
total_done = len(done_ids)

for i, paper_id in enumerate(all_paper_ids):
    if paper_id in done_ids:
        continue

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
        done_ids.add(paper_id)
        total_done += 1

    except Exception as e:
        failed += 1
        papers.append({
            "openalex_id": paper_id,
            "title": "Error fetching",
            "year": "N/A",
            "authors": [],
            "abstract": "Abstract not available"
        })
        done_ids.add(paper_id)
        total_done += 1

    # Save every 1000 papers
    if total_done % 1000 == 0:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(papers, f, ensure_ascii=False)
        print("Progress: " + str(total_done) + "/807912 fetched, " + str(failed) + " failed...")

    time.sleep(0.3)

# Final save
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(papers, f, indent=2, ensure_ascii=False)

print("Done! Total fetched: " + str(len(papers)))
print("Failed: " + str(failed))