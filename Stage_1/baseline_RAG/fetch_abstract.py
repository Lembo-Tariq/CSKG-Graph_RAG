"""
fetch_abstracts.py
==================
Fetches titles and abstracts for all 909 unique papers
from OpenAlex using their W IDs from the CSKG dataset.
Saves results to papers_with_abstracts.json
"""

import os
import json
import time
import requests
from collections import defaultdict

current_dir = os.path.dirname(os.path.abspath(__file__))
POLITE_EMAIL = "your_email@gmail.com"  # ← change this

# ─────────────────────────────────────────────
# STEP 1: Load SPARQL dataset and extract W IDs
# ─────────────────────────────────────────────
# We load the same dataset Dr. Danilo gave us
# and extract all unique paper W IDs

input_file = os.path.join(current_dir, "..", "RA_Parsing_CSKG-Text", "output_modified.json")

print("📂 Loading SPARQL dataset...")
with open(input_file, "r", encoding="utf-8") as f:
    raw = json.load(f)

bindings = raw["results"]["bindings"]

# Extract unique paper IDs
paper_ids = set()
for row in bindings:
    paper_uri = row["paper"]["value"]
    paper_id = paper_uri.split("/")[-1]  # → W1010415138
    paper_ids.add(paper_id)

paper_ids = list(paper_ids)
print(f"✅ Found {len(paper_ids)} unique papers")

# ─────────────────────────────────────────────
# STEP 2: Helper — reconstruct abstract
# ─────────────────────────────────────────────

def reconstruct_abstract(inverted_index: dict) -> str:
    """
    OpenAlex stores abstracts as inverted index.
    We reconstruct into readable text.
    """
    if not inverted_index:
        return ""
    words = []
    for word, positions in inverted_index.items():
        for pos in positions:
            words.append((pos, word))
    words.sort(key=lambda x: x[0])
    return " ".join(word for _, word in words)

# ─────────────────────────────────────────────
# STEP 3: Fetch abstracts from OpenAlex
# ─────────────────────────────────────────────

print(f"\n🌐 Fetching abstracts for {len(paper_ids)} papers...")
print("   This may take a few minutes...\n")

papers = []
failed = []

for i, paper_id in enumerate(paper_ids):
    url = f"https://api.openalex.org/works/{paper_id}?mailto={POLITE_EMAIL}"
    
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
        
        paper = {
            "openalex_id": paper_id,
            "title": title,
            "year": year,
            "authors": authors,
            "abstract": abstract if abstract else "Abstract not available"
        }
        papers.append(paper)
        
        # Progress update every 50 papers
        if (i + 1) % 50 == 0:
            print(f"   ✅ Fetched {i+1}/{len(paper_ids)} papers...")
    
    except Exception as e:
        failed.append(paper_id)
        papers.append({
            "openalex_id": paper_id,
            "title": "Error fetching",
            "year": "N/A",
            "authors": [],
            "abstract": "Abstract not available"
        })
    
    # Be polite to OpenAlex — 0.5 second between requests
    time.sleep(0.5)

print(f"\n✅ Successfully fetched: {len(papers) - len(failed)} papers")
print(f"⚠️  Failed: {len(failed)} papers")

# ─────────────────────────────────────────────
# STEP 4: Save results
# ─────────────────────────────────────────────

output_file = os.path.join(current_dir, "papers_with_abstracts.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(papers, f, indent=2, ensure_ascii=False)

print(f"\n✨ Done! Saved to: {output_file}")
print(f"\n📄 Preview (first 3 papers):")
for p in papers[:3]:
    print(f"\n   ID:       {p['openalex_id']}")
    print(f"   Title:    {p['title'][:60]}...")
    print(f"   Year:     {p['year']}")
    print(f"   Authors:  {', '.join(p['authors'])}")
    print(f"   Abstract: {p['abstract'][:100]}...")