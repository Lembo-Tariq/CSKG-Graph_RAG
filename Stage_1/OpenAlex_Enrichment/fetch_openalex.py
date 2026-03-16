"""
fetch_openalex.py
=================
Finds the most well-represented papers in our CSKG dataset
and fetches their metadata from the OpenAlex API.

No API key needed — OpenAlex is completely free and open.

Step 1: Extract all unique DOIs from cskg_text_10k.txt
        (we have 10k facts but many share the same DOI)

Step 2: Count how many facts each DOI supports
        (a DOI appearing 50 times = well represented in our DB)

Step 3: Pick the top 10 most frequent DOIs
        (these are the most "influential" papers in our dataset)

Step 4: Call the OpenAlex API with each DOI
        (convert DOI → OpenAlex ID + get paper metadata)

Step 5: Save the results
        (paper title, authors, year, OpenAlex ID)
"""

import json
import time
import requests
from collections import Counter
from urllib.parse import unquote
import os

# ─────────────────────────────────────────────
# STEP 1: Load the text file and extract DOIs
# ─────────────────────────────────────────────
# Each line looks like:
# "image classification uses the method neural network
#  (supported by 8 papers, source: https://doi.org/10.1109/...)"
# We extract the DOI from each line

current_dir = os.path.dirname(os.path.abspath(__file__))
text_file = os.path.join(current_dir, "..","RA_Parsing_CSKG-Text","cskg_text_10k.txt")

print("📂 Loading CSKG text file...")

dois=[]

with open(text_file, "r", encoding="utf8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # Each line ends with "source: https://doi.org/xxxx)"
        # We split on "source: " and take everything after it
        if "source" in line:
            doi_part = line.split("source:")[-1].strip()
            # Remove the closing parenthesis
            doi = doi_part.rstrip(")")
            dois.append(doi)
print(f"✅ Extracted {len(dois)} DOIs from {len(set(dois))} unique papers")

# ─────────────────────────────────────────────
# STEP 2: Count frequency of each DOI
# ─────────────────────────────────────────────
# A DOI appearing many times = paper is well represented
# in our knowledge graph

doi_counts = Counter(dois)
"""
    The Counter class counts how many instance in a list is found
    i.e : dois = [
            "10.1016/j.cell.2020.01.001",
            "10.1038/nature12345",
            "10.1016/j.cell.2020.01.001",
            "10.1126/science.abc123",
        ]
        Counter({
            "10.1016/j.cell.2020.01.001": 2,
            "10.1038/nature12345": 1,
            "10.1126/science.abc123": 1
        })
"""

print(f"\n📊 Top 10 most represented papers:")
for doi, count in doi_counts.most_common(10):
    print(f"   {count:3} facts → {doi}")

# Get top 10 DOIs
top_dois = [doi for doi, count in doi_counts.most_common(10)]

# ─────────────────────────────────────────────
# STEP 3: Query OpenAlex API for each DOI
# ─────────────────────────────────────────────
# OpenAlex API endpoint:
# https://api.openalex.org/works?filter=doi:YOUR_DOI
# No API key needed, but we add an email as "polite pool"
# which gives us faster, more reliable responses

POLITE_EMAIL = "Tariq.Ribhi.Yaseen@gmail.com"  

print(f"\n🌐 Querying OpenAlex API for top 10 papers...")

results = []

for i, doi in enumerate(top_dois):
    print(f"   [{i+1}/10] Fetching: {doi}")
    
    # Build the API URL
    # We filter by DOI to find the exact paper
    url = f"https://api.openalex.org/works?filter=doi:{doi}&mailto={POLITE_EMAIL}"
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        # OpenAlex returns a list of results
        # We take the first one
        if data["results"]:
            paper = data["results"][0]
            
            # Extract the fields we care about
            result = {
                "openalex_id": paper.get("id", "N/A"),
                "doi": doi,
                "title": paper.get("display_name", "N/A"),
                "year": paper.get("publication_year", "N/A"),
                "citations": paper.get("cited_by_count", 0),
                "authors": [
                    a["author"]["display_name"]
                    for a in paper.get("authorships", [])[:3]  # first 3 authors
                ],
                "facts_in_cskg": doi_counts[doi]
            }
            results.append(result)
            print(f"      ✅ Found: {result['title'][:60]}...")
        else:
            print(f"      ⚠️  Not found in OpenAlex")
    
    except Exception as e:
        print(f"      ❌ Error: {e}")
    
    # Be polite — wait 1 second between requests
    time.sleep(1)

# ─────────────────────────────────────────────
# STEP 4: Save results to JSON
# ─────────────────────────────────────────────

output_file = os.path.join(current_dir, "openalex_papers.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✨ Done! Results saved to: {output_file}")
print(f"\n📄 Summary:")
for r in results:
    print(f"\n   Title:      {r['title']}")
    print(f"   Year:       {r['year']}")
    print(f"   Citations:  {r['citations']}")
    print(f"   Authors:    {', '.join(r['authors'])}")
    print(f"   OpenAlex:   {r['openalex_id']}")
    print(f"   Facts in DB:{r['facts_in_cskg']}")