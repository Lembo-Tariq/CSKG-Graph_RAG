"""
generate_queries.py
===================
Step 2 of the evaluation pipeline.
Reads the OpenAlex papers JSON, deduplicates by OpenAlex ID,
fetches abstracts, feeds them to Groq to generate realistic
research queries, and saves the evaluation dataset.
"""

import os
import json
import time
import requests
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# STEP 1: Load and deduplicate papers
# ─────────────────────────────────────────────
# We deduplicate by OpenAlex ID to avoid sending
# the same paper twice to Groq

papers_file = os.path.join(current_dir, "openalex_papers.json")

with open(papers_file, "r", encoding="utf-8") as f:
    papers = json.load(f)

# Deduplicate by OpenAlex ID
seen_titles = set()
unique_papers = []
for paper in papers:
    title = paper["title"]
    if title not in seen_titles:
        seen_titles.add(title)
        unique_papers.append(paper)

print(f"✅ Loaded {len(papers)} papers, {len(unique_papers)} unique after deduplication")

# ─────────────────────────────────────────────
# STEP 2: Fetch abstracts from OpenAlex
# ─────────────────────────────────────────────
# We need the abstract of each paper to feed to Groq
# OpenAlex stores abstracts as an "inverted index" which
# we need to reconstruct into readable text

POLITE_EMAIL = "Tariq.Ribhi.Yaseen@gmail.com"

def reconstruct_abstract(inverted_index: dict) -> str:
    """
    OpenAlex stores abstracts as an inverted index:
    {"word": [position1, position2], ...}
    We reconstruct it back into a normal sentence.
    """
    if not inverted_index:
        return ""
    
    # Build a list of (position, word) pairs
    words = []
    for word, positions in inverted_index.items():
        for pos in positions:
            words.append((pos, word))
    
    # Sort by position and join
    words.sort(key=lambda x: x[0])
    return " ".join(word for _, word in words)

print("\n🌐 Fetching abstracts from OpenAlex...")

for paper in unique_papers:
    openalex_id = paper["openalex_id"]
    # Extract just the ID part (W1234567) from the full URL
    short_id = openalex_id.split("/")[-1]
    
    url = f"https://api.openalex.org/works/{short_id}?mailto={POLITE_EMAIL}"
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        # Reconstruct abstract from inverted index
        abstract_index = data.get("abstract_inverted_index", {})
        abstract = reconstruct_abstract(abstract_index)
        paper["abstract"] = abstract if abstract else "Abstract not available"
        
        print(f"   ✅ Got abstract for: {paper['title'][:50]}...")
    
    except Exception as e:
        paper["abstract"] = "Abstract not available"
        print(f"   ⚠️  Could not fetch abstract: {e}")
    
    time.sleep(1)

# ─────────────────────────────────────────────
# STEP 3: Generate queries using Groq 
# ─────────────────────────────────────────────
# For each paper we feed the title + abstract to Groq
# and ask it to generate 3 realistic research queries
# that someone would type to find this paper



print("\n🤖 Generating queries with Groq...")

eval_dataset = []

for i, paper in enumerate(unique_papers):
    print(f"\n   [{i+1}/{len(unique_papers)}] {paper['title'][:50]}...")
    
    prompt = f"""You are simulating a researcher searching for academic papers.

Given this paper:
Title: {paper['title']}
Abstract: {paper['abstract']}

Generate exactly 3 realistic, natural language search queries that a researcher would type to find this paper.
The queries should:
- Be short (5-15 words)
- Sound like real search queries, not questions
- Cover different aspects of the paper
- Be diverse from each other

Return ONLY a JSON array of 3 strings, nothing else. Example:
["query one here", "query two here", "query three here"]"""

    try:
        client = Groq()
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )

        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        queries = json.loads(raw)
        
        print(f"      ✅ Generated {len(queries)} queries")
        for q in queries:
            print(f"         - {q}")
        
        # Save to eval dataset
        eval_dataset.append({
            "openalex_id": paper["openalex_id"],
            "doi": paper["doi"],
            "title": paper["title"],
            "year": paper["year"],
            "citations": paper["citations"],
            "facts_in_cskg": paper["facts_in_cskg"],
            "abstract": paper["abstract"][:300] + "...",
            "generated_queries": queries
        })
    
    except Exception as e:
        print(f"      ❌ Error generating queries: {e}")
    
    # Be polite to Groq API
    time.sleep(2)

# ─────────────────────────────────────────────
# STEP 4: Save evaluation dataset
# ─────────────────────────────────────────────

output_file = os.path.join(current_dir, "eval_dataset.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(eval_dataset, f, indent=2, ensure_ascii=False)

print(f"\n✨ Done! Evaluation dataset saved to: {output_file}")
print(f"📊 {len(eval_dataset)} papers × 3 queries = {len(eval_dataset)*3} total queries")