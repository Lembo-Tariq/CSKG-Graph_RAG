"""
generate_queries_v2.py
======================
Step 2 of the new evaluation pipeline.
Fetches abstracts from OpenAlex for each paper in each sample,
feeds combined abstracts to Groq LLaMA to generate ONE query
per sample, and saves the evaluation dataset.
"""

import os
import json
import time
import requests
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
POLITE_EMAIL = "your_email@gmail.com"  # ← change this

# ─────────────────────────────────────────────
# STEP 1: Load samples
# ─────────────────────────────────────────────

samples_file = os.path.join(current_dir, "samples.json")
with open(samples_file, "r", encoding="utf-8") as f:
    samples = json.load(f)

print(f"✅ Loaded {len(samples)} samples")

# ─────────────────────────────────────────────
# STEP 2: Helper — reconstruct abstract
# ─────────────────────────────────────────────

def reconstruct_abstract(inverted_index: dict) -> str:
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

def fetch_abstract(paper_id: str) -> dict:
    """
    Fetches title and abstract for a paper from OpenAlex.
    paper_id is the W number e.g. W1010415138
    """
    url = f"https://api.openalex.org/works/{paper_id}?mailto={POLITE_EMAIL}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        title = data.get("display_name", "No title")
        abstract_index = data.get("abstract_inverted_index", {})
        abstract = reconstruct_abstract(abstract_index)
        
        return {
            "title": title,
            "abstract": abstract if abstract else "Abstract not available"
        }
    except Exception as e:
        return {
            "title": "Error fetching",
            "abstract": "Abstract not available"
        }

# ─────────────────────────────────────────────
# STEP 4: Generate ONE query per sample
# ─────────────────────────────────────────────

client = Groq()
eval_dataset = []

print(f"\n🌐 Fetching abstracts and generating queries...")

for sample in samples:
    sample_id = sample["sample_id"]
    paper_ids = sample["paper_ids"]
    
    print(f"\n📦 Sample {sample_id}/{len(samples)} — {len(paper_ids)} papers")
    
    # Fetch abstracts for all papers in this sample
    papers_info = []
    for paper_id in paper_ids:
        print(f"   Fetching: {paper_id}...")
        info = fetch_abstract(paper_id)
        info["openalex_id"] = paper_id
        papers_info.append(info)
        time.sleep(0.5)  # be polite to OpenAlex
    
    # Build combined context from all abstracts
    combined_context = ""
    for info in papers_info:
        combined_context += f"Title: {info['title']}\n"
        combined_context += f"Abstract: {info['abstract'][:300]}\n\n"
    
    # Generate ONE query from combined abstracts
    prompt = f"""You are simulating a researcher searching for academic papers.

Given these {len(papers_info)} related papers:

{combined_context}

Generate exactly ONE realistic natural language research query that:
- A researcher would type to find papers on this topic
- Captures the common theme across ALL these papers
- Is 10-20 words long
- Sounds like a real search query, not a question

Return ONLY the query as a plain string, no quotes, no explanation."""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        query = response.choices[0].message.content.strip()
        query = query.replace('"', '').replace("'", '').strip()
        
        print(f"   ✅ Query: {query}")
        
        eval_dataset.append({
            "sample_id": sample_id,
            "query": query,
            "paper_ids": paper_ids,
            "papers": papers_info
        })
    
    except Exception as e:
        print(f"   ❌ Error generating query: {e}")
    
    time.sleep(1)  # be polite to Groq

# ─────────────────────────────────────────────
# STEP 5: Save evaluation dataset
# ─────────────────────────────────────────────

output_file = os.path.join(current_dir, "eval_dataset_v2.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(eval_dataset, f, indent=2, ensure_ascii=False)

print(f"\n✨ Done! Saved to: {output_file}")
print(f"📊 {len(eval_dataset)} samples × 1 query = {len(eval_dataset)} total queries")