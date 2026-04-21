"""
build_baseline_db.py
====================
Builds a Chroma vector database from paper abstracts and titles.
This is the BASELINE RAG - uses actual paper content instead of
generic CSKG triples. Much fairer for paper-level retrieval.

Each chunk = one paper's full metadata:
  Title + Authors + Year + Abstract
  with paper ID embedded for evaluation.
"""

import os
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

current_dir = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# STEP 1: Load papers with abstracts
# ─────────────────────────────────────────────

input_file = os.path.join(current_dir, "papers_with_abstracts.json")
db_dir = os.path.join(current_dir, "db", "chroma_db_baseline")

print("Loading papers with abstracts...")
with open(input_file, "r", encoding="utf-8") as f:
    papers = json.load(f)

print(f"Loaded {len(papers)} papers")

# ─────────────────────────────────────────────
# STEP 2: Create one document per paper
# ─────────────────────────────────────────────
# Format:
# "Title: xxx
#  Authors: xxx
#  Year: xxx
#  Abstract: xxx
#  (paper: W1010415138)"
#
# We embed the paper ID at the end so evaluation
# can check if the correct paper was retrieved
# using the same W ID matching approach

docs = []
skipped = 0

for paper in papers:
    title = paper.get("title", "No title")
    authors = ", ".join(paper.get("authors", []))
    year = paper.get("year", "N/A")
    abstract = paper.get("abstract", "Abstract not available")
    paper_id = paper.get("openalex_id", "")

    # Skip papers with no useful content
    if title == "Error fetching" or title == "No title":
        skipped += 1
        continue

    # Build the chunk text
    chunk = f"Title: {title}\n"
    chunk += f"Authors: {authors}\n"
    chunk += f"Year: {year}\n"
    chunk += f"Abstract: {abstract}\n"
    chunk += f"(paper: {paper_id})"

    docs.append(Document(page_content=chunk))

print(f"Created {len(docs)} document chunks ({skipped} skipped)")
print(f"\nExample chunk:")
print(docs[0].page_content[:300])
print("...")

# ─────────────────────────────────────────────
# STEP 3: Load embedding model
# ─────────────────────────────────────────────

print("\nLoading MiniLM embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embedding model ready")

# ─────────────────────────────────────────────
# STEP 4: Embed and store in Chroma
# ─────────────────────────────────────────────

print(f"\nEmbedding {len(docs)} paper chunks...")
print("This may take a few minutes...")

db = Chroma.from_documents(docs, embeddings, persist_directory=db_dir)

print(f"\nDone! Database saved to: {db_dir}")
print(f"Total papers stored: {len(docs)}")
