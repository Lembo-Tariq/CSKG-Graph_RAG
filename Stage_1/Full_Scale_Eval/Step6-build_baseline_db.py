"""
step6_build_baseline_db.py
==========================
Builds Chroma DB from paper abstracts for sampled papers.
"""

import os
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

NUM_SAMPLES = 2000  # must match step2

current_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(current_dir, "Step3Out-papers_" + str(NUM_SAMPLES) + "_abstracts.json")
db_dir     = os.path.join(current_dir, "db", "baseline_db_" + str(NUM_SAMPLES))

print("Loading papers from: " + input_file)
with open(input_file, "r", encoding="utf-8") as f:
    papers = json.load(f)

valid_papers = [
    p for p in papers
    if p.get("abstract") and p["abstract"] != "Abstract not available"
    and p.get("title") and p["title"] != "Error fetching"
    and p["title"] != "No title"
]

print("Valid papers: " + str(len(valid_papers)))

docs = []
for paper in valid_papers:
    title    = str(paper.get("title", ""))
    authors  = ", ".join(paper.get("authors", []))
    year     = str(paper.get("year", "N/A"))
    abstract = str(paper.get("abstract", ""))
    paper_id = paper.get("openalex_id", "")

    chunk  = "Title: " + title + "\n"
    chunk += "Authors: " + authors + "\n"
    chunk += "Year: " + year + "\n"
    chunk += "Abstract: " + abstract + "\n"
    chunk += "(paper: " + paper_id + ")"

    docs.append(Document(page_content=chunk))

print("Created " + str(len(docs)) + " chunks")

print("Loading MiniLM embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embedding model ready")

print("Building Baseline DB...")
db = Chroma.from_documents(docs, embeddings, persist_directory=db_dir)

print("Done! Saved to: " + db_dir)
print("Total papers stored: " + str(len(docs)))