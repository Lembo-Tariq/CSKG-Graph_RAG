"""
step5_build_triple_db.py
========================
Builds Chroma DB from CSKG triples for sampled papers.
"""

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

NUM_SAMPLES = 2000  # must match step2

current_dir = os.path.dirname(os.path.abspath(__file__))
text_file = os.path.join(current_dir, "Step2Out-cskg_text_" + str(NUM_SAMPLES) + ".txt")
db_dir    = os.path.join(current_dir, "db", "triple_db_" + str(NUM_SAMPLES))

print("Loading triples from: " + text_file)
with open(text_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

docs = [
    Document(page_content=line.strip())
    for line in lines
    if line.strip()
]

print("Loaded " + str(len(docs)) + " facts")

print("Loading MiniLM embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embedding model ready")

print("Building Triple DB...")
db = Chroma.from_documents(docs, embeddings, persist_directory=db_dir)

print("Done! Saved to: " + db_dir)
print("Total facts stored: " + str(len(docs)))