"""
build_db.py
===========
Step 1 of the RAG pipeline — run this ONCE.
Loads the CSKG text file, splits by line (one fact = one chunk),
embeds them using MiniLM (local, free, no rate limits),
and saves to a local Chroma vector database.
"""

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
<<<<<<< HEAD
from langchain_core.documents import Document
=======
from langchain.schema import Document

>>>>>>> origin/baseline-rag

# ─────────────────────────────────────────────
# STEP 1: Define paths
# ─────────────────────────────────────────────

current_dir = os.path.dirname(os.path.abspath(__file__))
<<<<<<< HEAD
text_file = os.path.join(current_dir, "..", "RA_Parsing_CSKG-Text", "cskg_text_v2.txt")
=======
text_file = os.path.join(current_dir,"..","RA_Parsing_CSKG-Text","cskg_text_10k.txt")

>>>>>>> origin/baseline-rag
db_dir = os.path.join(current_dir, "db", "chroma_db")

print(f"📂 Loading text from: {text_file}")
print(f"💾 Database will be saved to: {db_dir}")

# ─────────────────────────────────────────────
# STEP 2: Load and split by line
# ─────────────────────────────────────────────

with open(text_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

docs = [
    Document(page_content=line.strip())
    for line in lines
    if line.strip()
]

<<<<<<< HEAD
=======

>>>>>>> origin/baseline-rag
print(f"✅ Loaded {len(docs)} facts (one per line)")

# ─────────────────────────────────────────────
# STEP 3: Load MiniLM embedding model
# ─────────────────────────────────────────────

<<<<<<< HEAD
print("⏳ Loading embedding model (may download ~80MB first time)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
=======
print("⏳ Loading embedding model")
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
>>>>>>> origin/baseline-rag
print("✅ Embedding model ready")

# ─────────────────────────────────────────────
# STEP 4: Embed and store in Chroma
# ─────────────────────────────────────────────
<<<<<<< HEAD
=======
# Each fact gets converted to a vector and stored in the DB
# Chroma saves everything to disk in the db/ folder
>>>>>>> origin/baseline-rag

print(f"⏳ Embedding {len(docs)} facts and building Chroma DB...")
print("   This may take a few minutes...")

db = Chroma.from_documents(docs, embeddings, persist_directory=db_dir)

print(f"\n✨ Done! Database saved to: {db_dir}")
print(f"📊 Total facts stored: {len(docs)}")


