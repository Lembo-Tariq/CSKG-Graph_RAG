"""
build_db.py
===========
Step 1 of the RAG pipeline — run this ONCE.
Loads the CSKG text file, splits by line (one fact = one chunk), 
prior to normal chunking size with numbers,
embeds them, and saves to a local Chroma vector database.
"""

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document


# ─────────────────────────────────────────────
# STEP 1: Define paths
# ─────────────────────────────────────────────

current_dir = os.path.dirname(os.path.abspath(__file__))
text_file = os.path.join(current_dir,"..","RA_Parsing_CSKG-Text","cskg_text_5000.txt")

db_dir = os.path.join(current_dir, "db", "chroma_db")

print(f"📂 Loading text from: {text_file}")
print(f"💾 Database will be saved to: {db_dir}")

# ─────────────────────────────────────────────
# STEP 2: Load and split by line
# ─────────────────────────────────────────────
# Each line is one complete fact — no need to chunk by character
# We just split on newline and wrap each line as a Document object
# which is what Chroma expects

with open(text_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Strip whitespace and skip any empty lines
docs = [
    Document(page_content=line.strip())
    for line in lines
    if line.strip()
]

# Slice to first 1000 for testing
docs = docs[:1000]

print(f"✅ Loaded {len(docs)} facts (one per line)")


# ─────────────────────────────────────────────
# STEP 3: Load the embedding model
# ─────────────────────────────────────────────
# Downloads automatically on first run (~80MB), then cached locally
# Completely free, runs on your machine, no API key needed

print("⏳ Loading embedding model")
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
print("✅ Embedding model ready")

# ─────────────────────────────────────────────
# STEP 4: Embed and store in Chroma
# ─────────────────────────────────────────────
# Each fact gets converted to a vector and stored in the DB
# Chroma saves everything to disk in the db/ folder

print(f"⏳ Embedding {len(docs)} facts and building Chroma DB...")
print("   This may take a few minutes...")

db = Chroma.from_documents(docs, embeddings, persist_directory=db_dir)

print(f"\n✨ Done! Database saved to: {db_dir}")
print(f"📊 Total facts stored: {len(docs)}")


