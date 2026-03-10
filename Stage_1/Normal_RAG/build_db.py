"""
build_db.py
===========
Step 1 of the RAG pipeline — run this ONCE.
Loads the CSKG text file, splits by line (one fact = one chunk), 
prior to normal chunking size with numbers,
embeds them, and saves to a local Chroma vector database.
"""

import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from dotenv import load_dotenv
import time

load_dotenv()

# ─────────────────────────────────────────────
# STEP 1: Define paths
# ─────────────────────────────────────────────

text_file = r"C:\Users\GIGABYITE\Desktop\RA_CSKG-GraphRAG\Stage_1\RA_Parsing_CSKG-Text\cskg_text_5000.txt"

current_dir = os.path.dirname(os.path.abspath(__file__))
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
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
print("✅ Embedding model ready")

# # ─────────────────────────────────────────────
# # STEP 4: Embed and store in Chroma
# # ─────────────────────────────────────────────
# # Each fact gets converted to a vector and stored in the DB
# # Chroma saves everything to disk in the db/ folder

# print(f"⏳ Embedding {len(docs)} facts and building Chroma DB...")
# print("   This may take a few minutes...")

# db = Chroma.from_documents(docs, embeddings, persist_directory=db_dir)

# print(f"\n✨ Done! Database saved to: {db_dir}")
# print(f"📊 Total facts stored: {len(docs)}")

# We can't use this as there is a limit for the free google embedder so we will import time and do it in batches it might take up to 50 minutes for 5000 lines so we will reduce it to 1000



# ─────────────────────────────────────────────
# STEP 4: Embed in batches to respect rate limits
# ─────────────────────────────────────────────
# Google free tier = 100 requests/minute
# We process in batches of 50, pausing 30 seconds between batches

BATCH_SIZE = 50
all_texts = [doc.page_content for doc in docs]
all_batches = [all_texts[i:i+BATCH_SIZE] for i in range(0, len(all_texts), BATCH_SIZE)]

print(f"📦 Total batches: {len(all_batches)} (batch size: {BATCH_SIZE})")

# Initialize Chroma DB
db = Chroma(persist_directory=db_dir, embedding_function=embeddings)

for i, batch in enumerate(all_batches):
    print(f"⏳ Embedding batch {i+1}/{len(all_batches)}...")
    db.add_texts(batch)
    
    # Pause every batch to avoid hitting rate limit
    if i < len(all_batches) - 1:
        print(f"   💤 Waiting 30 seconds to respect rate limit...")
        time.sleep(30)

print(f"\n✨ Done! Database saved to: {db_dir}")
print(f"📊 Total facts stored: {len(docs)}")