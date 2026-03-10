"""
query_rag.py
============
Step 2 of the RAG pipeline — run this every time you want to ask a question.
Loads the Chroma DB, retrieves relevant facts, sends to Groq LLaMA for an answer.
"""

import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# ─────────────────────────────────────────────
# STEP 1: Load the Chroma DB
# ─────────────────────────────────────────────
# We point to the same db/ folder that build_db.py created
# We must use the SAME embedding model used to build it

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db", "chroma_db")

print("📂 Loading Chroma DB...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
db = Chroma(persist_directory=db_dir, embedding_function=embeddings)
print("✅ DB loaded")

# ─────────────────────────────────────────────
# STEP 2: Define your query
# ─────────────────────────────────────────────

query = "what methods are used for image classification?"

# ─────────────────────────────────────────────
# STEP 3: Retrieve relevant facts
# ─────────────────────────────────────────────
# similarity search finds the top k most relevant facts
# from the DB based on the query's meaning

print(f"\n🔍 Searching for: '{query}'")
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
relevant_docs = retriever.invoke(query)

print(f"\n--- {len(relevant_docs)} Relevant Facts Retrieved ---")
for i, doc in enumerate(relevant_docs):
    print(f"  {i+1}. {doc.page_content}")

# ─────────────────────────────────────────────
# STEP 4: Build the prompt
# ─────────────────────────────────────────────
# We combine all retrieved facts as context
# and ask the LLM to answer based only on that context

context = "\n".join([doc.page_content for doc in relevant_docs])

combined_input = f"""Use the following facts from a Computer Science Knowledge Graph to answer the question.
Only use the information provided — do not make anything up.

Facts:
{context}

Question: {query}
Answer:"""

# ─────────────────────────────────────────────
# STEP 5: Send to Groq LLaMA and get answer
# ─────────────────────────────────────────────

print("\n🤖 Sending to Groq LLaMA...")
model = ChatGroq(model="llama-3.1-8b-instant")

result = model.invoke([
    SystemMessage(content="You are a helpful Artificial assistant. Answer using only the provided facts. Answering in a natural way as if it's a conversation"),
    HumanMessage(content=combined_input)
])

print("\n--- Answer ---")
print(result.content)