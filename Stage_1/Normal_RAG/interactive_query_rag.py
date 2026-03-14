"""
interactive_query_RAG.py
========================
Interactive version of the RAG pipeline.
Type your question in the terminal and get an answer.
Type 'exit' or 'quit' to stop.
"""

import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# ─────────────────────────────────────────────
# STEP 1: Load the Chroma DB once at startup
# ─────────────────────────────────────────────
# We load this ONCE before the loop starts
# so we don't reload it on every question

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db", "chroma_db")

print("📂 Loading Chroma DB...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=db_dir, embedding_function=embeddings)
print("✅ DB loaded")

# ─────────────────────────────────────────────
# STEP 2: Load the LLM once at startup
# ─────────────────────────────────────────────

model = ChatGroq(model="llama-3.1-8b-instant")

# ─────────────────────────────────────────────
# STEP 3: Set up the retriever
# ─────────────────────────────────────────────

retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)

# ─────────────────────────────────────────────
# STEP 4: Interactive loop
# ─────────────────────────────────────────────
# Keeps asking for questions until user types exit/quit

print("\n🤖 CSKG RAG Assistant ready!")
print("   Ask any question about Computer Science concepts.")
print("   Type 'exit' or 'quit' to stop.\n")

while True:
    # Get question from user
    query = input("❓ Your question: ").strip()

    # Exit condition
    if query.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    # Skip empty input
    if not query:
        print("   ⚠️  Please type a question.\n")
        continue

    # Retrieve relevant facts
    print(f"\n🔍 Searching knowledge graph...")
    relevant_docs = retriever.invoke(query)

    print(f"\n--- {len(relevant_docs)} Relevant Facts Retrieved ---")
    for i, doc in enumerate(relevant_docs):
        print(f"  {i+1}. {doc.page_content}")

    # Build prompt
    context = "\n".join([doc.page_content for doc in relevant_docs])
    combined_input = f"""Use the following facts from a Computer Science Knowledge Graph to answer the question.
Only use the information provided — do not make anything up.

Facts:
{context}

Question: {query}
Answer:"""

    # Get answer from LLM
    print(f"\n🤖 Thinking...")
    result = model.invoke([
        SystemMessage(content="You are a helpful Computer Science assistant. Answer using only the provided facts. Be concise and clear."),
        HumanMessage(content=combined_input)
    ])

    print(f"\n--- Answer ---")
    print(result.content)
    print("\n" + "─"*50 + "\n")