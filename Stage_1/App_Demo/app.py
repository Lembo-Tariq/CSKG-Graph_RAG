"""
app.py
======
Streamlit UI for the CSKG RAG pipeline.
Run with: streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

load_dotenv()

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="CSKG RAG Assistant",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 CSKG RAG Assistant")
st.markdown("Ask questions about Computer Science concepts using the **Computer Science Knowledge Graph**.")
st.divider()

# ─────────────────────────────────────────────
# Load DB and model once using Streamlit cache
# ─────────────────────────────────────────────
# @st.cache_resource means this only runs ONCE
# even if the user asks multiple questions
# without it, the DB would reload on every query

@st.cache_resource
def load_db():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_dir = os.path.join(current_dir, "..", "Normal_RAG", "db", "chroma_db")
    text_file = os.path.join(current_dir, "..", "RA_Parsing_CSKG-Text", "cskg_text_10k.txt")
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # If DB doesn't exist yet, build it from the text file
    if not os.path.exists(db_dir):
        st.info("🔨 First launch — building knowledge base from scratch. This takes a few minutes...")
        
        with open(text_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        docs = [
            Document(page_content=line.strip())
            for line in lines
            if line.strip()
        ]
        
        db = Chroma.from_documents(docs, embeddings, persist_directory=db_dir)
        st.success("✅ Knowledge base built successfully!")
    else:
        db = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    
    return db

@st.cache_resource
def load_model():
    return ChatGroq(model="llama-3.1-8b-instant")

with st.spinner("Loading knowledge base..."):
    db = load_db()
    model = load_model()

# ─────────────────────────────────────────────
# Query input
# ─────────────────────────────────────────────

query = st.text_input(
    "❓ Your Question:",
    placeholder="e.g. what methods are used for image classification?"
)

num_results = st.slider("Number of facts to retrieve:", min_value=3, max_value=10, value=5)

search_button = st.button("🔍 Search", type="primary")

# ─────────────────────────────────────────────
# Run RAG pipeline on button click
# ─────────────────────────────────────────────

if search_button and query:

    # Retrieve facts
    with st.spinner("Searching knowledge graph..."):
        retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": num_results, "fetch_k": 20}
        )
        relevant_docs = retriever.invoke(query)

    # Show retrieved facts
    st.subheader("📄 Retrieved Facts")
    for i, doc in enumerate(relevant_docs):
        st.markdown(f"**{i+1}.** {doc.page_content}")

    st.divider()

    # Generate answer
    with st.spinner("Generating answer..."):
        context = "\n".join([doc.page_content for doc in relevant_docs])
        combined_input = f"""Use the following facts from a Computer Science Knowledge Graph to answer the question.
Only use the information provided — do not make anything up.

Facts:
{context}

Question: {query}
Answer:"""

        result = model.invoke([
            SystemMessage(content="You are a helpful Computer Science assistant. Answer using only the provided facts. Be concise and clear."),
            HumanMessage(content=combined_input)
        ])

    # Show answer
    st.subheader("🤖 Answer")
    st.markdown(result.content)

elif search_button and not query:
    st.warning("Please type a question first!")