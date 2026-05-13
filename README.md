# CSKG Graph-RAG Research Project

## Overview
This repository contains the implementation and evaluation of RAG (Retrieval-Augmented Generation) systems applied to the **Computer Science Knowledge Graph (CSKG 2.0)**. The research goal is to compare Normal RAG vs Graph-RAG when querying a structured CS knowledge graph.

**Supervisor:** Dr. Danilo Dessì  
**Research Assistant:** Tariq Ribhi Yaseen

---

## Research Question
> Can Graph-RAG outperform Normal RAG when applied to the CSKG? And how do generic knowledge graph triples compare to enriched paper abstracts for paper-level retrieval?

---

## Repository Structure
RA_CSKG-GraphRAG/
│
├── Stage_1/                        ← Normal RAG baseline (COMPLETE)
│   ├── RA_Parsing_CSKG-Text/       ← Step 1: Parse raw CSKG data
│   ├── Normal_RAG/                 ← Step 2: Build & query triple-based RAG
│   ├── App_Demo/                   ← Step 3: Streamlit web app
│   ├── OpenAlex_Enrichment/        ← Step 4: Early evaluation pipeline
│   ├── baseline_RAG/               ← Step 5: Abstract-based RAG & full evaluation
│   └── master_results_table.csv    ← All experiment results in one table
│

---

## Key Results

### Baseline RAG (Paper Abstracts) vs Triple RAG (CSKG Facts)
| K | Baseline RAG (Abstracts) | Triple RAG (CSKG) | Difference |
|---|---|---|---|
| 1 | 0.907 | 0.470 | +0.437 |
| 3 | 0.962 | 0.597 | +0.365 |
| 5 | 0.970 | 0.640 | +0.330 |
| 10 | 0.971 | 0.683 | +0.288 |
| 20 | 0.974 | 0.725 | +0.249 |

**Metric:** Hit Rate@K — did the correct paper appear in the top K results?

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/Lembo-Tariq/CSKG-Graph_RAG.git
cd CSKG-Graph_RAG
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Create .env file for API Keys

---

## How to Run

### Build the Triple RAG database
```bash
python Stage_1/Normal_RAG/build_db.py
```

### Query the Triple RAG interactively
```bash
python Stage_1/Normal_RAG/interactive_query_rag.py
```

### Build the Baseline RAG database
```bash
python Stage_1/baseline_RAG/build_baseline_db.py
```

### Run full evaluation (734 papers)
```bash
python Stage_1/baseline_RAG/evaluate_baseline_full.py
python Stage_1/baseline_RAG/evaluate_triple_full.py
```
---

## Branches
| Branch | Purpose |
|---|---|
| `master` | Stable, production-ready code |
| `evaluation_v2` | New evaluation pipeline with W ID matching |
| `HuggingFace-embedding-model` | Switch from Google to MiniLM embeddings |
| `openalex` | OpenAlex API integration |
