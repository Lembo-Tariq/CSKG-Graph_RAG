# baseline_RAG

## Purpose
Implements and evaluates a RAG system using actual paper abstracts
instead of generic CSKG triples. Serves as the strong baseline
for comparison with future Graph-RAG implementation.

## Pipeline
fetch_abstracts.py          → papers_with_abstracts.json (887 papers)
↓
build_baseline_db.py        → Chroma DB (887 abstract chunks)
↓
sample_papers_full.py       → all_paper_samples.json (734 papers)
↓
generate_queries_full.py    → eval_dataset_full.json (734 queries)
↓
evaluate_baseline_full.py   → baseline_eval_full_results.csv/json
evaluate_triple_full.py     → triple_eval_full_results.csv/json

## Files

| File | Purpose | Experiment |
|---|---|---|
| `fetch_abstracts.py` | Fetches abstracts for 909 papers from OpenAlex | Setup |
| `build_baseline_db.py` | Builds Chroma DB from 887 paper abstracts | Exp 5-9 |
| `sample_papers_full.py` | Collects all 734 valid papers | Setup |
| `generate_queries_full.py` | Generates 734 specific queries via Groq | Setup |
| `sample_single_paper.py` | Samples 20 individual papers (small test) | Exp 4 |
| `generate_queries_single.py` | Generates 20 queries (small test) | Exp 4 |
| `evaluate_baseline_full.py` | Evaluates Baseline RAG at K=1,3,5,10,20 | Exp 5-9 |
| `evaluate_triple_full.py` | Evaluates Triple RAG at K=1,3,5,10,20 | Exp 10-14 |
| `evaluate_baseline_rag.py` | Old evaluation (Precision/Recall/F1) | Legacy |
| `evaluate_baseline_rag_v2.py` | New evaluation (Hit Rate + MRR, 20 papers) | Exp 4 |

## Results

### Baseline RAG (Abstracts) — Experiments 5-9
| K | Hit Rate@K | MRR |
|---|---|---|
| 1 | 0.907 | 0.907 |
| 3 | 0.962 | 0.933 |
| 5 | 0.970 | 0.935 |
| 10 | 0.971 | 0.935 |
| 20 | 0.974 | 0.935 |

### Triple RAG (CSKG Facts) — Experiments 10-14
| K | Hit Rate@K | MRR |
|---|---|---|
| 1 | 0.470 | 0.470 |
| 3 | 0.597 | 0.526 |
| 5 | 0.640 | 0.536 |
| 10 | 0.683 | 0.541 |
| 20 | 0.725 | 0.544 |
