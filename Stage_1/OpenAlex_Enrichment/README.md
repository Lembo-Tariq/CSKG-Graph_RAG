# OpenAlex_Enrichment

## Purpose
Early evaluation pipeline — fetches paper metadata from OpenAlex,
generates synthetic queries, and evaluates retrieval quality.
Contains both the old (combined queries) and new (single paper) approaches.

## Files

| File | Purpose | Experiment |
|---|---|---|
| `fetch_openalex.py` | Fetches top 10 most cited papers from DB | Setup Exp 1,2 |
| `generate_queries.py` | Generates 3 queries per paper (old approach) | Exp 1,2 |
| `evaluate_rag.py` | Evaluates with DOI matching | Exp 1 |
| `evaluate_rag_v2.py` | Evaluates with paper ID matching | Exp 2,3 |
| `sample_papers.py` | Samples 20 groups of 5-10 papers | Setup Exp 3 |
| `generate_queries_v2.py` | Generates 1 query per group of papers | Exp 3 |
| `sample_single_papers.py` | Samples 20 individual papers | Setup Exp 4 |
| `generate_queries_single.py` | Generates 1 specific query per paper | Exp 4 |
| `test_gemini_model.py` | Tests available Gemini models | Utility |
| `small_paper.py` | Tests OpenAlex query for a specific paper | Utility |

## Key Finding
Combined multi-paper queries (Exp 1-3) produced near-zero results.
Single paper specific queries (Exp 4) produced Hit@20=0.95, MRR=0.90.
Query quality is as important as the retrieval system itself.
