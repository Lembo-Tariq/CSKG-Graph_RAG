# Normal_RAG

## Purpose
Builds and queries a vector database from CSKG triples using MiniLM embeddings.
Used in Experiments 10-14 (Triple RAG Full Evaluation).

## Files

| File | Purpose |
|---|---|
| `build_db.py` | Embeds all 10,000 facts from cskg_text_v2.txt into Chroma DB |
| `query_rag.py` | Single query interface — type a question, get an answer |
| `interactive_query_rag.py` | Interactive terminal loop for multiple queries |
| `test_models.py` | Tests available Google embedding models |

## How to Run

### Build the database (run once)
```bash
python Stage_1/Normal_RAG/build_db.py
```

### Query interactively
```bash
python Stage_1/Normal_RAG/interactive_query_rag.py
```

## Results (Experiments 10-14)
| K | Hit Rate@K | MRR |
|---|---|---|
| 1 | 0.470 | 0.470 |
| 3 | 0.597 | 0.526 |
| 5 | 0.640 | 0.536 |
| 10 | 0.683 | 0.541 |
| 20 | 0.725 | 0.544 |
