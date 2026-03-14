"""
CSKG (Computer Science Knowledge Graph) Parser
================================================
Converts SPARQL query results (JSON format) into human-readable natural language text.
Designed as Step 1 for Graph-RAG pipeline ingestion.

Usage:
    python parse_cskg.py --input output.json --limit 1000 --output cskg_text.txt
"""

import json
from urllib.parse import unquote
import os


# ─────────────────────────────────────────────
# STEP 1: Helper — extract the "slug" from a URI
# ─────────────────────────────────────────────
# Every node in the CSKG is a URI like:
#   https://w3id.org/cskg/resource/image_classification
#   https://w3id.org/cskg/ontology#usesMethod
#   http://www.w3.org/2004/02/skos/core#broader
# We only want the human-readable tail end: "image_classification", "usesMethod", "broader"

def extract_label(uri: str) -> str:
    """
    Strips the URI prefix and returns just the concept label.
    Underscores are replaced with spaces for readability.
    """
    # Split on '#' first (used by ontology predicates like ontology#usesMethod)
    if "#" in uri:
        label = uri.split("#")[-1]
    else:
        # Split on '/' for resource URIs like resource/image_classification
        label = uri.split("/")[-1]
    #[-1] with the split function to return the last element of the array that the split function creates
    
    # Replace underscores with spaces → "image classification"
    return label.replace("_", " ")


# ─────────────────────────────────────────────
# STEP 2: Map predicate labels to natural language verbs
# ─────────────────────────────────────────────
# The CSKG ontology has ~15 relationship types. We map each to a clear English phrase
# so the output reads naturally for an LLM to understand.

PREDICATE_MAP = {
    "analyzesTask":        "analyzes the task of",
    "usesMethod":          "uses the method",
    "usesTask":            "uses the task of",
    "usesMaterial":        "uses the material",
    "producesMethod":      "produces the method",
    "executesMethod":      "executes the method",
    "matchesMethod":       "matches the method",
    "based-onMethod":      "is based on the method",
    "based-onTask":        "is based on the task of",
    "broader":             "is a broader concept than",   
    "narrower":            "is a narrower concept than",
    "related":             "is related to",
    "exactMatch":          "exactly matches",
    "closeMatch":          "closely matches",
    "hasMethod":           "has the method",
    "hasTask":             "has the task of",
    "hasMaterial":         "has the material",
}

def predicate_to_text(pred_label: str) -> str:
    """
    Returns the natural language form of a predicate.
    Falls back to the raw label if not in our map.
    """
    return PREDICATE_MAP.get(pred_label, pred_label)


# ─────────────────────────────────────────────
# STEP 3: Convert a single triple into a sentence
# ─────────────────────────────────────────────
# Each SPARQL result "binding" has 5 fields:
#   s       → subject   (a CS concept, e.g. image_classification)
#   p       → predicate (a relationship, e.g. usesMethod)
#   o       → object    (another CS concept, e.g. neural_network)
#   doi     → a paper DOI that supports this relationship
#   support → how many papers support this relationship (confidence score)

def binding_to_sentence(binding: dict) -> str:
    """
    Transforms one SPARQL result row into a natural language sentence.
    
    Example output:
    "image classification uses the method neural network 
     (supported by 11 papers, source: https://doi.org/10.5958/...)"
    """
    subject   = extract_label(binding["s"]["value"])
    predicate = predicate_to_text(extract_label(binding["p"]["value"]))
    obj       = extract_label(binding["o"]["value"])
    
    # DOI is URL-encoded in the data (e.g. "https%3A//doi.org/...")
    # unquote() converts it back to a normal URL
    doi_raw   = binding["doi"]["value"]
    doi       = unquote(doi_raw)
    
    support   = binding["support"]["value"]  # number of papers
    
    sentence = (
        f"{subject} {predicate} {obj} "
        f"(supported by {support} papers, source: {doi})"
    )
    return sentence


# ─────────────────────────────────────────────
# STEP 4: Main parser — load JSON and process N lines
# ─────────────────────────────────────────────

def parse_cskg(input_path: str, limit: int, output_path: str):
    """
    Reads the SPARQL JSON output, converts up to `limit` triples
    to natural language sentences, and writes them to a text file.
    """

    print(f"📂 Loading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # The JSON from your curl command has a wrapper from the server:
    # { "returncode": 0, "stdout": "... actual JSON ...", "stderr": "" }
    # So we need to parse the "stdout" field as JSON again.
    if "stdout" in raw:
        print("   ↳ Detected server wrapper — extracting inner SPARQL JSON...")
        inner_json_str = raw["stdout"]
        data = json.loads(inner_json_str)
    else:
        # If the file is already plain SPARQL JSON, use it directly
        data = raw

    bindings = data["results"]["bindings"]
    total_available = len(bindings)
    print(f"✅ Found {total_available:,} total triples in dataset.")

    # Apply the limit — this is where you control dataset size
    # Change --limit 1000 to any number when running from command line
    selected = bindings[:limit]
    print(f"🔢 Processing {len(selected):,} triples (limit = {limit})...")

    sentences = []
    for i, binding in enumerate(selected):
        try:
            sentence = binding_to_sentence(binding)
            sentences.append(sentence)
        except KeyError as e:
            # Skip any malformed rows gracefully
            print(f"   ⚠️  Skipping row {i} — missing field: {e}")

    # ─────────────────────────────────────────────
    # STEP 5: Write output
    # ─────────────────────────────────────────────
    # We write one sentence per line. This is the simplest format for:
    #   - Direct LLM ingestion
    #   - Chunking for vector embeddings (each line = one chunk)
    #   - Further processing (grouping by subject, etc.)

    with open(output_path, "w", encoding="utf-8") as f:
        for sentence in sentences:
            f.write(sentence + "\n")

    print(f"\n✨ Done! {len(sentences):,} sentences written to: {output_path}")
    print(f"\n📄 Preview (first 5 lines):")
    for s in sentences[:5]:
        print(f"   {s}")


# ─────────────────────────────────────────────
# Entry point with command-line arguments
# ─────────────────────────────────────────────

if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))

    parse_cskg(
        input_path= os.path.join(current_dir,"output.json"),
        limit=10000,
        output_path= os.path.join(current_dir,"cskg_text_10k.txt")
    )
