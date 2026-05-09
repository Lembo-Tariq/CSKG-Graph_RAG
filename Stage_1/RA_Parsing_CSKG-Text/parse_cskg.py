"""
parse_cskg_v2.py
================
Parser for the new CSKG SPARQL format from Dr. Danilo.
New format has 4 fields: paper, any_s, any_p, any_o
Key difference: includes paper ID (W number) directly in each fact.
This allows evaluation by paper ID instead of DOI.
"""

import os
import json
from urllib.parse import unquote
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# STEP 1: Load the new SPARQL JSON
# ─────────────────────────────────────────────

input_file = os.path.join(current_dir, "output_modified.json")
output_file = os.path.join(current_dir, "cskg_text_v2.txt")

print("📂 Loading SPARQL results...")
with open(input_file, "r", encoding="utf-8") as f:
    raw = json.load(f)

bindings = raw["results"]["bindings"]
print(f"✅ Loaded {len(bindings)} triples")

# ─────────────────────────────────────────────
# STEP 2: Helper functions
# ─────────────────────────────────────────────

def extract_label(uri: str) -> str:
    """
    Extracts human readable label from a URI.
    Handles both # and / separators.
    """
    if "#" in uri:
        label = uri.split("#")[-1]
    else:
        label = uri.split("/")[-1]
    return label.replace("_", " ")


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
    "analyzesMethod":      "analyzes the method",
    "includesMethod":      "includes the method",
    "proposesMethod":      "proposes the method",
    "improvesMethod":      "improves the method",
    "improvesMetric":      "improves the metric",
    "usesMetric":          "uses the metric",
    "usesOtherEntity":     "uses the entity",
    "producesOtherEntity": "produces the entity",
    "predictsTask":        "predicts the task of",
    "acquiresMethod":      "acquires the method",
    "learnsMethod":        "learns the method",
    "solvesMethod":        "solves using the method",
    "affectsMethod":       "affects the method",
    "providesMethod":      "provides the method",
    "improvesOtherEntity": "improves the entity",
}

def predicate_to_text(pred_label: str) -> str:
    return PREDICATE_MAP.get(pred_label, pred_label)

# ─────────────────────────────────────────────
# STEP 3: Convert each triple to a sentence
# ─────────────────────────────────────────────
# New format includes paper ID directly:
# "subject predicate object (paper: W1010415138)"
# This lets us match facts back to papers during evaluation

sentences = []
skipped = 0

for i, row in enumerate(bindings):
    try:
        # Extract paper W ID
        paper_uri = row["paper"]["value"]
        paper_id = paper_uri.split("/")[-1]  # → W1010415138

        # Extract subject, predicate, object
        subject   = extract_label(row["any_s"]["value"])
        predicate = predicate_to_text(extract_label(row["any_p"]["value"]))
        obj       = extract_label(row["any_o"]["value"])

        # Build sentence with paper ID embedded
        sentence = f"{subject} {predicate} {obj} (paper: {paper_id})"
        sentences.append(sentence)

    except KeyError as e:
        skipped += 1
        continue

print(f"✅ Converted {len(sentences)} triples ({skipped} skipped)")

# ─────────────────────────────────────────────
# STEP 4: Save to text file
# ─────────────────────────────────────────────

with open(output_file, "w", encoding="utf-8") as f:
    for sentence in sentences:
        f.write(sentence + "\n")

<<<<<<< HEAD
print(f"✨ Done! Saved to: {output_file}")
print(f"\n📄 Preview (first 5 lines):")
for s in sentences[:5]:
    print(f"   {s}")
=======
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
>>>>>>> origin/baseline-rag
