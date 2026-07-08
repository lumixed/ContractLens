
import json
import random
from pathlib import Path
from collections import Counter

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
CLEANED_FILE = SCRIPT_DIR / "cleaned" / "all_contracts_normalized.json"
MANIFEST_PATH = SCRIPT_DIR / "split_manifest.json"

TRAIN_OUT = SCRIPT_DIR / "train.jsonl"
VAL_OUT = SCRIPT_DIR / "val.jsonl"
TEST_OUT = SCRIPT_DIR / "test.jsonl"

RANDOM_SEED = 42
NONE_SUBSAMPLE_RATIO = 0.15 # Keep 15% of the "None" paragraphs

SELECTED_CATEGORIES = [
    "Governing Law",
    "Anti-Assignment",
    "Cap On Liability",
    "License Grant",
    "Audit Rights",
    "Termination For Convenience",
    "Exclusivity",
    "Renewal Term",
    "Insurance",
    "Ip Ownership Assignment",
    "Change Of Control",
    "Non-Compete",
    "Uncapped Liability",
    "Revenue/Profit Sharing",
]

INSTRUCTION_TEXT = (
    "Classify the following contract clause into one of the following categories:\n"
    + ", ".join(f"'{c}'" for c in SELECTED_CATEGORIES)
    + ", or 'None' if it does not match any of these categories."
)

def chunk_into_paragraphs(text: str):
    """Yields (start_char, end_char, paragraph_text)."""
    # Find all \n\n to split paragraphs, keeping track of indices
    start = 0
    for chunk in text.split("\n\n"):
        chunk = chunk.strip()
        end = start + len(chunk)
        if chunk: # ignore empty chunks
            # Find the actual start in the original text (accounting for split stripping)
            actual_start = text.find(chunk, start)
            if actual_start != -1:
                yield actual_start, actual_start + len(chunk), chunk
        start = end + 2 # +2 for \n\n


def format_example(paragraph: str, label: str) -> dict:
    return {
        "instruction": INSTRUCTION_TEXT,
        "input": paragraph,
        "output": label
    }


def build_datasets():
    print(f"Loading cleaned data from {CLEANED_FILE}...")
    with open(CLEANED_FILE) as f:
        contracts = json.load(f)

    print(f"Loading manifest from {MANIFEST_PATH}...")
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    random.seed(RANDOM_SEED)

    splits = {"train": [], "val": [], "test": []}
    class_counts = {"train": Counter(), "val": Counter(), "test": Counter()}

    for contract in contracts:
        cid = contract["contract_id"]
        split = manifest.get(cid)
        if not split:
            continue

        text = contract["contract_text"]
        clauses = contract["clauses"]

        for p_start, p_end, p_text in chunk_into_paragraphs(text):
            # Too short paragraphs are usually section headers or page numbers
            if len(p_text) < 50:
                continue

            # Find clauses that overlap with this paragraph
            overlapping_clauses = []
            for clause in clauses:
                c_start, c_end = clause["start_char"], clause["end_char"]
                # Overlap logic: max(0, min(e1, e2) - max(s1, s2)) > 0
                overlap = max(0, min(p_end, c_end) - max(p_start, c_start))
                if overlap > 0:
                    overlapping_clauses.append((overlap, clause["category"]))

            if overlapping_clauses:
                # Pick the category with the largest overlap in this paragraph
                overlapping_clauses.sort(reverse=True)
                best_category = overlapping_clauses[0][1]
                example = format_example(p_text, best_category)
                splits[split].append(example)
                class_counts[split][best_category] += 1
            else:
                # No clause -> "None" class
                if split != "train" or random.random() < NONE_SUBSAMPLE_RATIO:
                    example = format_example(p_text, "None")
                    splits[split].append(example)
                    class_counts[split]["None"] += 1

    # Save to JSONL
    for split_name, out_path in [("train", TRAIN_OUT), ("val", VAL_OUT), ("test", TEST_OUT)]:
        with open(out_path, "w") as f:
            for ex in splits[split_name]:
                f.write(json.dumps(ex) + "\n")
        print(f"\n{split_name.upper()} set saved to {out_path} ({len(splits[split_name])} examples)")
        print("Class distribution:")
        for cls, count in class_counts[split_name].most_common():
            print(f"  {cls:<30} {count:>6} ({count/len(splits[split_name])*100:.1f}%)")

    # Show some examples
    print("\n" + "="*50)
    print("SAMPLE EXAMPLES (Train)")
    print("="*50)
    
    # Grab one None and one positive
    train_exs = splits["train"]
    pos_ex = next(ex for ex in train_exs if ex["output"] != "None")
    neg_ex = next(ex for ex in train_exs if ex["output"] == "None")

    for name, ex in [("Positive", pos_ex), ("Negative", neg_ex)]:
        print(f"\n--- {name} Example ---")
        print(f"INSTRUCTION: {ex['instruction'][:100]}...")
        print(f"INPUT (preview): {repr(ex['input'][:150])}...")
        print(f"OUTPUT: {ex['output']}")

if __name__ == "__main__":
    build_datasets()
