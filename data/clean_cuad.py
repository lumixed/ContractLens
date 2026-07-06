
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_FILE = SCRIPT_DIR / "raw" / "CUAD_v1.json"
CLEANED_DIR = SCRIPT_DIR / "cleaned"
CLEANED_DIR.mkdir(exist_ok=True)

# ── The 14 confirmed categories + None ──────────────────────────────────────
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


def extract_category(question: str) -> str:
    """Extract category name from CUAD question text."""
    match = re.search(r"related to ['\"]([^'\"]+)['\"]", question)
    if match:
        return match.group(1)
    match = re.search(r"related to (.+?)(?:\.|,|\s+that)", question)
    if match:
        return match.group(1).strip().strip("'\"")
    return question[:80]


def clean_text(raw_text: str) -> tuple[str, list[tuple[int, int]]]:
    """
    Clean contract text and build an offset mapping.

    Returns:
        cleaned_text: the cleaned string
        char_map: list of (old_pos, new_pos) for every character in the
                  cleaned text, allowing offset recomputation.
                  Specifically, char_map[new_pos] = old_pos.
    """
    # Build a character-level mapping from old positions to new positions.
    # We process the text character by character, applying transforms,
    # and track where each surviving character came from.

    # Strategy: apply regex substitutions sequentially, tracking offset
    # shifts. This is simpler and more maintainable than char-by-char.

    # We'll use an incremental approach: apply each transform, rebuild
    # the mapping from original positions to current positions.

    # Start with identity mapping
    # current_text[i] came from original position old_positions[i]
    old_positions = list(range(len(raw_text)))
    current_text = raw_text

    def apply_sub(text, positions, pattern, replacement, flags=0):
        """Apply regex substitution and update position tracking."""
        new_text_parts = []
        new_positions = []
        last_end = 0

        for m in re.finditer(pattern, text, flags=flags):
            # Copy unchanged region before match
            new_text_parts.append(text[last_end:m.start()])
            new_positions.extend(positions[last_end:m.start()])

            # Add replacement — map all replacement chars to the start
            # of the original match (best-effort for offset recomputation)
            new_text_parts.append(replacement)
            for j in range(len(replacement)):
                if j < len(positions[m.start():m.end()]):
                    new_positions.append(positions[m.start() + j])
                else:
                    # Replacement is longer than match (shouldn't happen
                    # with our transforms, but handle gracefully)
                    new_positions.append(positions[m.start()])

            last_end = m.end()

        # Copy remainder
        new_text_parts.append(text[last_end:])
        new_positions.extend(positions[last_end:])

        return "".join(new_text_parts), new_positions

    # 1. Strip leading exhibit header: "Exhibit 10.6\n\n" or similar
    #    Pattern: starts with optional digits/spaces, then "Exhibit" + number,
    #    followed by whitespace. Case-insensitive to catch EXHIBIT, Exhibit, etc.
    current_text, old_positions = apply_sub(
        current_text, old_positions,
        r"^\d*\s*exhibit\s+[\d.A-Za-z-]+\s*",
        "",
        flags=re.IGNORECASE,
    )

    # 2. Collapse runs of 2+ spaces to single space
    current_text, old_positions = apply_sub(
        current_text, old_positions,
        r" {2,}",
        " "
    )

    # 3. Strip trailing whitespace on each line
    current_text, old_positions = apply_sub(
        current_text, old_positions,
        r"[ \t]+\n",
        "\n"
    )

    # 4. Collapse 3+ newlines to 2 newlines (preserve paragraph breaks)
    current_text, old_positions = apply_sub(
        current_text, old_positions,
        r"\n{3,}",
        "\n\n"
    )

    # 5. Strip leading/trailing whitespace from entire document
    stripped = current_text.strip()
    start_offset = len(current_text) - len(current_text.lstrip())
    old_positions = old_positions[start_offset:start_offset + len(stripped)]
    current_text = stripped

    return current_text, old_positions


def recompute_offset(old_start: int, old_end: int, old_positions: list[int]) -> tuple[int, int]:
    """
    Given an original (old_start, old_end) span and the old_positions mapping,
    find the new (start, end) in the cleaned text.

    old_positions[new_pos] = old_pos, so we need to find new_pos where
    old_positions[new_pos] == old_start (or closest).
    """
    # Build reverse lookup: old_pos -> new_pos (first occurrence)
    # This is O(n) but we only do it once per contract.

    # Find new_start: first new_pos where old_positions[new_pos] >= old_start
    new_start = None
    new_end = None

    for new_pos, old_pos in enumerate(old_positions):
        if new_start is None and old_pos >= old_start:
            new_start = new_pos
        if old_pos < old_end:
            new_end = new_pos + 1  # exclusive end

    if new_start is None or new_end is None:
        return -1, -1

    return new_start, new_end


def clean_and_normalize():
    """Main cleaning pipeline."""
    print(f"Loading raw data from {RAW_FILE}...")
    with open(RAW_FILE) as f:
        raw_data = json.load(f)

    documents = raw_data["data"]
    print(f"Processing {len(documents)} contracts...")

    normalized = []
    stats = {
        "total_contracts": 0,
        "contracts_with_clauses": 0,
        "total_clauses": 0,
        "offset_mismatches": 0,
        "categories_found": defaultdict(int),
    }

    for doc in documents:
        title = doc["title"]
        context = doc["paragraphs"][0]["context"]
        qas = doc["paragraphs"][0]["qas"]

        # Clean text
        cleaned_text, old_positions = clean_text(context)

        # Extract and recompute annotations for selected categories
        clauses = []
        for qa in qas:
            category = extract_category(qa["question"])
            if category not in SELECTED_CATEGORIES:
                continue

            if qa.get("is_impossible", False):
                continue

            for answer in qa.get("answers", []):
                old_start = answer["answer_start"]
                old_text = answer["text"]
                old_end = old_start + len(old_text)

                # Recompute offset in cleaned text
                new_start, new_end = recompute_offset(old_start, old_end, old_positions)

                if new_start == -1:
                    stats["offset_mismatches"] += 1
                    continue

                # Verify the text matches
                extracted = cleaned_text[new_start:new_end]

                # Allow minor whitespace differences from cleaning
                if extracted.replace(" ", "").replace("\n", "") != old_text.replace(" ", "").replace("\n", ""):
                    stats["offset_mismatches"] += 1
                    # Try to still use the span if it's close
                    continue

                clauses.append({
                    "category": category,
                    "start_char": new_start,
                    "end_char": new_end,
                    "text": extracted,
                })
                stats["categories_found"][category] += 1

        stats["total_contracts"] += 1
        if clauses:
            stats["contracts_with_clauses"] += 1
        stats["total_clauses"] += len(clauses)

        normalized.append({
            "contract_id": title,
            "contract_text": cleaned_text,
            "clauses": clauses,
        })

    # Save all normalized contracts
    output_path = CLEANED_DIR / "all_contracts_normalized.json"
    with open(output_path, "w") as f:
        json.dump(normalized, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print(f"CLEANING SUMMARY")
    print(f"{'='*70}")
    print(f"Total contracts processed:     {stats['total_contracts']}")
    print(f"Contracts with ≥1 clause:      {stats['contracts_with_clauses']}")
    print(f"Total clause annotations:      {stats['total_clauses']}")
    print(f"Offset mismatches (dropped):   {stats['offset_mismatches']}")

    print(f"\nAnnotations per category:")
    for cat in SELECTED_CATEGORIES:
        count = stats["categories_found"].get(cat, 0)
        print(f"  {cat:<40} {count:>5}")

    print(f"\nNormalized data saved to: {output_path}")

    # Show before/after examples for 3 contracts
    print(f"\n{'='*70}")
    print(f"BEFORE/AFTER CLEANING EXAMPLES")
    print(f"{'='*70}")

    for i in [0, 50, 200]:  # sample from different parts of the dataset
        if i >= len(documents):
            continue
        doc = documents[i]
        norm = normalized[i]
        raw_ctx = doc["paragraphs"][0]["context"]

        print(f"\n--- Contract {i}: {norm['contract_id'][:60]} ---")
        print(f"  Raw length:     {len(raw_ctx):,} chars")
        print(f"  Cleaned length: {len(norm['contract_text']):,} chars")
        print(f"  Reduction:      {(1 - len(norm['contract_text'])/len(raw_ctx))*100:.1f}%")
        print(f"  Clauses found:  {len(norm['clauses'])}")

        print(f"\n  BEFORE (first 200 chars):")
        print(f"  {repr(raw_ctx[:200])}")
        print(f"\n  AFTER (first 200 chars):")
        print(f"  {repr(norm['contract_text'][:200])}")

        # Show a sample clause with offset verification
        if norm["clauses"]:
            clause = norm["clauses"][0]
            print(f"\n  Sample clause: [{clause['category']}]")
            print(f"    Offset: {clause['start_char']}:{clause['end_char']}")
            print(f"    Text: {repr(clause['text'][:150])}")
            # Verify offset
            extracted = norm["contract_text"][clause["start_char"]:clause["end_char"]]
            print(f"    Verify: {repr(extracted[:150])}")
            print(f"    Match: {extracted == clause['text']}")

    # Save stats
    stats_clean = {k: v for k, v in stats.items() if k != "categories_found"}
    stats_clean["categories_found"] = dict(stats["categories_found"])
    stats_path = CLEANED_DIR / "cleaning_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats_clean, f, indent=2)

    return normalized, stats


if __name__ == "__main__":
    clean_and_normalize()
