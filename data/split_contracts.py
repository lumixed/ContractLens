
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
CLEANED_FILE = SCRIPT_DIR / "cleaned" / "all_contracts_normalized.json"
MANIFEST_PATH = SCRIPT_DIR / "split_manifest.json"

# Config
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO = 0.15 (implicit remainder)

# The 14 selected categories
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


def split_contracts():
    """Split contracts into train/val/test sets."""
    print(f"Loading cleaned data from {CLEANED_FILE}...")
    with open(CLEANED_FILE) as f:
        contracts = json.load(f)

    contract_ids = [c["contract_id"] for c in contracts]
    n = len(contract_ids)
    print(f"Total contracts: {n}")

    # Shuffle with fixed seed
    random.seed(RANDOM_SEED)
    indices = list(range(n))
    random.shuffle(indices)

    # Compute split boundaries
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    # n_test = remainder

    train_indices = set(indices[:n_train])
    val_indices = set(indices[n_train:n_train + n_val])
    test_indices = set(indices[n_train + n_val:])

    # Build manifest
    manifest = {}
    for i, cid in enumerate(contract_ids):
        if i in train_indices:
            manifest[cid] = "train"
        elif i in val_indices:
            manifest[cid] = "val"
        else:
            manifest[cid] = "test"

    # Save manifest
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    # ── Verification ────────────────────────────────────────────────────
    split_counts = Counter(manifest.values())
    print(f"\nSplit sizes:")
    print(f"  Train: {split_counts['train']} ({split_counts['train']/n*100:.1f}%)")
    print(f"  Val:   {split_counts['val']} ({split_counts['val']/n*100:.1f}%)")
    print(f"  Test:  {split_counts['test']} ({split_counts['test']/n*100:.1f}%)")

    # Leakage check: no contract in multiple splits
    train_ids = {cid for cid, s in manifest.items() if s == "train"}
    val_ids = {cid for cid, s in manifest.items() if s == "val"}
    test_ids = {cid for cid, s in manifest.items() if s == "test"}

    assert len(train_ids & val_ids) == 0, "LEAKAGE: train ∩ val"
    assert len(train_ids & test_ids) == 0, "LEAKAGE: train ∩ test"
    assert len(val_ids & test_ids) == 0, "LEAKAGE: val ∩ test"
    assert len(train_ids) + len(val_ids) + len(test_ids) == n, "Missing contracts"
    print("\n✓ No leakage: all contract IDs appear in exactly one split.")

    # Category distribution per split
    split_category_counts = defaultdict(lambda: defaultdict(int))
    for contract in contracts:
        split = manifest[contract["contract_id"]]
        for clause in contract["clauses"]:
            split_category_counts[split][clause["category"]] += 1

    print(f"\n{'Category':<40} {'Train':>7} {'Val':>7} {'Test':>7} {'Total':>7}")
    print(f"{'-'*40} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for cat in SELECTED_CATEGORIES:
        tr = split_category_counts["train"].get(cat, 0)
        va = split_category_counts["val"].get(cat, 0)
        te = split_category_counts["test"].get(cat, 0)
        print(f"{cat:<40} {tr:>7} {va:>7} {te:>7} {tr+va+te:>7}")

    totals = {s: sum(split_category_counts[s].values()) for s in ["train", "val", "test"]}
    print(f"{'TOTAL':<40} {totals['train']:>7} {totals['val']:>7} {totals['test']:>7} {sum(totals.values()):>7}")

    # Show a few sample contract IDs per split
    print(f"\nSample train IDs: {sorted(train_ids)[:3]}")
    print(f"Sample val IDs:   {sorted(val_ids)[:3]}")
    print(f"Sample test IDs:  {sorted(test_ids)[:3]}")

    print(f"\nManifest saved to: {MANIFEST_PATH}")
    return manifest


if __name__ == "__main__":
    split_contracts()
