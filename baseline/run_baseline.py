"""
Step 7 & 8: Frontier-model baseline for contract clause classification.

Uses the Anthropic Claude API (claude-3-5-haiku) with a 5-shot prompt.
Runs on the full test set and records per-category P/R/F1, macro-F1,
cost (tokens), and latency.

Usage:
    # Dry run on 5 examples (Step 7 - prompt review)
    python baseline/run_baseline.py --dry-run

    # Full test set evaluation (Step 8)
    python baseline/run_baseline.py

Requirements:
    pip install anthropic scikit-learn
    ANTHROPIC_API_KEY set in environment
"""

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
import anthropic
from sklearn.metrics import classification_report, f1_score

# Load API keys from .env if present
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TEST_FILE = REPO_ROOT / "data" / "test.jsonl"
RESULTS_FILE = SCRIPT_DIR / "baseline_results.json"

# ── Categories ─────────────────────────────────────────────────────────────
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
    "None",
]

CATEGORY_LIST_STR = "\n".join(f"  - {c}" for c in SELECTED_CATEGORIES)

# ── Few-shot examples ───────────────────────────────────────────────────────
# Selected from the TRAIN set — none of these contracts appear in test.
FEW_SHOT_EXAMPLES = [
    {
        "input": (
            "This Agreement was entered into in the State of Florida, and its validity, "
            "construction, interpretation, and legal effect shall be governed by the laws "
            "and judicial decisions of the State of Florida applicable to contracts entered "
            "into and performed entirely within the State of Florida."
        ),
        "output": "Governing Law",
    },
    {
        "input": (
            "Either party may terminate this Agreement without cause at any time effective "
            "upon thirty (30) days' written notice. Notwithstanding anything to the contrary "
            "contained in this Agreement, no termination of this Agreement for any reason "
            "whatsoever shall relieve the Customer of the obligation to pay all amounts due."
        ),
        "output": "Termination For Convenience",
    },
    {
        "input": (
            "i-on will not be liable under any circumstances for any lost profits or other "
            "consequential damages, even if i-on has been advised as to the possibility of "
            "such damages. i-on's liability for damages to the Customer for any cause "
            "whatsoever, regardless of the form of action, and whether in contract or in "
            "tort, shall be limited to the amounts paid by Customer to i-on during the "
            "twelve (12) month period prior to the event giving rise to the claim."
        ),
        "output": "Cap On Liability",
    },
    {
        "input": (
            "Subject to the terms and conditions of this Agreement, Company hereby grants "
            "to Distributor a non-exclusive, non-transferable license to use the Software "
            "solely for Distributor's internal business purposes in connection with the "
            "services provided under this Agreement."
        ),
        "output": "License Grant",
    },
    {
        "input": (
            "This Agreement shall be binding upon and inure to the benefit of the parties "
            "and their respective successors and assigns. Neither party shall assign or "
            "transfer this Agreement or any of its rights or obligations hereunder without "
            "the prior written consent of the other party, which shall not be unreasonably "
            "withheld or delayed."
        ),
        "output": "Anti-Assignment",
    },
]


def build_system_prompt() -> str:
    return (
        "You are a legal AI assistant specialized in analyzing commercial contracts. "
        "Your task is to classify contract clauses into predefined categories. "
        "Respond with ONLY the category name and nothing else. "
        "Do not include explanations, punctuation, or any other text."
    )


def build_user_prompt(clause_text: str) -> str:
    """Build the full few-shot user message."""
    lines = [
        "Classify the following contract clause into exactly one of these categories:",
        "",
        CATEGORY_LIST_STR,
        "",
        "Use 'None' if the text does not match any specific clause category.",
        "",
        "--- EXAMPLES ---",
    ]

    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        lines.append(f"\nExample {i}:")
        lines.append(f"Clause: {ex['input']}")
        lines.append(f"Category: {ex['output']}")

    lines.append("\n--- CLAUSE TO CLASSIFY ---")
    lines.append(f"Clause: {clause_text}")
    lines.append("Category:")

    return "\n".join(lines)


def classify_clause(client: anthropic.Anthropic, clause_text: str, model: str) -> tuple[str, dict]:
    """Call the API and return (prediction, usage_info)."""
    t0 = time.time()

    message = client.messages.create(
        model=model,
        max_tokens=32,  # Category names are short
        system=build_system_prompt(),
        messages=[
            {"role": "user", "content": build_user_prompt(clause_text)},
        ],
        temperature=0,  # Deterministic for evaluation
    )

    latency = time.time() - t0
    raw = message.content[0].text.strip()

    # Normalise: strip quotes/punctuation the model may add
    prediction = raw.strip("'\".,;: ")

    # Fuzzy match back to known categories
    prediction_lower = prediction.lower()
    matched = None
    for cat in SELECTED_CATEGORIES:
        if cat.lower() == prediction_lower:
            matched = cat
            break
    if matched is None:
        # Try partial match (e.g., model outputs "IP Ownership")
        for cat in SELECTED_CATEGORIES:
            if cat.lower() in prediction_lower or prediction_lower in cat.lower():
                matched = cat
                break
    if matched is None:
        matched = "None"  # Default to None if we can't parse

    usage = {
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
        "latency_s": round(latency, 3),
        "raw_response": raw,
        "matched": matched,
    }
    return matched, usage


def run(dry_run: bool = False):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. "
            "Add it to a .env file in the repo root: ANTHROPIC_API_KEY=sk-..."
        )

    client = anthropic.Anthropic(api_key=api_key)
    model = "claude-haiku-4-5"   # Fast, cheap; frontier-quality for this task

    # Load test examples
    with open(TEST_FILE) as f:
        examples = [json.loads(l) for l in f]

    if dry_run:
        examples = examples[:5]
        print(f"=== DRY RUN: evaluating {len(examples)} examples ===\n")
    else:
        print(f"=== FULL RUN: evaluating {len(examples)} examples ===\n")

    y_true, y_pred = [], []
    all_usage = []
    errors = []

    for i, ex in enumerate(examples):
        clause = ex["input"]
        gold = ex["output"]

        try:
            pred, usage = classify_clause(client, clause, model)
        except Exception as e:
            print(f"  [ERROR] example {i}: {e}")
            errors.append({"index": i, "error": str(e)})
            pred = "None"
            usage = {}

        y_true.append(gold)
        y_pred.append(pred)
        all_usage.append(usage)

        if dry_run or i % 50 == 0:
            status = "✓" if pred == gold else "✗"
            print(f"  [{i:4d}] {status} gold={gold!r:35} pred={pred!r:35} ({usage.get('latency_s', '?')}s)")

        # Small rate-limit buffer
        if not dry_run:
            time.sleep(0.05)

    # ── Metrics ─────────────────────────────────────────────────────────────
    labels = [c for c in SELECTED_CATEGORIES if c in y_true or c in y_pred]
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

    total_input_tokens = sum(u.get("input_tokens", 0) for u in all_usage)
    total_output_tokens = sum(u.get("output_tokens", 0) for u in all_usage)
    avg_latency = sum(u.get("latency_s", 0) for u in all_usage) / max(len(all_usage), 1)

    # Claude Haiku pricing (as of mid-2025): $0.80/M input, $4.00/M output
    cost_per_run = (total_input_tokens / 1_000_000 * 0.80) + (total_output_tokens / 1_000_000 * 4.00)
    # Scale to cost per 1,000 contracts
    n_contracts_sampled = len(set(ex["input"][:50] for ex in examples))  # rough proxy
    contracts_per_run = max(1, n_contracts_sampled)
    avg_examples_per_contract = len(examples) / contracts_per_run
    cost_per_1k_contracts = (cost_per_run / contracts_per_run) * 1000

    print(f"\n{'='*60}")
    print(f"RESULTS (model={model})")
    print(f"{'='*60}")
    print(f"Macro-F1:                {macro_f1:.4f}")
    print(f"Total input tokens:      {total_input_tokens:,}")
    print(f"Total output tokens:     {total_output_tokens:,}")
    print(f"Est. cost this run:      ${cost_per_run:.4f}")
    print(f"Est. cost/1k contracts:  ${cost_per_1k_contracts:.2f}")
    print(f"Avg latency/example:     {avg_latency:.3f}s")
    print(f"Errors:                  {len(errors)}")

    print(f"\n{'Category':<35} {'P':>6} {'R':>6} {'F1':>6} {'Support':>8}")
    print(f"{'-'*35} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")
    for cat in SELECTED_CATEGORIES:
        if cat in report:
            r = report[cat]
            print(f"{cat:<35} {r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1-score']:>6.3f} {int(r['support']):>8}")
    print(f"\n{'macro avg':<35} {report['macro avg']['precision']:>6.3f} {report['macro avg']['recall']:>6.3f} {macro_f1:>6.3f}")

    # Save results (only on full run)
    if not dry_run:
        results = {
            "model": model,
            "n_examples": len(examples),
            "macro_f1": round(macro_f1, 4),
            "per_category": {
                cat: report.get(cat, {}) for cat in SELECTED_CATEGORIES
            },
            "cost": {
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "cost_usd_this_run": round(cost_per_run, 4),
                "cost_usd_per_1k_contracts": round(cost_per_1k_contracts, 2),
                "avg_latency_s": round(avg_latency, 3),
            },
            "errors": errors,
        }
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {RESULTS_FILE}")
    else:
        print("\n[DRY RUN] Results not saved. Re-run without --dry-run for the full evaluation.")

    return macro_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Only run on 5 examples for prompt review")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
