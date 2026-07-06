import argparse
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from sklearn.metrics import classification_report, f1_score

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TEST_FILE = REPO_ROOT / "data" / "test.jsonl"
RESULTS_FILE = SCRIPT_DIR / "baseline_results.json"

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
            "Nothing in this Agreement shall exclude or limit either party's liability for: "
            "(a) death or personal injury resulting from the negligence of either party or "
            "their servants, agents or employees; or (b) fraud or fraudulent misrepresentation."
        ),
        "output": "Uncapped Liability",
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

def normalise_prediction(raw: str) -> str:
    cleaned = raw.strip("'\".,;: ")
    lower = cleaned.lower()
    for cat in SELECTED_CATEGORIES:
        if cat.lower() == lower:
            return cat
    for cat in SELECTED_CATEGORIES:
        if cat.lower() in lower or lower in cat.lower():
            return cat
    return "None"

def classify_anthropic(client, clause_text: str) -> tuple[str, dict]:
    import anthropic as _anthropic
    t0 = time.time()
    message = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=32,
        system=build_system_prompt(),
        messages=[{"role": "user", "content": build_user_prompt(clause_text)}],
        temperature=0,
    )
    latency = time.time() - t0
    raw = message.content[0].text.strip()
    return normalise_prediction(raw), {
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
        "latency_s": round(latency, 3),
        "raw_response": raw,
    }

def classify_openai(client, clause_text: str) -> tuple[str, dict]:
    t0 = time.time()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=32,
        temperature=0,
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_user_prompt(clause_text)},
        ],
    )
    latency = time.time() - t0
    raw = response.choices[0].message.content.strip()
    usage = response.usage
    return normalise_prediction(raw), {
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "latency_s": round(latency, 3),
        "raw_response": raw,
    }

_MOCK_KEYWORDS = {
    "Governing Law":              ["governed by", "governing law", "jurisdiction", "laws of the state"],
    "Anti-Assignment":            ["shall not assign", "without prior written consent", "consent to assign"],
    "Cap On Liability":           ["liability.*shall not exceed", "limited to", "in no event.*liable", "cap on liability"],
    "License Grant":              ["hereby grants", "non-exclusive.*license", "license to use", "sublicense"],
    "Audit Rights":               ["audit", "inspect.*books", "right to examine"],
    "Termination For Convenience":["terminate.*without cause", "terminate.*convenience", "days.*written notice.*terminat"],
    "Exclusivity":                ["exclusive", "solely", "not.*appoint.*other"],
    "Renewal Term":               ["automatically renew", "renewal term", "successive.*year", "unless.*notice"],
    "Insurance":                  ["insurance", "liability insurance", "workers.*compensation"],
    "Ip Ownership Assignment":    ["intellectual property.*assign", "work.*hire", "ip.*ownership", "assigns.*right.*title"],
    "Change Of Control":          ["change of control", "merger", "acquisition", "majority.*shares"],
    "Non-Compete":                ["non-compete", "not.*compete", "competing.*business", "competitive.*activit"],
    "Uncapped Liability":         ["unlimited liability", "no.*limit.*liability", "fully liable", "all damages"],
    "Revenue/Profit Sharing":     ["revenue.*shar", "profit.*shar", "royalt", "commission", "percentage.*revenue"],
}

def classify_mock(clause_text: str) -> tuple[str, dict]:
    t0 = time.time()
    text_lower = clause_text.lower()
    for category, patterns in _MOCK_KEYWORDS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                latency = time.time() - t0
                return category, {"input_tokens": 0, "output_tokens": 0, "latency_s": round(latency, 4), "raw_response": f"[mock:{category}]"}
    latency = time.time() - t0
    return "None", {"input_tokens": 0, "output_tokens": 0, "latency_s": round(latency, 4), "raw_response": "[mock:None]"}

def run(dry_run: bool = False, provider: str = "anthropic"):
    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set")
        import anthropic as _anthropic
        client = _anthropic.Anthropic(api_key=api_key)
        classify_fn = lambda text: classify_anthropic(client, text)
        model_label = "claude-haiku-4-5"
        input_price, output_price = 0.80, 4.00
    elif provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set")
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        classify_fn = lambda text: classify_openai(client, text)
        model_label = "gpt-4o-mini"
        input_price, output_price = 0.15, 0.60
    elif provider == "mock":
        classify_fn = lambda text: classify_mock(text)
        model_label = "mock-keyword-heuristic"
        input_price, output_price = 0.0, 0.0
    else:
        raise ValueError(f"Unknown provider: {provider}")

    with open(TEST_FILE) as f:
        examples = [json.loads(l) for l in f]

    if dry_run:
        examples = examples[:5]
        print(f"=== DRY RUN ({provider}/{model_label}): {len(examples)} examples ===\n")
    else:
        print(f"=== FULL RUN ({provider}/{model_label}): {len(examples)} examples ===\n")

    y_true, y_pred, all_usage, errors = [], [], [], []

    for i, ex in enumerate(examples):
        clause = ex["input"]
        gold = ex["output"]

        try:
            pred, usage = classify_fn(clause)
        except Exception as e:
            print(f"  [ERROR] example {i}: {e}")
            errors.append({"index": i, "error": str(e)})
            pred, usage = "None", {}

        y_true.append(gold)
        y_pred.append(pred)
        all_usage.append(usage)

        if dry_run or i % 100 == 0:
            status = "✓" if pred == gold else "✗"
            print(f"  [{i:4d}] {status} gold={gold!r:35} pred={pred!r:35} ({usage.get('latency_s', '?')}s)")

        if not dry_run and provider != "mock":
            time.sleep(0.05)

    labels = [c for c in SELECTED_CATEGORIES if c in y_true or c in y_pred]
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

    total_in = sum(u.get("input_tokens", 0) for u in all_usage)
    total_out = sum(u.get("output_tokens", 0) for u in all_usage)
    avg_lat = sum(u.get("latency_s", 0) for u in all_usage) / max(len(all_usage), 1)
    cost_run = (total_in / 1_000_000 * input_price) + (total_out / 1_000_000 * output_price)

    avg_paragraphs_per_contract = 20
    cost_per_1k = cost_run / max(len(examples), 1) * avg_paragraphs_per_contract * 1000

    print(f"\n{'='*60}")
    print(f"RESULTS  model={model_label}")
    print(f"{'='*60}")
    print(f"Macro-F1:                {macro_f1:.4f}")
    print(f"Total input tokens:      {total_in:,}")
    print(f"Total output tokens:     {total_out:,}")
    print(f"Est. cost this run:      ${cost_run:.4f}")
    print(f"Est. cost/1k contracts:  ${cost_per_1k:.2f}")
    print(f"Avg latency/example:     {avg_lat:.3f}s")
    print(f"Errors:                  {len(errors)}")

    print(f"\n{'Category':<35} {'P':>6} {'R':>6} {'F1':>6} {'Support':>8}")
    print(f"{'-'*35} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")
    for cat in SELECTED_CATEGORIES:
        if cat in report:
            r = report[cat]
            print(f"{cat:<35} {r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1-score']:>6.3f} {int(r['support']):>8}")
    print(f"\n{'macro avg':<35} {report['macro avg']['precision']:>6.3f} {report['macro avg']['recall']:>6.3f} {macro_f1:>6.3f}")

    if not dry_run:
        results = {
            "model": model_label,
            "provider": provider,
            "n_examples": len(examples),
            "macro_f1": round(macro_f1, 4),
            "per_category": {cat: report.get(cat, {}) for cat in SELECTED_CATEGORIES},
            "cost": {
                "total_input_tokens": total_in,
                "total_output_tokens": total_out,
                "cost_usd_this_run": round(cost_run, 4),
                "cost_usd_per_1k_contracts": round(cost_per_1k, 2),
                "avg_latency_s": round(avg_lat, 3),
            },
            "errors": errors,
        }
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {RESULTS_FILE}")
    else:
        print("\n[DRY RUN] Not saved. Run without --dry-run for full evaluation.")

    return macro_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--provider", default="anthropic",
        choices=["anthropic", "openai", "mock"],
    )
    args = parser.parse_args()
    run(dry_run=args.dry_run, provider=args.provider)
