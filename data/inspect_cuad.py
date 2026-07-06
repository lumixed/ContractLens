
import json
import os
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR / "raw"
RAW_DIR.mkdir(exist_ok=True)

# CUAD official data source (Zenodo)
# Downloaded from: https://zenodo.org/records/4595826/files/CUAD_v1.zip
CUAD_FILE = RAW_DIR / "CUAD_v1.json"


def download_cuad():
    """Load the raw CUAD JSON (already downloaded from Zenodo)."""
    if not CUAD_FILE.exists():
        raise FileNotFoundError(
            f"CUAD data not found at {CUAD_FILE}. "
            "Download from https://zenodo.org/records/4595826 and extract CUAD_v1.json to data/raw/."
        )
    
    print(f"Loading CUAD dataset from {CUAD_FILE}...")
    with open(CUAD_FILE, "r") as f:
        data = json.load(f)
    
    print(f"  → Top-level keys: {list(data.keys())}")
    print(f"  → Version: {data.get('version', 'N/A')}")
    print(f"  → Number of documents: {len(data.get('data', []))}")
    
    return data


def extract_category_from_question(question: str) -> str:
    """
    CUAD questions follow patterns like:
      "Highlight the parts (if any) of this contract related to 'Anti-Assignment'..."
    Extract the category name from the question text.
    """
    # Look for the pattern: related to "Category Name" or 'Category Name'
    import re
    match = re.search(r"related to ['\"]([^'\"]+)['\"]", question)
    if match:
        return match.group(1)
    
    # Fallback: look for "Highlight the parts..." pattern
    match = re.search(r"related to (.+?)(?:\.|,|\s+that)", question)
    if match:
        return match.group(1).strip().strip("'\"")
    
    return question[:80]  # fallback: truncate


def inspect_dataset(data):
    """Analyze the dataset and produce summary statistics."""
    
    documents = data.get("data", [])
    
    # Track contracts and categories
    contracts = {}  # contract_title -> context_text_length
    category_counter = Counter()  # total QA pairs per category
    category_positive = Counter()  # QA pairs with actual answer spans
    all_categories = set()
    total_qa_pairs = 0
    total_annotations = 0
    
    for doc in documents:
        contract_title = doc.get("title", "unknown")
        
        for paragraph in doc.get("paragraphs", []):
            context = paragraph.get("context", "")
            context_len = len(context)
            
            # Track unique contracts by their context (some titles may repeat
            # but contexts differ due to paragraph splitting)
            if contract_title not in contracts:
                contracts[contract_title] = context_len
            else:
                contracts[contract_title] = max(contracts[contract_title], context_len)
            
            for qa in paragraph.get("qas", []):
                total_qa_pairs += 1
                question = qa.get("question", "")
                category = extract_category_from_question(question)
                all_categories.add(category)
                category_counter[category] += 1
                
                # Check for actual answer spans
                answers = qa.get("answers", [])
                if answers and any(a.get("text", "").strip() for a in answers):
                    category_positive[category] += 1
                    total_annotations += len([
                        a for a in answers if a.get("text", "").strip()
                    ])
    
    # Build summary
    summary = {
        "total_documents": len(documents),
        "unique_contracts": len(contracts),
        "total_qa_pairs": total_qa_pairs,
        "total_annotations": total_annotations,
        "unique_categories": len(all_categories),
    }
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"CUAD DATASET SUMMARY")
    print(f"{'='*80}")
    print(f"Total documents (titles):    {summary['total_documents']}")
    print(f"Unique contracts:            {summary['unique_contracts']}")
    print(f"Total QA pairs:              {summary['total_qa_pairs']}")
    print(f"Total answer annotations:    {summary['total_annotations']}")
    print(f"Unique categories:           {summary['unique_categories']}")
    
    print(f"\n{'='*80}")
    print(f"CATEGORY FREQUENCY (sorted by positive example count)")
    print(f"{'='*80}")
    print(f"{'#':<4} {'Category':<50} {'Total QA':<10} {'Positive':<10} {'% Pos':<8}")
    print(f"{'-'*4} {'-'*50} {'-'*10} {'-'*10} {'-'*8}")
    
    # Sort by positive count (descending)
    sorted_categories = sorted(
        all_categories,
        key=lambda c: category_positive.get(c, 0),
        reverse=True
    )
    
    category_details = []
    for i, cat in enumerate(sorted_categories, 1):
        total = category_counter[cat]
        pos = category_positive.get(cat, 0)
        pct = (pos / total * 100) if total > 0 else 0
        print(f"{i:<4} {cat:<50} {total:<10} {pos:<10} {pct:<8.1f}")
        category_details.append({
            "rank": i,
            "category": cat,
            "total_qa_pairs": total,
            "positive_examples": pos,
            "positive_pct": round(pct, 1)
        })
    
    summary["category_details"] = category_details
    summary["sample_contract_titles"] = sorted(list(contracts.keys()))[:15]
    
    # Recommended top categories (positive examples >= threshold)
    threshold = 50
    recommended = [
        cd for cd in category_details if cd["positive_examples"] >= threshold
    ]
    print(f"\n{'='*80}")
    print(f"RECOMMENDED CATEGORIES (>= {threshold} positive examples)")
    print(f"{'='*80}")
    for cd in recommended:
        print(f"  {cd['rank']:>2}. {cd['category']:<50} ({cd['positive_examples']} positive)")
    print(f"\nTotal recommended: {len(recommended)} categories")
    
    # Also show the long tail
    print(f"\n{'='*80}")
    print(f"SPARSE CATEGORIES (< {threshold} positive examples)")
    print(f"{'='*80}")
    sparse = [cd for cd in category_details if cd["positive_examples"] < threshold]
    for cd in sparse:
        print(f"  {cd['rank']:>2}. {cd['category']:<50} ({cd['positive_examples']} positive)")
    print(f"\nTotal sparse: {len(sparse)} categories")
    
    # Save summary
    summary_path = SCRIPT_DIR / "cuad_inspection_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull summary saved to: {summary_path}")
    
    return summary


if __name__ == "__main__":
    data = download_cuad()
    summary = inspect_dataset(data)
