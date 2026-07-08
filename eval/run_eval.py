import argparse
import json
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TEST_FILE = REPO_ROOT / "data" / "test.jsonl"
RESULTS_FILE = SCRIPT_DIR / "finetuned_results.json"
PREDICTIONS_FILE = SCRIPT_DIR / "predictions.jsonl"

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

def normalise_prediction(raw: str) -> str:
    cleaned = raw.strip("'\".,;: \n")
    lower = cleaned.lower()
    for cat in SELECTED_CATEGORIES:
        if cat.lower() == lower:
            return cat
    for cat in SELECTED_CATEGORIES:
        if cat.lower() in lower or lower in cat.lower():
            return cat
    return "None"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter_path", type=str, default="training/adapters/run1_r8")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--smoke_test", action="store_true", help="Run on only 32 examples")
    args = parser.parse_args()

    print(f"Loading tokenizer {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model {args.base_model} in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto"
    )

    print(f"Loading LoRA adapter from {args.adapter_path}...")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    print(f"Loading test dataset from {TEST_FILE}...")
    with open(TEST_FILE) as f:
        examples = [json.loads(line) for line in f]
    
    if args.smoke_test:
        examples = examples[:32]
        print(f"=== SMOKE TEST: {len(examples)} examples ===")
    else:
        print(f"=== FULL RUN: {len(examples)} examples ===")

    y_true = []
    y_pred = []
    raw_outputs = []
    latencies = []
    
    total_start = time.time()
    
    for i in tqdm(range(0, len(examples), args.batch_size)):
        batch_examples = examples[i:i+args.batch_size]
        
        prompts = []
        for ex in batch_examples:
            prompt = (
                f"{ex['instruction']}\n\n"
                f"Clause: {ex['input']}\n\n"
                f"Category: "
            )
            prompts.append(prompt)
            y_true.append(ex["output"])
            
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
        input_lengths = inputs.input_ids.shape[1]
        
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        latency = time.time() - t0
        
        # Only decode the newly generated tokens
        generated_tokens = outputs[:, input_lengths:]
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        for i_batch, pred in enumerate(decoded_preds):
            pred_norm = normalise_prediction(pred)
            y_pred.append(pred_norm)
            raw_outputs.append({
                "gold": batch_examples[i_batch]["output"],
                "pred_norm": pred_norm,
                "raw_generated": pred
            })
            latencies.append(latency / len(batch_examples))
            
    total_time = time.time() - total_start
    
    # Debug: Print 20 mismatched outputs (focusing on rare categories)
    print("\n" + "="*60)
    print("DEBUG: SAMPLE OF 20 RAW OUTPUTS VS GOLD LABELS")
    print("="*60)
    debug_count = 0
    for res in raw_outputs:
        # Prioritize printing failures for rare classes
        if res["gold"] != res["pred_norm"] and res["gold"] != "None":
            print(f"GOLD: {res['gold']:<30} | PRED_NORM: {res['pred_norm']:<20} | RAW: {repr(res['raw_generated'])}")
            debug_count += 1
        if debug_count >= 20:
            break
    
    # Save detailed predictions to file
    with open(PREDICTIONS_FILE, "w") as f:
        for res in raw_outputs:
            f.write(json.dumps(res) + "\n")
    print(f"\nRaw predictions saved to {PREDICTIONS_FILE}")
    
    labels = [c for c in SELECTED_CATEGORIES if c in y_true or c in y_pred]
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

    avg_lat = sum(latencies) / max(len(latencies), 1)

    print(f"\n{'='*60}")
    print(f"RESULTS  model={args.adapter_path}")
    print(f"{'='*60}")
    print(f"Macro-F1:                {macro_f1:.4f}")
    print(f"Est. cost this run:      $0.00 (Self-hosted)")
    print(f"Est. cost/1k contracts:  $0.00 (Self-hosted)")
    print(f"Avg latency/example:     {avg_lat:.3f}s")
    print(f"Total evaluation time:   {total_time:.1f}s")
    
    print(f"\n{'Category':<35} {'P':>6} {'R':>6} {'F1':>6} {'Support':>8}")
    print(f"{'-'*35} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")
    for cat in SELECTED_CATEGORIES:
        if cat in report:
            r = report[cat]
            print(f"{cat:<35} {r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1-score']:>6.3f} {int(r['support']):>8}")
    print(f"\n{'macro avg':<35} {report['macro avg']['precision']:>6.3f} {report['macro avg']['recall']:>6.3f} {macro_f1:>6.3f}")

    results = {
        "model": args.adapter_path,
        "n_examples": len(examples),
        "macro_f1": round(macro_f1, 4),
        "per_category": {cat: report.get(cat, {}) for cat in SELECTED_CATEGORIES},
        "cost": {
            "cost_usd_this_run": 0.0,
            "cost_usd_per_1k_contracts": 0.0,
            "avg_latency_s": round(avg_lat, 3),
            "total_time_s": round(total_time, 1)
        }
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
