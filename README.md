# ContractLens

A fine-tuned open-weight LLM that extracts and classifies clauses from commercial contracts, benchmarked against a prompted frontier-model baseline, served via an API with an evaluation dashboard.

> **⚠️ Disclaimer:** This is a portfolio/demonstration system, not a production legal tool. It should not be used for actual legal analysis or compliance decisions.

## Project Overview

Legal and compliance teams manually review commercial contracts to identify and classify key clauses (indemnification, termination rights, IP assignment, non-compete, etc.). This project builds a system that, given raw contract text, automatically classifies clause paragraphs into standard categories — approximating expert-level extraction at a fraction of the cost and latency of using a frontier LLM API per contract.

## Architecture

```
Raw Contracts (CUAD) → Data Pipeline → Fine-Tuned LLM (QLoRA) → FastAPI Serving → Dashboard
                                        ↕
                              Frontier Model Baseline (comparison)
```

## Repository Structure

```
contractlens/
├── data/                     # Dataset files and documentation
│   ├── raw/                  # Original CUAD download (git-ignored)
│   ├── train.jsonl           # Training split
│   ├── val.jsonl             # Validation split
│   ├── test.jsonl            # Test split
│   ├── split_manifest.json   # Contract ID → split mapping
│   └── DATA_CARD.md          # Dataset documentation
├── baseline/                 # Frontier model baseline
│   ├── run_baseline.py       # Baseline evaluation script
│   └── baseline_results.json # Baseline metrics
├── training/                 # Fine-tuning code and artifacts
│   ├── train_qlora.py        # QLoRA training script
│   ├── configs/              # Hyperparameter configurations
│   └── adapters/             # Saved LoRA adapters (git-ignored)
├── eval/                     # Evaluation pipeline
│   ├── run_eval.py           # Metrics computation
│   ├── EVAL_REPORT.md        # Full evaluation report
│   └── llm_judge.py          # LLM-as-judge scoring
├── serving/                  # Production serving
│   ├── api/
│   │   └── main.py           # FastAPI application
│   ├── Dockerfile
│   └── DEPLOYMENT.md         # Deployment instructions
├── dashboard/                # Interactive evaluation dashboard
│   └── app.py                # Streamlit/Gradio app
├── README.md                 # This file
└── LICENSE                   # MIT License
```

## Quick Start

*Coming soon — each section will be filled in as the corresponding build step is completed.*

## Key Design Decisions

*This section will be populated as decisions are made during the build process (ADR-style).*

## Results Summary

*Will contain the head-to-head comparison table between the fine-tuned model and the frontier baseline once evaluation is complete.*

## License

MIT — see [LICENSE](LICENSE) for details.

## Dataset

This project uses the [CUAD (Contract Understanding Atticus Dataset)](https://www.atticusprojectai.org/cuad), licensed under CC BY 4.0.
