# Data Card

## Dataset Source

- **Name:** CUAD (Contract Understanding Atticus Dataset)
- **License:** CC BY 4.0
- **Source:** [TheAtticusProject/cuad](https://github.com/TheAtticusProject/cuad)

## Category Selection

The dataset originally contains 41 clause categories. We selected 14 target categories for v1:
- `Governing Law`
- `Anti-Assignment`
- `Cap On Liability`
- `License Grant`
- `Audit Rights`
- `Termination For Convenience`
- `Exclusivity`
- `Renewal Term`
- `Insurance`
- `Ip Ownership Assignment`
- `Change Of Control`
- `Non-Compete`
- `Uncapped Liability`
- `Revenue/Profit Sharing`

**Exclusions:**
- **Sparse categories** (e.g., `Source Code Escrow`, `Price Restrictions`) were excluded due to insufficient positive examples (<50 per category) to train or evaluate reliably.
- **Metadata categories** (e.g., `Document Name`, `Parties`, `Agreement Date`) were excluded. These are entity extraction tasks, not semantic clause classification.

## Task Framing

We adopted a **Classification Framing**. 
The model is given a single paragraph of contract text and asked to classify it into one of the 14 categories above, or `None`.

To prevent class imbalance, the `None` class (paragraphs containing no relevant clauses) is sub-sampled at a 15% keep rate.

## Split Strategy

The data was split **by contract document**, not by clause. This prevents data leakage where boilerplate paragraphs from the same contract might end up in both train and test splits.
- Split ratios: 70% Train, 15% Validation, 15% Test
- Random seed: `42`

## Sample Counts

The resulting paragraph-level examples across the 3 splits:

| Category | Train | Val | Test |
|---|---|---|---|
| **None** (sub-sampled) | 5228 | 866 | 863 |
| **License Grant** | 507 | 95 | 101 |
| **Audit Rights** | 316 | 80 | 51 |
| **Cap On Liability** | 312 | 62 | 59 |
| **Revenue/Profit Sharing** | 303 | 42 | 54 |
| **Anti-Assignment** | 298 | 73 | 61 |
| **Governing Law** | 282 | 58 | 62 |
| **Insurance** | 261 | 69 | 23 |
| **Non-Compete** | 169 | 24 | 27 |
| **Change Of Control** | 149 | 15 | 27 |
| **Exclusivity** | 147 | 38 | 36 |
| **Termination For Convenience** | 144 | 28 | 23 |
| **Ip Ownership Assignment** | 141 | 35 | 35 |
| **Renewal Term** | 131 | 19 | 31 |
| **Uncapped Liability** | 90 | 23 | 22 |
| **TOTAL** | **8478** | **1527** | **1475** |
