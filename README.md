# AI Risk Manager

Hybrid fraud detection system for the Razorpay AI Buildathon (Track 02) — a transaction-level classifier combined with identity and behavioral ring detection, wrapped in a case-queue tool for an analyst to review and act on.

**Full explanation of the problem, approach, and results:** see [docs/](docs/) — start with [docs/problem-and-approach.md](docs/problem-and-approach.md).

## Status

In progress, building toward Sept 4, 2026 submission. See [docs/README.md](docs/README.md) for the current per-component status.

## Repo structure

```
data/raw/          Raw IEEE-CIS CSVs (not committed — see Setup)
data/processed/    Cached/derived data (not committed)
notebooks/         Analysis and model notebooks
src/               Shared, reusable, tested code (data loading, feature engineering, ...)
tests/             Tests for everything in src/
results/           Saved metrics from each stage, for tracking progress over time
docs/              Concept-based product documentation
```

## Setup

1. **Python**: this project pins Python 3.11.15 via [mise](https://mise.jdx.dev/) (`.mise.toml`).
2. **Create the virtual environment and install dependencies** (isolated to this project — nothing installed globally):
   ```
   python -m venv .venv
   .venv/Scripts/pip install -r requirements.txt
   ```
3. **Get the dataset**: download [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data) from Kaggle (accept the competition rules, then "Download All") and unzip the CSVs into `data/raw/`.
4. **Run the tests** to verify the code works before touching notebooks:
   ```
   .venv/Scripts/python -m pytest
   ```
5. **Run a notebook**: open `notebooks/` in Jupyter or VS Code using the `.venv` kernel.

## Testing

Code in `src/` (data loading, feature engineering, and more as it's added) has unit tests in `tests/`, run with `pytest`. Notebooks stay for analysis and reporting; anything reused across notebooks is expected to live in `src/` with a test alongside it.
