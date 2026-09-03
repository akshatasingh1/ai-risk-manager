# Backend-only image (src/api.py + src/serving.py). The Streamlit UI is
# deployed separately (Streamlit Community Cloud) and talks to this over
# HTTP -- it never needs these artifacts on its own host.
#
# Build from the repo root:  docker build -t ai-risk-manager-api .
# Run locally:                docker run -p 8080:8080 ai-risk-manager-api

FROM python:3.11-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt \
    && pip install --no-cache-dir --no-deps xgboost==3.2.0

COPY src/ src/

# Model/graph artifacts are gitignored (see .gitignore) but not
# dockerignored -- COPY reads the local build context, not git, so this
# still works. Only the subset serving.py's ScoringContext actually loads --
# precomputed lookup tables (src/precompute.py), not the raw graph pickles
# (~154MB, ~1.39GB once loaded as live networkx objects -- see
# docs/case-queue-and-api.md). These three parquet files replace them.
COPY models/classifier_v3_1.joblib models/category_weights.json models/rule_thresholds.json models/
COPY data/processed/transaction_graph_features.parquet data/processed/cluster_summary.parquet data/processed/cluster_edges.parquet data/processed/demo_sample.parquet data/processed/

EXPOSE 8080
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]
