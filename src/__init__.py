import os

# MLflow 3.x prints an "agent hint" on import; keep training logs clean.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
