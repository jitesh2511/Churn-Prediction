from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "customer_churn.csv"
MODEL_DIR = BASE_DIR / "model"