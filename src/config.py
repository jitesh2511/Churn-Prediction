from pathlib import Path

"""
This module serves as the central location for configuration constants and file paths. 
It ensures that various components of the project access consistent settings for data paths, 
directories, model thresholds, and feature definitions. Maintaining all essential configurations 
here makes the codebase easier to maintain and less error-prone.
"""


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "customer_churn.csv"
MODEL_DIR = BASE_DIR / "model"

NUMERIC_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']

THRESHOLD = 0.35

API_URL = "https://churn-api-1b90.onrender.com"