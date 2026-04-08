# Churn Prediction ML Project

This repository contains the foundation for a Churn Prediction Machine Learning project.

## Project Overview

Churn prediction aims to identify customers who are likely to stop using a service or product. By using machine learning techniques, we can predict customer churn and help businesses take proactive measures to retain users.

The goal of this project is to build an **end-to-end machine learning pipeline**, from data analysis to a deployable prediction system.

---

## Dataset

This project uses the **Telco Customer Churn dataset**, which contains customer-level information such as demographics, services used, and account details.

Dataset Link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

---

## Project Structure

```
churn-prediction/
│
├── data/                  # Directory for datasets (excluded from version control)
├── notebooks/             # Jupyter notebooks (EDA and experimentation)
│
├── src/                   # Core Python modules for data processing, model training, and saving the trained model
├── requirements.txt       # Project dependencies (mirrored in pyproject.toml)
├── pyproject.toml         # Editable install so notebooks can `import src`
├── README.md
├── LICENSE
```

---

## Tech Stack

* Python
* Pandas, NumPy
* scikit-learn
* Matplotlib, Seaborn
* (Planned) Streamlit for UI

---

## Getting Started

### 1. Clone the repository

```bash
git clone <repo-url>
cd churn-prediction
```

### 2. Set up your environment

```bash
python -m venv .venv

# Activate the virtual environment:
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Prepare your data

- Download the dataset from the link above.
- Place the CSV file inside a `data/` folder in the project root.

### 4. Run notebooks

```bash
jupyter notebook
```


---

## Project Roadmap

- [x] Phase 1: Understand the data sources, features, and business goals for churn prediction  
  - Dataset inspection completed (shape, types, missing values)
  - Initial observations documented
- [x] Phase 2: Exploratory Data Analysis (EDA)
  - Performed univariate and bivariate analysis (distributions, box plots, correlations)
  - Identified key drivers of churn using visualizations
  - Analyzed relationships between features and target variable
  - Summarized insights to guide preprocessing and modeling
- [x] Phase 3: Clean and preprocess data (handle missing values, outliers, encoding, scaling)
  - Cleaned and converted the `TotalCharges` column to numeric values, filling missing entries with 0.
  - Removed the `customerID` column as it is a non-predictive unique identifier.
  - Transformed all binary "Yes"/"No" columns (including the target `Churn`) into 1/0 binary format.
  - Applied one-hot encoding to categorical variables for compatibility with machine learning algorithms.
  - Scaled numerical features to ensure uniformity and improve model performance.
- [x] Phase 4: Model Building and Evaluation
  - Trained a baseline Logistic Regression model on the preprocessed dataset
  - Evaluated performance using classification metrics (accuracy, precision, recall, F1-score)
  - Achieved a **recall of 0.78**, successfully identifying 78% of customers likely to churn
  - Demonstrated the impact of class imbalance handling using balanced class weights
  - Established a strong baseline for further model improvement and tuning
- [x] Phase 5: Model Improvement & Threshold Tuning
  - Generated prediction probabilities and analyzed model behavior across different decision thresholds
  - Tuned classification threshold to optimize recall-precision trade-off
  - Achieved **90% recall** at threshold = 0.35, significantly improving churn detection
  - Evaluated model performance using Precision-Recall and ROC curves
  - Selected optimal threshold based on business objective of maximizing churn detection
  - Demonstrated understanding of trade-offs between recall and precision in imbalanced classification problems
- [x] Phase 6: Develop an inference pipeline for practical usage of the model
  - Refactored model inference into a reusable prediction function
  - Built an end-to-end pipeline to process raw input data and generate churn predictions
  - Integrated preprocessing, feature encoding, and scaling with trained model artifacts
  - Ensured consistency between training and inference by reusing saved scaler and feature schema
  - Enabled predictions on unseen data by accepting external input files
  - Returned model outputs (predictions and probabilities) in a format suitable for downstream applications (e.g., UI)
  - Designed the pipeline to be modular and extensible for future deployment and integration

* [ &nbsp; ] Phase 7: Create an interactive interface for predictions (Streamlit)
* [ &nbsp; ] Phase 8: Refactor and organize codebase for clarity and maintainability
* [ &nbsp; ] Phase 9: Add explainability tools and documentation to interpret model outputs

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---
