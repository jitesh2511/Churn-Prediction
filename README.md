# Churn Prediction ML Project

This project implements an end-to-end machine learning pipeline to predict customer churn using real-world telecom data.

Customer churn prediction helps businesses identify users who are likely to leave a service, enabling proactive retention strategies and improving long-term revenue.

The project covers the complete ML lifecycle — from data analysis and preprocessing to model development, evaluation and explainability — along with an interactive web application for real-time predictions.

---

## Dataset

This project uses the **Telco Customer Churn dataset**, which contains customer-level information such as demographics, services used, and account details.

Dataset Link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

---

## Project Structure

```
churn-prediction/
│
├── .venv/                 # Virtual environment for project-specific Python packages (excluded from version control)
├── data/                  # Directory for datasets (excluded from version control)
├── model/                 # Directory for storing model and required artifacts (excluded from version control)
│
├── notebooks/             # Jupyter notebooks (EDA and experimentation)
├── src/                   # Core Python modules for configurations, data processing and prediction pipeline
├── app.py                 # Streamlit UI for predictions; generates results based on user input using the saved model
├── requirements.txt       # Project dependencies
├── README.md
├── LICENSE
```

---

## Tech Stack

* Python
* Pandas, NumPy
* scikit-learn
* Matplotlib, Seaborn
* Streamlit for UI

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/jitesh2511/Churn-Prediction
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
- Place the CSV file inside a `data/` directory in the project root

### 4. Train the model
This step generates the trained model and required artifacts for inference. Create a `model/` directory in project root if the step fails
```bash
python -m src.train
```

### 5. Run the app

```bash
streamlit run app.py
```

> Note: Model files are not included in the repository, you must run the training step (Step 4) before using the app

---

## Project Roadmap

- [x] Phase 1: Data Understanding  
  - Explored dataset structure, features, and initial insights  

- [x] Phase 2: Exploratory Data Analysis  
  - Identified key patterns and drivers of churn through visual analysis  

- [x] Phase 3: Data Preprocessing  
  - Cleaned data, encoded categorical features, and applied scaling  

- [x] Phase 4: Model Building  
  - Trained baseline Logistic Regression model and evaluated performance  

- [x] Phase 5: Model Improvement  
  - Optimized decision threshold and improved recall to 90%  

- [x] Phase 6: Prediction Pipeline  
  - Built reusable inference pipeline for processing unseen data  

- [x] Phase 7: Streamlit UI  
  - Developed interactive web app for real-time churn prediction  

- [x] Phase 8: Code Refactoring  
  - Modularized codebase and improved maintainability  

- [x] Phase 9: Explainability  
  - Added global and local interpretation of model predictions    
---

## 🔍 Key Highlights

- End-to-end ML pipeline from EDA to deployment  
- High recall (90%) optimized for business use-case  
- Explainable predictions with feature-level insights  
- Interactive UI for real-time inference  

---

## 🚀 Future Improvements

- Deploy the application to a cloud platform (e.g., Streamlit Cloud, AWS, or Render) for public access  
- Experiment with advanced models such as Random Forest, XGBoost, and Gradient Boosting to improve performance  
- Perform hyperparameter tuning for further optimization  
- Add model monitoring and logging for tracking performance over time  
- Improve UI/UX for a more intuitive and polished user experience  
- Extend the project into a production-ready system with API support

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---
