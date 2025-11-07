# Insurance Risk & Retention Intelligence (Streamlit Cloud)

Streamlit app for **risk assessment, claim likelihood, and retention analysis** using `Insurance.csv`.
- Flat repo (no folders), main entry: `app.py`
- Default packages (no version pins): `streamlit`, `pandas`, `numpy`, `scikit-learn`, `altair`, `matplotlib`
- Robust to nulls (median / most-frequent imputation)

## Tabs
1. **Overview & Insights** — Summary metrics + clustered bar (Avg Claim by Region × Policy)
2. **Deep-Dive Charts** —
   - Satisfaction vs Claim Amount (regression trend)
   - Heatmap: Age bucket × Smoker → Avg Premium
   - Random Forest feature importance (against target)
   - Bubble: BMI × Age vs Total Charges
3. **Model Training & Evaluation** — Decision Tree, Random Forest, Gradient Boosting with Accuracy, Precision, Recall, F1, AUC, ROC curves, Confusion matrices.
4. **Predict & Download** — Upload new CSV, fit on filtered data, get predictions and download CSV.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud
Push these files to a new GitHub repo (root-only), then create the app pointing to `app.py`.
