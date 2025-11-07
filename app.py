# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, auc
)
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

st.set_page_config(page_title="Insurance Risk & Retention Intelligence", layout="wide")

# ------------- Helpers -------------
@st.cache_data(show_spinner=False)
def load_csv(path_or_buffer):
    try:
        return pd.read_csv(path_or_buffer)
    except Exception:
        try:
            return pd.read_csv(path_or_buffer, sep=';')
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            return None

def clean_data(df):
    return df.dropna(axis=1, how='all').drop_duplicates()

def normalize_binary(y):
    if y is None:
        return None
    if y.dtype == object:
        y2 = y.astype(str).str.lower().map(lambda s: 1 if s in ['yes','y','true','1','churn','claimed','madeclaim'] else (0 if s in ['no','n','false','0','renewed','retained'] else np.nan))
        if y2.notna().mean() >= 0.5:
            return y2.astype(int)
        return pd.Categorical(y).codes
    y2 = pd.to_numeric(y, errors='coerce')
    vals = pd.Series(y2).dropna().unique()
    if not set(vals).issubset({0,1}):
        med = np.nanmedian(y2)
        return (y2 > med).astype(int)
    return y2.astype(int)

def guess_columns(df):
    cols = df.columns.tolist()
    # typical insurance columns
    def has(name): return next((c for c in cols if name.lower() in c.lower()), None)
    label = next((c for c in cols if any(k in c.lower() for k in ['churn','attrition','renew','claimmade','madeclaim'])), None)
    region = has('region')
    policytype = has('policy') or has('policytype')
    satis = has('satisf') or has('score')
    if satis:
        # ensure numeric-ish
        if not pd.api.types.is_numeric_dtype(df[satis]):
            try:
                pd.to_numeric(df[satis])
            except:
                satis = None
    age = has('age')
    smoker = has('smoker') or has('smoking')
    bmi = has('bmi')
    charges = has('charges') or has('claim') or has('billed') or has('amount')
    premium = has('premium')
    segment = has('segment')
    return dict(label=label, region=region, policytype=policytype, satis=satis, age=age, smoker=smoker, bmi=bmi, charges=charges, premium=premium, segment=segment)

def build_preprocessor(X):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    pre = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
    ])
    return pre, num_cols, cat_cols

def split_xy(df, label_col):
    X = df.drop(columns=[label_col])
    y = normalize_binary(df[label_col])
    return X, y

def apply_filters(df, region_col, policy_col, regions, policies, slider_col, slider_min, slider_max):
    out = df.copy()
    if region_col and regions:
        out = out[out[region_col].astype(str).isin(regions)]
    if policy_col and policies:
        out = out[out[policy_col].astype(str).isin(policies)]
    if slider_col and slider_min is not None and slider_max is not None:
        out[slider_col] = pd.to_numeric(out[slider_col], errors='coerce')
        out = out[(out[slider_col] >= slider_min) & (out[slider_col] <= slider_max)]
    return out

def train_models(X, y, random_state=42):
    pre, _, _ = build_preprocessor(X)
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state)
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    rows, rocs = [], {}
    for name, clf in models.items():
        pipe = Pipeline([('prep', pre), ('clf', clf)])
        y_true, y_pred, y_prob = [], [], []
        for tr, te in skf.split(X, y):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            ytr, yte = y.iloc[tr], y.iloc[te]
            pipe.fit(Xtr, ytr)
            yp = pipe.predict(Xte)
            if hasattr(pipe.named_steps['clf'], 'predict_proba'):
                ypp = pipe.predict_proba(Xte)[:,1]
            else:
                try:
                    dec = pipe.decision_function(Xte)
                    ypp = (dec - dec.min()) / (dec.max() - dec.min() + 1e-9)
                except:
                    ypp = yp.astype(float)
            y_true.extend(yte); y_pred.extend(yp); y_prob.extend(ypp)
        y_true, y_pred, y_prob = np.array(y_true), np.array(y_pred), np.array(y_prob)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc_val = roc_auc_score(y_true, y_prob)
        except:
            fpr, tpr, _ = roc_curve(y_true, y_pred); auc_val = auc(fpr, tpr)
        rows.append(dict(Algorithm=name, **{
            "Testing Accuracy": round(acc,4),
            "Precision": round(prec,4),
            "Recall": round(rec,4),
            "F1": round(f1,4),
            "AUC": round(auc_val,4)
        }))
        fpr, tpr, _ = roc_curve(y_true, y_prob); rocs[name] = (fpr, tpr)
    return pd.DataFrame(rows).set_index("Algorithm"), rocs, pre, models

# ------------- Data load -------------
st.sidebar.header("Dataset")
st.sidebar.caption("Place Insurance.csv in repo root or upload here.")
upl = st.sidebar.file_uploader("Upload Insurance CSV (optional)", type=['csv'])
if upl is not None:
    df = load_csv(upl)
else:
    df = None
    for fname in ["Insurance.csv", "sample_insurance.csv"]:
        try:
            df = load_csv(fname)
            if df is not None and not df.empty:
                break
        except:
            pass

if df is None or df.empty:
    st.error("No data found. Upload Insurance.csv or include it in the repo root.")
    st.stop()

df = clean_data(df)

# ------------- Column mapping -------------
st.sidebar.header("Column Mapping")
g = guess_columns(df)
label_col = st.sidebar.selectbox("Target (Churn/Renewal/ClaimMade)", [None]+df.columns.tolist(), index=(df.columns.tolist().index(g['label'])+1 if g['label'] in df.columns else 0))
region_col = st.sidebar.selectbox("Region", [None]+df.columns.tolist(), index=(df.columns.tolist().index(g['region'])+1 if g['region'] in df.columns else 0))
policy_col = st.sidebar.selectbox("Policy Type", [None]+df.columns.tolist(), index=(df.columns.tolist().index(g['policytype'])+1 if g['policytype'] in df.columns else 0))
satis_col = st.sidebar.selectbox("Satisfaction (numeric) (used for slider)", [None]+df.columns.tolist(), index=(df.columns.tolist().index(g['satis'])+1 if g['satis'] in df.columns else 0))
age_col = st.sidebar.selectbox("Age (numeric)", [None]+df.columns.tolist(), index=(df.columns.tolist().index(g['age'])+1 if g['age'] in df.columns else 0))
smoker_col = st.sidebar.selectbox("Smoker", [None]+df.columns.tolist(), index=(df.columns.tolist().index(g['smoker'])+1 if g['smoker'] in df.columns else 0))
bmi_col = st.sidebar.selectbox("BMI", [None]+df.columns.tolist(), index=(df.columns.tolist().index(g['bmi'])+1 if g['bmi'] in df.columns else 0))
charges_col = st.sidebar.selectbox("Charges / Claim Amount", [None]+df.columns.tolist(), index=(df.columns.tolist().index(g['charges'])+1 if g['charges'] in df.columns else 0))
premium_col = st.sidebar.selectbox("Premium", [None]+df.columns.tolist(), index=(df.columns.tolist().index(g['premium'])+1 if g['premium'] in df.columns else 0))
segment_col = st.sidebar.selectbox("Customer Segment (optional)", [None]+df.columns.tolist(), index=(df.columns.tolist().index(g['segment'])+1 if g['segment'] in df.columns else 0))

# ------------- Filters -------------
st.sidebar.header("Filters (apply to charts)")
if region_col:
    regions = sorted(df[region_col].dropna().astype(str).unique().tolist())
    sel_regions = st.sidebar.multiselect("Region(s)", regions, default=regions[:min(3, len(regions))])
else:
    sel_regions = []

if policy_col:
    pols = sorted(df[policy_col].dropna().astype(str).unique().tolist())
    sel_pols = st.sidebar.multiselect("Policy Type(s)", pols, default=pols[:min(3, len(pols))])
else:
    sel_pols = []

slider_col = satis_col if satis_col else charges_col
if slider_col:
    s_series = pd.to_numeric(df[slider_col], errors='coerce')
    low = float(np.nanmin(s_series)) if np.isfinite(np.nanmin(s_series)) else 0.0
    high = float(np.nanmax(s_series)) if np.isfinite(np.nanmax(s_series)) else 1.0
    s_lo, s_hi = st.sidebar.slider(f"{slider_col} range", min_value=low, max_value=high, value=(low, high))
else:
    s_lo, s_hi = None, None

filtered = apply_filters(df, region_col, policy_col, sel_regions, sel_pols, slider_col, s_lo, s_hi)

# ------------- Tabs -------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview & Insights", "📈 Deep-Dive Charts", "🧠 Model Training & Evaluation", "🪄 Predict & Download"])

with tab1:
    st.markdown("## Overview & Insights")
    st.metric("Records (after filters)", len(filtered))
    if label_col:
        try:
            ybin = normalize_binary(filtered[label_col])
            rate = float(np.nanmean(ybin))
            st.metric("Positive rate (Churn/Claim/Renewal target)", f"{rate*100:.2f}%")
        except Exception as e:
            st.info(f"Could not compute target rate: {e}")

    st.divider()
    st.subheader("1) Average Claim Amount by Region and Policy Type (clustered bar)")
    if region_col and policy_col and charges_col:
        tmp = filtered[[region_col, policy_col, charges_col]].copy()
        tmp[charges_col] = pd.to_numeric(tmp[charges_col], errors='coerce')
        agg = tmp.groupby([region_col, policy_col], dropna=False)[charges_col].mean().reset_index(name='AvgCharges')
        chart = alt.Chart(agg).mark_bar().encode(
            x=alt.X(f'{region_col}:N', title='Region'),
            y=alt.Y('AvgCharges:Q', title='Avg Claim Amount'),
            color=alt.Color(f'{policy_col}:N', title='Policy Type'),
            tooltip=[region_col, policy_col, alt.Tooltip('AvgCharges:Q', format=',.2f')]
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Map Region, Policy Type, and Charges/Claim Amount to view this chart.")

with tab2:
    st.markdown("## Deep-Dive Charts")
    st.caption("Five complementary visualizations. All reflect the global filters.")
    # 2) Satisfaction vs. Claim Amount regression
    st.subheader("2) Satisfaction vs. Claim Amount (regression trend)")
    if satis_col and charges_col:
        t2 = filtered[[satis_col, charges_col]].dropna()
        t2[satis_col] = pd.to_numeric(t2[satis_col], errors='coerce')
        t2[charges_col] = pd.to_numeric(t2[charges_col], errors='coerce')
        t2 = t2.dropna()
        chart = alt.Chart(t2).mark_circle(opacity=0.4).encode(
            x=alt.X(f'{satis_col}:Q', title='Satisfaction'),
            y=alt.Y(f'{charges_col}:Q', title='Claim Amount'),
            tooltip=[satis_col, charges_col]
        ).properties(height=300)
        trend = chart.transform_regression(satis_col, charges_col, method='linear').mark_line()
        st.altair_chart(chart + trend, use_container_width=True)
    else:
        st.info("Select Satisfaction and Charges/Claim Amount.")

    # 3) Heatmap: Age group x Smoker -> Avg Premium
    st.subheader("3) Heatmap: Age group × Smoking Status → Average Premium")
    if age_col and smoker_col and premium_col:
        tmp = filtered[[age_col, smoker_col, premium_col]].dropna()
        tmp[age_col] = pd.to_numeric(tmp[age_col], errors='coerce')
        tmp[premium_col] = pd.to_numeric(tmp[premium_col], errors='coerce')
        tmp = tmp.dropna()
        tmp['AgeBin'] = pd.cut(tmp[age_col], bins=5)
        agg = tmp.groupby(['AgeBin', smoker_col], dropna=False)[premium_col].mean().reset_index(name='AvgPremium')
        hm = alt.Chart(agg).mark_rect().encode(
            x=alt.X('AgeBin:O', title='Age Bucket'),
            y=alt.Y(f'{smoker_col}:N', title='Smoker'),
            color=alt.Color('AvgPremium:Q', title='Avg Premium', scale=alt.Scale(scheme='blues')),
            tooltip=['AgeBin', smoker_col, alt.Tooltip('AvgPremium:Q', format=',.2f')]
        )
        st.altair_chart(hm, use_container_width=True)
    else:
        st.info("Select Age, Smoker, and Premium columns.")

    # 4) Feature Importance (Random Forest) wrt target
    st.subheader("4) Feature Importance (Random Forest)")
    if label_col:
        use_df = filtered.dropna(subset=[label_col])
        if not use_df.empty:
            X, y = split_xy(use_df, label_col)
            pre, num_cols, cat_cols = build_preprocessor(X)
            rf = RandomForestClassifier(n_estimators=300, random_state=42)
            pipe = Pipeline([('prep', pre), ('clf', rf)])
            try:
                pipe.fit(X, y)
                cat_features = []
                if len(cat_cols):
                    ohe = pipe.named_steps['prep'].named_transformers_['cat'].named_steps['onehot']
                    cat_features = list(ohe.get_feature_names_out(cat_cols))
                feat_names = list(num_cols) + cat_features
                imps = pipe.named_steps['clf'].feature_importances_
                imp_df = pd.DataFrame({'Feature': feat_names, 'Importance': imps}).sort_values('Importance', ascending=False).head(20)
                chart = alt.Chart(imp_df).mark_bar().encode(
                    x=alt.X('Importance:Q'),
                    y=alt.Y('Feature:N', sort='-x'),
                    tooltip=['Feature', alt.Tooltip('Importance:Q', format='.4f')]
                )
                st.altair_chart(chart, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not compute feature importance: {e}")
        else:
            st.info("Not enough rows with target to compute feature importance.")
    else:
        st.info("Select a target column for feature importance.")

    # 5) Bubble: BMI x Age vs Total Charges (size)
    st.subheader("5) Bubble: BMI × Age vs Total Charges")
    if bmi_col and age_col and charges_col:
        tmp = filtered[[bmi_col, age_col, charges_col]].dropna()
        tmp[bmi_col] = pd.to_numeric(tmp[bmi_col], errors='coerce')
        tmp[age_col] = pd.to_numeric(tmp[age_col], errors='coerce')
        tmp[charges_col] = pd.to_numeric(tmp[charges_col], errors='coerce')
        tmp = tmp.dropna()
        bubble = alt.Chart(tmp).mark_circle(opacity=0.35).encode(
            x=alt.X(f'{bmi_col}:Q', title='BMI'),
            y=alt.Y(f'{age_col}:Q', title='Age'),
            size=alt.Size(f'{charges_col}:Q', title='Total Charges', legend=None),
            tooltip=[bmi_col, age_col, charges_col]
        ).interactive()
        st.altair_chart(bubble, use_container_width=True)
    else:
        st.info("Select BMI, Age, and Charges columns.")

with tab3:
    st.markdown("## Model Training & Evaluation")
    if not label_col:
        st.error("Select a binary target column (e.g., Churn/Renewal/ClaimMade).")
    else:
        use_df = filtered.dropna(subset=[label_col])
        if use_df.empty:
            st.error("No rows available after filters/NA removal for the selected target.")
        else:
            X, y = split_xy(use_df, label_col)
            st.write(f"Using **{len(X)}** records after filtering/cleanup.")
            go = st.button("Run 5-fold Stratified CV on Decision Tree / Random Forest / Gradient Boosting")
            if go:
                with st.spinner("Training..."):
                    metrics_df, roc_curves, preprocessor, models = train_models(X, y)
                st.subheader("Metrics Table")
                st.dataframe(metrics_df)

                st.subheader("ROC Curves")
                fig = plt.figure(figsize=(6,4))
                for name, (fpr, tpr) in roc_curves.items():
                    plt.plot(fpr, tpr, label=name)
                plt.plot([0,1], [0,1], linestyle='--')
                plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curves"); plt.legend()
                st.pyplot(fig)

                # Confusion matrices on holdout split
                st.subheader("Confusion Matrices (holdout split)")
                try:
                    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
                except Exception:
                    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
                for name, clf in models.items():
                    pipe = Pipeline([('prep', preprocessor), ('clf', clf)])
                    pipe.fit(Xtr, ytr)
                    yhat = pipe.predict(Xte)
                    cm = confusion_matrix(yte, yhat, labels=[0,1])
                    cm_df = pd.DataFrame(cm, index=['Actual 0','Actual 1'], columns=['Pred 0','Pred 1'])
                    st.write(f"**{name}**"); st.dataframe(cm_df)

with tab4:
    st.markdown("## Predict on New Data & Download Results")
    st.write("Upload a CSV **without the target**. We'll fit on current filtered data and return predictions.")
    uploaded = st.file_uploader("Upload CSV for prediction", type=['csv'])
    model_choice = st.selectbox("Model for final fit", ["Random Forest", "Decision Tree", "Gradient Boosting"])
    predict_btn = st.button("Fit on Current Data & Predict on Uploaded File")

    if predict_btn:
        if uploaded is None:
            st.error("Please upload a CSV file to predict.")
        elif not label_col:
            st.error("Please select the target column in the sidebar first.")
        else:
            use_df = filtered.dropna(subset=[label_col])
            if use_df.empty:
                st.error("No rows available for training. Adjust filters or column mapping.")
            else:
                X_full, y_full = split_xy(use_df, label_col)
                pre, _, _ = build_preprocessor(X_full)
                if model_choice == "Decision Tree":
                    clf = DecisionTreeClassifier(random_state=42)
                elif model_choice == "Gradient Boosting":
                    clf = GradientBoostingClassifier(random_state=42)
                else:
                    clf = RandomForestClassifier(n_estimators=300, random_state=42)
                pipe = Pipeline([('prep', pre), ('clf', clf)])
                pipe.fit(X_full, y_full)

                new_df = load_csv(uploaded)
                if new_df is None or new_df.empty:
                    st.error("Uploaded CSV could not be read or is empty.")
                else:
                    # Drop any label-like columns to avoid leakage
                    drop_cols = [c for c in new_df.columns if any(k in c.lower() for k in ['churn','attrition','renew','claimmade','madeclaim'])]
                    newX = new_df.drop(columns=drop_cols) if drop_cols else new_df.copy()
                    try:
                        if hasattr(pipe.named_steps['clf'], 'predict_proba'):
                            probs = pipe.predict_proba(newX)[:,1]
                        else:
                            preds_raw = pipe.predict(newX)
                            probs = preds_raw.astype(float)
                        preds = pipe.predict(newX)
                        out = new_df.copy()
                        out['prediction'] = preds
                        out['probability'] = probs
                        st.success("Predictions generated.")
                        st.dataframe(out.head(50))
                        csv_bytes = out.to_csv(index=False).encode('utf-8')
                        st.download_button("Download predictions CSV", data=csv_bytes, file_name="insurance_predictions.csv", mime="text/csv")
                    except Exception as e:
                        st.error(f"Prediction failed. Make sure columns are compatible. Error: {e}")

st.caption("Notes: Nulls are imputed (median for numeric, most frequent for categorical). OneHotEncoding is used for categoricals. Global filters apply to charts. If names differ (e.g., policy type column), use the sidebar mapping. Null values are ignored where necessary.")
