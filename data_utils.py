import pandas as pd
import streamlit as st
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import io
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# === Column Handling ===

def fix_duplicate_columns(df):
    seen = {}
    new_cols = []
    for col in df.columns:
        base = str(col).strip().upper().replace(" ", "_").replace("%", "PERCENT")
        if base in seen:
            seen[base] += 1
            new_cols.append(f"{base}_{seen[base]}")
        else:
            seen[base] = 0
            new_cols.append(base)
    df.columns = new_cols
    return df

def standardize_columns(df):
    column_map = {
        "DG_POWER": "DG_UNIT",
        "DG_(UNIT)": "DG_UNIT",
        "TOTAL": "TOTAL_UNIT_(UPPCL+DG)",
        "UPPCL_POWER": "UPPCL_(UNIT_)",
        "UPPCL_(UNIT)": "UPPCL_(UNIT_)",
    }
    df.rename(columns={k: v for k, v in column_map.items() if k in df.columns}, inplace=True)
    return df

# === Data Loading ===

def extract_df_from_excel(uploaded_file):
    try:
        xls = pd.ExcelFile(uploaded_file, engine='openpyxl')
        for sheet_name in xls.sheet_names:
            sheet_df = xls.parse(sheet_name, header=None, nrows=10)
            for i in range(10):
                if sheet_df.iloc[i].astype(str).str.upper().str.contains("DATE").any():
                    df = xls.parse(sheet_name, skiprows=i)
                    return df
        return xls.parse(xls.sheet_names[0])
    except Exception:
        return None

def load_and_process_files(uploaded_files):
    dfs = []

    for uploaded_file in uploaded_files:
        df = extract_df_from_excel(uploaded_file)
        if df is not None:
            # Extract month from file name
            month_name = uploaded_file.name.split(" ")[-1].split(".")[0].replace(".", "")
            df['MONTH'] = month_name

            df = fix_duplicate_columns(df)
            df = standardize_columns(df)

            # Convert DATE to datetime
            if 'DATE' in df.columns:
                df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')

            for col in df.columns:
                if col != 'DATE' and df[col].dtype == 'object':
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            dfs.append(df)

    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        numeric_cols = combined_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        return combined_df, numeric_cols
    else:
        return pd.DataFrame(), []

# === Core Analysis Functions ===

def detect_anomalies(df, numeric_cols):
    iso = IsolationForest(contamination=0.05, random_state=42)
    df['ANOMALY'] = iso.fit_predict(df[numeric_cols].fillna(0))
    return df[df['ANOMALY'] == -1]

def run_classifier(df):
    if 'TOTAL_UNIT_(UPPCL+DG)' not in df.columns:
        return "TOTAL_UNIT_(UPPCL+DG) column not found."

    median = df['TOTAL_UNIT_(UPPCL+DG)'].median()
    y = (df['TOTAL_UNIT_(UPPCL+DG)'] > median).astype(int)

    X = df.select_dtypes(include=np.number).drop(columns=['TOTAL_UNIT_(UPPCL+DG)'], errors='ignore')
    X = X.fillna(X.mean())

    model = LogisticRegression()
    model.fit(X, y)
    preds = model.predict(X)
    return classification_report(y, preds)

def analyze_downtime(df):
    if 'PRODUCTION' not in df.columns or 'TOTAL_UNIT_(UPPCL+DG)' not in df.columns:
        return "Required columns not found."
    low_prod = df[df['PRODUCTION'] < df['PRODUCTION'].mean() * 0.5]
    low_energy = df[df['TOTAL_UNIT_(UPPCL+DG)'] < df['TOTAL_UNIT_(UPPCL+DG)'].mean() * 0.5]
    return f"Low Production Days: {len(low_prod)}\nLow Energy Days: {len(low_energy)}"

def analyze_correlation(df, numeric_cols):
    return df[numeric_cols].corr().round(2)

# === EDA Summary Function ===

def summarize_eda(df):
    eda_summary = ""

    # General info
    eda_summary += f"**Rows:** {df.shape[0]}\n\n"
    eda_summary += f"**Columns:** {df.shape[1]}\n\n"
    eda_summary += "**Missing Values (Top 10):**\n"
    eda_summary += df.isnull().sum().sort_values(ascending=False).head(10).to_string()
    eda_summary += "\n\n"

    # Dtypes
    eda_summary += "**Column Types:**\n"
    eda_summary += df.dtypes.value_counts().to_string()
    eda_summary += "\n\n"

    # Numeric summary
    num_cols = df.select_dtypes(include='number').columns.tolist()
    if num_cols:
        desc = df[num_cols].describe().T
        eda_summary += "**Numeric Summary:**\n"
        eda_summary += desc[['mean', 'std', 'min', 'max']].round(2).to_string()
        eda_summary += "\n\n"

    # Top categorical values
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    for col in cat_cols[:3]:
        top_vals = df[col].value_counts().head(3)
        eda_summary += f"**Top values in `{col}`:**\n{top_vals.to_string()}\n\n"

    return eda_summary

# === Monthly Analysis ===

def generate_monthly_insights(df, numeric_cols):
    months = df['MONTH'].unique()
    for month in sorted(months):
        st.subheader(f"📅 Month: {month}")
        month_df = df[df['MONTH'] == month]
        st.dataframe(month_df[numeric_cols].describe().T)
