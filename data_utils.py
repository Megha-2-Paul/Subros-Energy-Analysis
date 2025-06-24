import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import streamlit as st

def fix_duplicate_columns(df):
    seen = {}
    new = []
    for c in df.columns:
        b = str(c).strip().upper().replace(" ", "_").replace("%", "PERCENT")
        new.append(b + f"_{seen[b]}" if b in seen else b)
        seen[b] = seen.get(b, 0) + 1
    df.columns = new
    return df

def standardize_columns(df):
    m = {"DG_POWER":"DG_UNIT","DG_(UNIT)":"DG_UNIT",
         "TOTAL":"TOTAL_UNIT_(UPPCL+DG)",
         "UPPCL_POWER":"UPPCL_(UNIT_)","UPPCL_(UNIT)":"UPPCL_(UNIT_)"}
    return df.rename(columns={k:v for k,v in m.items() if k in df})

def extract_df_from_excel(f):
    try:
        x = pd.ExcelFile(f, engine="openpyxl")
        for sheet in x.sheet_names:
            hdr = x.parse(sheet, header=None, nrows=8)
            for i in range(len(hdr)):
                if hdr.iloc[i].astype(str).str.contains("DATE", case=False).any():
                    return x.parse(sheet, skiprows=i)
        return x.parse(x.sheet_names[0])
    except:
        return None

def load_and_process_files(files):
    df_list = []
    for f in files:
        d = extract_df_from_excel(f)
        if d is None: continue
        m = f.name.rsplit(" ",1)[-1].split(".")[0]
        d["MONTH"] = m
        d = fix_duplicate_columns(d)
        d = standardize_columns(d)
        if "DATE" in d:
            d["DATE"] = pd.to_datetime(d["DATE"], errors="coerce")
        for c in d.columns:
            if c!="DATE":
                d[c] = pd.to_numeric(d[c], errors="coerce")
        df_list.append(d)
    if not df_list: return pd.DataFrame(), []
    df = pd.concat(df_list, ignore_index=True)
    num = df.select_dtypes(include=np.number).columns.tolist()
    return df, num

def detect_anomalies(df, numeric_cols):
    df = df.copy()
    iso = IsolationForest(contamination=0.05, random_state=42)
    df["ANOMALY"] = iso.fit_predict(df[numeric_cols].fillna(0))
    return df[df["ANOMALY"]==-1]

def run_classifier(df):
    if "TOTAL_UNIT_(UPPCL+DG)" not in df:
        return "Column missing"
    y = (df["TOTAL_UNIT_(UPPCL+DG)"] > df["TOTAL_UNIT_(UPPCL+DG)"].median()).astype(int)
    X = df.select_dtypes(include=np.number).drop(columns=["TOTAL_UNIT_(UPPCL+DG)"])
    X = X.fillna(X.mean())
    m = LogisticRegression(max_iter=1000)
    m.fit(X,y)
    return classification_report(y, m.predict(X))

def analyze_downtime(df):
    if "PRODUCTION" not in df or "TOTAL_UNIT_(UPPCL+DG)" not in df:
        return "Required columns missing"
    p, t = df["PRODUCTION"], df["TOTAL_UNIT_(UPPCL+DG)"]
    return f"Low Prod Days: {(p<p.mean()*0.5).sum()}\nLow Energy Days: {(t<t.mean()*0.5).sum()}"

def analyze_correlation(df, numeric_cols):
    return df[numeric_cols].corr().round(2)

def summarize_eda(df):
    txt = f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n"
    miss = df.isnull().sum().sort_values(ascending=False).head(5)
    txt += "Missing (top 5):\n" + miss.to_string() + "\n"
    txt += "Numeric summary:\n" + df.select_dtypes(include=np.number).describe().T[["mean","std","min","max"]].round(2).to_string()
    return txt

def generate_monthly_insights(df, numeric_cols):
    for m in sorted(df["MONTH"].unique()):
        st.subheader(f"🔹 Month: {m}")
        st.dataframe(df[df["MONTH"]==m][numeric_cols].describe().round(2).T)
