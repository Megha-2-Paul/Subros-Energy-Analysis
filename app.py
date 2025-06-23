import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, r2_score
from sklearn.cluster import KMeans
from statsmodels.tsa.arima.model import ARIMA
import shap
import base64
from datetime import datetime
from io import BytesIO

st.set_page_config(layout="wide", page_title="Energy Dashboard - Subros")
st.title("⚡ Subros Energy Analysis Dashboard")

uploaded_files = st.file_uploader("Upload Excel files (Jan–May 2025, include 'Production' column if available)", type='xlsx', accept_multiple_files=True)

def extract_df_from_excel(uploaded_file):
    try:
        xls = pd.ExcelFile(uploaded_file, engine='openpyxl')
        for sheet_name in xls.sheet_names:
            sheet_df = xls.parse(sheet_name, header=None, nrows=10)
            for i in range(10):
                if sheet_df.iloc[i].astype(str).str.upper().str.contains("DATE").any():
                    df = xls.parse(sheet_name, skiprows=i)
                    return df
        return xls.parse(xls.sheet_names[0])  # fallback
    except Exception as e:
        st.warning(f"Error reading {uploaded_file.name}: {e}")
        return None

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

if uploaded_files:
    dfs = []
    with st.spinner("📊 Processing uploaded files..."):
        for uploaded_file in uploaded_files:
            df = extract_df_from_excel(uploaded_file)
            if df is not None:
                month_name = uploaded_file.name.split(" ")[-1].split(".")[0].replace(".", "")
                df['MONTH'] = month_name
                df = fix_duplicate_columns(df)
                df = standardize_columns(df)
                if 'DATE' in df.columns:
                    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
                    dfs.append(df)

    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        df = df[df['DATE'].notna()]
        df.replace(['#DIV/0!', '#REF!'], np.nan, inplace=True)
        df.fillna(method='ffill', inplace=True)
        df.fillna(0, inplace=True)

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        df[numeric_cols] = df[numeric_cols].clip(lower=0)
        df = df[~(df[numeric_cols] > df[numeric_cols].mean() + 3 * df[numeric_cols].std()).any(axis=1)]

        st.sidebar.header("📅 Filter Options")
        selected_months = st.sidebar.multiselect("Select Month(s)", sorted(df['MONTH'].unique()), default=list(df['MONTH'].unique()))
        filtered_df = df[df['MONTH'].isin(selected_months)].copy()

        st.subheader("📈 Summary Statistics")
        st.write(filtered_df[numeric_cols].describe())

        st.subheader("📍 Key Performance Indicators")
        if 'DG_UNIT' in filtered_df.columns and 'TOTAL_UNIT_(UPPCL+DG)' in filtered_df.columns:
            total_energy = filtered_df['TOTAL_UNIT_(UPPCL+DG)'].sum()
            dg_energy = filtered_df['DG_UNIT'].sum()
            dg_share = (dg_energy / total_energy) * 100
            st.metric("DG Share %", f"{dg_share:.2f}%", delta=f"{dg_share - 30:.1f}%")
            if dg_share > 30:
                st.error("⚠️ High DG usage! Consider alternatives like battery backup or off-peak scheduling.")

        if 'PF' in filtered_df.columns:
            pf_min = filtered_df['PF'].min()
            if pf_min < 0.9:
                st.warning(f"⚠️ Power Factor low: {pf_min:.2f} (Target > 0.90)")

        if 'PRODUCTION' in filtered_df.columns and 'TOTAL_UNIT_(UPPCL+DG)' in filtered_df.columns:
            energy_per_unit = filtered_df['TOTAL_UNIT_(UPPCL+DG)'].sum() / filtered_df['PRODUCTION'].sum()
            st.metric("Energy per Unit Production", f"{energy_per_unit:.2f} kWh/unit")

            st.subheader("📌 Energy vs Production")
            correlation = filtered_df['PRODUCTION'].corr(filtered_df['TOTAL_UNIT_(UPPCL+DG)'])
            st.info(f"Correlation: {correlation:.2f}")
            model = LinearRegression().fit(filtered_df[['PRODUCTION']], filtered_df['TOTAL_UNIT_(UPPCL+DG)'])
            st.write(f"R² Score: {r2_score(filtered_df['TOTAL_UNIT_(UPPCL+DG)'], model.predict(filtered_df[['PRODUCTION']])):.2f}")

        st.subheader("🔌 Heatmap: Top Energy Consumers")
        top_cols = [col for col in numeric_cols if 'TOTAL_KWH' in col.upper()]
        if top_cols:
            top10 = filtered_df[top_cols].mean().sort_values(ascending=False).head(10)
            fig = px.bar(x=top10.index, y=top10.values, labels={"x": "Panel/Machine", "y": "Avg kWh"}, title="Top 10 Energy-Consuming Panels")
            st.plotly_chart(fig)

        st.subheader("📈 Forecasting Future Energy Use")
        try:
            ts = filtered_df.set_index('DATE')['TOTAL_UNIT_(UPPCL+DG)'].resample('D').sum().fillna(method='ffill')
            model = ARIMA(ts, order=(1, 1, 1)).fit()
            forecast = model.forecast(steps=7)
            fig = px.line(x=forecast.index, y=forecast, title="Next 7 Days Energy Forecast", labels={"x": "Date", "y": "kWh"})
            st.plotly_chart(fig)
        except Exception as e:
            st.warning(f"Forecasting failed: {e}")

        st.subheader("🕵️ Anomaly Detection")
        try:
            iso = IsolationForest(contamination=0.05, random_state=42)
            filtered_df['ANOMALY'] = iso.fit_predict(filtered_df[numeric_cols])
            anomalies = filtered_df[filtered_df['ANOMALY'] == -1]
            st.write(f"Detected {len(anomalies)} anomalies.")
            csv = anomalies.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="anomaly_report.csv">📥 Download Anomaly Report</a>'
            st.markdown(href, unsafe_allow_html=True)
        except:
            st.warning("Anomaly detection failed.")

        # st.subheader("🔍 SHAP Explainability (Random Forest)")
        # try:
        #     if all(col in filtered_df.columns for col in ['DG_UNIT', 'UPPCL_(UNIT_)', 'TOTAL_UNIT_(UPPCL+DG)']):
        #         X = filtered_df[['DG_UNIT', 'UPPCL_(UNIT_)']]
        #         y = filtered_df['TOTAL_UNIT_(UPPCL+DG)']
        #         rf = RandomForestRegressor().fit(X, y)
        #         explainer = shap.Explainer(rf, X)
        #         shap_values = explainer(X)
        #         st.set_option('deprecation.showPyplotGlobalUse', False)
        #         st.pyplot(shap.plots.beeswarm(shap_values, show=False))
        # except:
        #     st.warning("SHAP explanation failed.")

        st.subheader("🔀 KMeans Clustering")
        try:
            km = KMeans(n_clusters=3, random_state=42)
            filtered_df['CLUSTER'] = km.fit_predict(filtered_df[numeric_cols].fillna(0))
            fig = px.scatter(filtered_df, x='DATE', y='TOTAL_UNIT_(UPPCL+DG)', color='CLUSTER')
            st.plotly_chart(fig)
        except:
            st.warning("KMeans clustering failed.")

        st.subheader("⚠️ High Usage Classifier")
        try:
            y_class = (filtered_df['TOTAL_UNIT_(UPPCL+DG)'] > filtered_df['TOTAL_UNIT_(UPPCL+DG)'].median()).astype(int)
            clf = LogisticRegression()
            clf.fit(X, y_class.loc[X.index])
            preds = clf.predict(X)
            st.write(pd.DataFrame(classification_report(y_class.loc[X.index], preds, output_dict=True)).T)
        except:
            st.warning("Classifier training failed.")

        st.subheader("📤 Export Cleaned Data")
        if st.button("Download Cleaned CSV"):
            export_csv = filtered_df.to_csv(index=False)
            b64 = base64.b64encode(export_csv.encode()).decode()
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            href = f'<a href="data:file/csv;base64,{b64}" download="subros_cleaned_{now}.csv">📥 Download CSV</a>'
            st.markdown(href, unsafe_allow_html=True)

        st.subheader("📅 Monthly Insights")
        for month in sorted(filtered_df['MONTH'].unique()):
            st.markdown(f"### 📂 {month} Summary")
            month_df = filtered_df[filtered_df['MONTH'] == month]
            st.write(month_df.describe())
            try:
                fig_month = px.line(month_df, x='DATE', y='TOTAL_UNIT_(UPPCL+DG)', title=f"Energy Trend - {month}")
                st.plotly_chart(fig_month)
            except:
                st.warning(f"Could not plot energy trend for {month}.")
    else:
        st.error("❌ No valid data loaded.")
else:
    st.info("📥 Upload one or more Excel files to begin.")
