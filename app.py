import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px

from data_utils import (
    load_and_process_files,
    detect_anomalies,
    run_classifier,
    analyze_downtime,
    analyze_correlation,
    summarize_eda,
    generate_monthly_insights
)
from time_series_utils import generate_forecast, decompose_and_plot
from gpt_utils import generate_section_insight
from insights_export import download_report_button
from advanced_modules import run_kmeans_visualization

st.set_page_config(layout="wide", page_title="Energy Dashboard")
st.title("⚡ Energy Dashboard")

uploaded = st.file_uploader("Upload Excel files (multiple)", type="xlsx", accept_multiple_files=True)
if not uploaded:
    st.info("📥 Upload files to begin.")
    st.stop()

with st.spinner("Processing..."):
    df, numeric_cols = load_and_process_files(uploaded)
    st.success(f"Loaded {len(uploaded)} file(s).")

# Sidebar toggles
toggles = {
    "Summary": True,
    "KPIs": True,
    "Trend": True,
    "Forecast": True,
    "Decomposition": True,
    "Anomalies": True,
    "Classifier": False,
    "Downtime": True,
    "Correlation": True,
    "EDA": True,
    "KMeans": True
}
for t in toggles:
    toggles[t] = st.sidebar.checkbox(f"Show {t}", value=toggles[t])

insights = []

if toggles["Summary"]:
    st.subheader("📈 Summary Statistics")
    summary = df[numeric_cols].describe()
    st.write(summary)
    insight = generate_section_insight("Summary Statistics", summary.to_string())
    st.markdown("**📝 Key insights:** " + insight)
    insights.append(("Summary Statistics", insight))

if toggles["KPIs"]:
    st.subheader("🎯 Energy KPIs")
    df["DG_UNIT"] = pd.to_numeric(df.get("DG_UNIT", pd.Series()), errors="coerce")
    df["TOTAL_UNIT_(UPPCL+DG)"] = pd.to_numeric(df.get("TOTAL_UNIT_(UPPCL+DG)", pd.Series()), errors="coerce")
    if "DG_UNIT" in df and "TOTAL_UNIT_(UPPCL+DG)" in df:
        total, dg = df["TOTAL_UNIT_(UPPCL+DG)"].sum(), df["DG_UNIT"].sum()
        share = 100*dg/total if total else 0
        st.metric("DG Share %", f"{share:.2f}%")
        insight = generate_section_insight("Energy KPIs", f"DG Share: {share:.2f}%")
        st.markdown("**📝 Key insights:** " + insight)
        insights.append(("Energy KPIs", insight))
    else:
        st.warning("Missing DG_UNIT or TOTAL_UNIT_(UPPCL+DG)")

if toggles["Trend"]:
    st.subheader("📊 Daily Trend")
    fig = px.line(df, x="DATE", y="TOTAL_UNIT_(UPPCL+DG)", color="MONTH")
    st.plotly_chart(fig, use_container_width=True)
    insight = generate_section_insight("Trend", "Daily energy consumption trend")
    st.markdown("**📝 Key insights:** " + insight)
    insights.append(("Trend", insight))

if toggles["Forecast"]:
    st.subheader("🔮 7‑Day Forecast")
    fig, vals = generate_forecast(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
        insight = generate_section_insight("Forecast", vals.to_string())
        st.markdown("**📝 Key insights:** " + insight)
        insights.append(("Forecast", insight))
    else:
        st.warning("Forecast failed.")

if toggles["Decomposition"]:
    st.subheader("🧪 Decomposition")
    fig = decompose_and_plot(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
        insight = generate_section_insight("Decomposition", "Trend/Seasonal/Residual plot")
        st.markdown("**📝 Key insights:** " + insight)
        insights.append(("Decomposition", insight))
    else:
        st.warning("Decomposition failed.")

if toggles["Anomalies"]:
    st.subheader("🚨 Anomaly Detection")
    ann = detect_anomalies(df, numeric_cols)
    st.write(ann[["DATE"] + numeric_cols[:5]])
    insight = generate_section_insight("Anomalies", ann.head().to_string())
    st.markdown("**📝 Key insights:** " + insight)
    insights.append(("Anomalies", insight))

if toggles["Classifier"]:
    st.subheader("⚠️ High Usage Classifier")
    report = run_classifier(df)
    st.text(report)
    insight = generate_section_insight("Classifier", report)
    st.markdown("**📝 Key insights:** " + insight)
    insights.append(("Classifier", insight))

if toggles["Downtime"]:
    st.subheader("📉 Downtime Summary")
    downtime = analyze_downtime(df)
    st.text(downtime)
    insight = generate_section_insight("Downtime", downtime)
    st.markdown("**📝 Key insights:** " + insight)
    insights.append(("Downtime", insight))

if toggles["Correlation"]:
    st.subheader("📊 Correlation Heatmap")
    corr = analyze_correlation(df, numeric_cols)
    st.dataframe(corr)
    fig = px.imshow(corr, text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
    insight = generate_section_insight("Correlation", corr.to_string())
    st.markdown("**📝 Key insights:** " + insight)
    insights.append(("Correlation", insight))

if toggles["EDA"]:
    st.subheader("🔍 Exploratory Data Analysis")
    st.text(summarize_eda(df))
    monthly = st.checkbox("Include monthly breakdown in EDA", value=True)
    if monthly:
        generate_monthly_insights(df, numeric_cols)

    insight = generate_section_insight("EDA", "EDA summary")
    st.markdown("**📝 Key insights:** " + insight)
    insights.append(("EDA", insight))

if toggles["KMeans"]:
    run_kmeans_visualization(df, numeric_cols)

download_report_button(insights)
