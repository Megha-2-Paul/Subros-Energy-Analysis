import pandas as pd
import streamlit as st
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

st.set_page_config(layout="wide", page_title="Energy Dashboard - Subros")
st.title("⚡ Subros Energy Analysis Dashboard")

uploaded_files = st.file_uploader("📁 Upload Excel files (Jan–May 2025)", type='xlsx', accept_multiple_files=True)

if uploaded_files:
    with st.spinner("📊 Processing uploaded files..."):
        df, numeric_cols = load_and_process_files(uploaded_files)
        st.success(f"✅ Loaded {len(uploaded_files)} file(s) successfully.")

    # Sidebar options
    st.sidebar.header("📊 Chart Toggle Options")
    show_summary = st.sidebar.checkbox("Show Summary Stats", True)
    show_energy = st.sidebar.checkbox("Show Energy KPIs", True)
    show_chart = st.sidebar.checkbox("Energy Trend Chart", True)
    show_forecast = st.sidebar.checkbox("Forecast Next 7 Days", True)
    show_anomalies = st.sidebar.checkbox("Anomaly Detection", True)
    show_classifier = st.sidebar.checkbox("High Usage Classifier", False)
    show_loadfactor = st.sidebar.checkbox("Load Factor & Downtime", True)
    show_correlation = st.sidebar.checkbox("Cross-Section Correlation", True)
    show_decomposition = st.sidebar.checkbox("Time Series Decomposition", True)
    show_eda = st.sidebar.checkbox("Run Exploratory Data Analysis", True)
    show_kmeans = st.sidebar.checkbox("KMeans Clustering", True)

    all_insights = []

    if show_summary:
        st.subheader("📈 Summary Statistics")
        summary_stats = df[numeric_cols].describe()
        st.write(summary_stats)
        insight = generate_section_insight("Summary Statistics", summary_stats.to_string())
        st.markdown("**📝 Summary:** " + insight)
        all_insights.append(("Summary Statistics", insight))

    if show_energy:
        st.subheader("📍 Key Energy KPIs")
        df['DG_UNIT'] = pd.to_numeric(df['DG_UNIT'], errors='coerce')
        df['TOTAL_UNIT_(UPPCL+DG)'] = pd.to_numeric(df['TOTAL_UNIT_(UPPCL+DG)'], errors='coerce')
        if 'DG_UNIT' in df.columns and 'TOTAL_UNIT_(UPPCL+DG)' in df.columns:
            total = df['TOTAL_UNIT_(UPPCL+DG)'].sum()
            dg = df['DG_UNIT'].sum()
            share = (dg / total) * 100 if total else 0
            st.metric("DG Share %", f"{share:.2f}%")
            insight = generate_section_insight("Energy KPIs", f"DG Share: {share:.2f}%")
            st.markdown("**📝 Summary:** " + insight)
            all_insights.append(("Energy KPIs", insight))
        else:
            st.warning("Required energy columns not found.")

    if show_chart:
        st.subheader("📊 Daily Energy Trend")
        fig = px.line(df, x='DATE', y='TOTAL_UNIT_(UPPCL+DG)', color='MONTH', title="Daily Energy Consumption")
        st.plotly_chart(fig, use_container_width=True)
        insight = generate_section_insight("Energy Trend", "Line chart showing daily TOTAL_UNIT_(UPPCL+DG)")
        st.markdown("**📝 Summary:** " + insight)
        all_insights.append(("Energy Trend", insight))

    if show_forecast:
        st.subheader("🔮 ARIMA Forecast (7 Days)")
        forecast_fig, forecast_values = generate_forecast(df)
        if forecast_fig:
            st.plotly_chart(forecast_fig, use_container_width=True)
            insight = generate_section_insight("Forecast", str(forecast_values))
            st.markdown("**📝 Summary:** " + insight)
            all_insights.append(("Forecast", insight))
        else:
            st.warning("Forecasting failed.")

    if show_decomposition:
        st.subheader("🧪 Time Series Decomposition")
        decompose_fig = decompose_and_plot(df)
        if decompose_fig:
            st.plotly_chart(decompose_fig, use_container_width=True)
            insight = generate_section_insight("Time Series Decomposition", "Trend/Seasonality plotted")
            st.markdown("**📝 Summary:** " + insight)
            all_insights.append(("Decomposition", insight))

    if show_anomalies:
        st.subheader("🕵️ Anomaly Detection")
        anomalies_df = detect_anomalies(df, numeric_cols)
        st.write(anomalies_df[['DATE'] + numeric_cols[:5]])
        insight = generate_section_insight("Anomalies", anomalies_df.head().to_string())
        st.markdown("**📝 Summary:** " + insight)
        all_insights.append(("Anomaly Detection", insight))

    if show_classifier:
        st.subheader("⚠️ High Usage Classifier")
        clf_report = run_classifier(df)
        st.text(clf_report)
        insight = generate_section_insight("High Usage Classifier", clf_report)
        st.markdown("**📝 Summary:** " + insight)
        all_insights.append(("Classifier", insight))

    if show_loadfactor:
        st.subheader("📉 Load Factor & Downtime")
        downtime = analyze_downtime(df)
        st.text(downtime)
        insight = generate_section_insight("Load Factor & Downtime", downtime)
        st.markdown("**📝 Summary:** " + insight)
        all_insights.append(("Downtime", insight))

    if show_correlation:
        st.subheader("📊 Correlation Analysis")
        corr_df = analyze_correlation(df, numeric_cols)
        st.dataframe(corr_df)
        fig = px.imshow(corr_df, text_auto=True, aspect="auto", title="Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)
        insight = generate_section_insight("Correlation", corr_df.to_string())
        st.markdown("**📝 Summary:** " + insight)
        all_insights.append(("Correlation", insight))

    if show_eda:
        st.subheader("📊 Exploratory Data Analysis (EDA)")
        with st.expander("📈 Descriptive Statistics"):
            st.dataframe(df[numeric_cols].describe().T)

        with st.expander("📊 Monthly Trend Summary"):
            for month in sorted(df['MONTH'].unique()):
                month_df = df[df['MONTH'] == month]
                st.markdown(f"### 🔹 {month}")
                st.dataframe(month_df[numeric_cols].describe().T)

        with st.expander("🧠 EDA Summary Text"):
            eda_text = summarize_eda(df)
            st.text(eda_text)

        insight = generate_section_insight("EDA", "Exploratory analysis of the dataset")
        st.markdown("**📝 Summary:** " + insight)
        all_insights.append(("EDA", insight))

    view_option = st.radio("Choose Analysis View", ["Overall", "Monthly"], horizontal=True)
    if view_option == "Overall":
        st.subheader("📊 Overall Analysis")
        st.dataframe(df[numeric_cols].describe().T)
    else:
        st.subheader("📆 Monthly Analysis")
        generate_monthly_insights(df, numeric_cols)

    if show_kmeans:
        run_kmeans_visualization(df, numeric_cols)

    download_report_button(all_insights)

else:
    st.info("📥 Please upload Excel file(s) to begin analysis.")
