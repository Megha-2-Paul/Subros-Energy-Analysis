# advanced_modules.py

import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import shap

def run_kmeans_visualization(df, numeric_cols):
    st.subheader("🔀 KMeans Clustering")
    try:
        km = KMeans(n_clusters=3, random_state=42)
        df['CLUSTER'] = km.fit_predict(df[numeric_cols].fillna(0))
        fig = px.scatter(df, x='DATE', y='TOTAL_UNIT_(UPPCL+DG)', color='CLUSTER', title="KMeans Clusters")
        st.plotly_chart(fig)
    except Exception as e:
        st.warning(f"KMeans failed: {e}")

def run_shap_explainer(df):
    st.subheader("🧠 SHAP Explainability")
    try:
        if all(col in df.columns for col in ['DG_UNIT', 'UPPCL_(UNIT_)', 'TOTAL_UNIT_(UPPCL+DG)']):
            X = df[['DG_UNIT', 'UPPCL_(UNIT_)']]
            y = df['TOTAL_UNIT_(UPPCL+DG)']
            model = RandomForestRegressor().fit(X, y)
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot(shap.plots.beeswarm(shap_values, show=False))
        else:
            st.warning("Missing required columns for SHAP.")
    except Exception as e:
        st.warning(f"SHAP failed: {e}")
