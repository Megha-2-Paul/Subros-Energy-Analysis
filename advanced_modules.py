# advanced_modules.py

import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

def run_kmeans_visualization(df, numeric_cols):
    st.subheader("🔀 KMeans Clustering")
    try:
        km = KMeans(n_clusters=3, random_state=42)
        df['CLUSTER'] = km.fit_predict(df[numeric_cols].fillna(0))
        fig = px.scatter(df, x='DATE', y='TOTAL_UNIT_(UPPCL+DG)', color='CLUSTER', title="KMeans Clusters")
        st.plotly_chart(fig)
    except Exception as e:
        st.warning(f"KMeans failed: {e}")
