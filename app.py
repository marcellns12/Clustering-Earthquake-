# app.py
import streamlit as st
import pandas as pd
from sklearn.cluster import DBSCAN
import joblib

st.set_page_config(page_title="DBSCAN Clustering Gempa", layout="wide")

st.title("DBSCAN Clustering - Dataset Gempa")

# Load dataset
DATA_PATH = "gempa_sample.csv"
MODEL_PATH = "model_clustering_optimal.pkl"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_or_fit_dbscan(df, eps=0.5, min_samples=5):
    try:
        dbscan = joblib.load(MODEL_PATH)
        df['dbscan_cluster'] = dbscan.labels_
    except:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        df['dbscan_cluster'] = dbscan.fit_predict(df)
        joblib.dump(dbscan, MODEL_PATH)
    return df, dbscan

# Load data
gempa_sample = load_data()

# Sidebar: user bisa ubah parameter DBSCAN
st.sidebar.header("Parameter DBSCAN")
eps = st.sidebar.slider("Eps", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
min_samples = st.sidebar.slider("Min Samples", min_value=1, max_value=20, value=5)

# Fit/load DBSCAN
gempa_sample, dbscan = load_or_fit_dbscan(gempa_sample, eps=eps, min_samples=min_samples)

# Tampilkan hasil cluster
st.subheader("Distribusi Cluster")
st.write(gempa_sample['dbscan_cluster'].value_counts())

st.subheader("10 Data Pertama")
st.dataframe(gempa_sample.head(10))

st.subheader("Rata-rata Fitur per Cluster")
st.dataframe(gempa_sample.groupby('dbscan_cluster').mean())
