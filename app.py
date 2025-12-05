# app.py
import streamlit as st
import pandas as pd
from sklearn.cluster import DBSCAN
import joblib
import os

st.set_page_config(page_title="DBSCAN Clustering Gempa", layout="wide")
st.title("DBSCAN Clustering - Dataset Gempa")

# Path untuk menyimpan model
MODEL_PATH = "dbscan_model.pkl"

# Sidebar: parameter DBSCAN
st.sidebar.header("Parameter DBSCAN")
eps = st.sidebar.slider("Eps", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
min_samples = st.sidebar.slider("Min Samples", min_value=1, max_value=20, value=5)

# Upload CSV dataset
uploaded_file = st.file_uploader("Upload CSV dataset gempa", type="csv")

if uploaded_file is not None:
    # Load dataset dari file yang diupload
    gempa_sample = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(gempa_sample.head(10))

    # Tombol fit DBSCAN
    if st.button("Fit DBSCAN"):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        gempa_sample['dbscan_cluster'] = dbscan.fit_predict(gempa_sample)

        # Simpan model
        joblib.dump(dbscan, MODEL_PATH)
        st.success(f"DBSCAN selesai di-fit dan model tersimpan sebagai '{MODEL_PATH}'")

        # Tampilkan distribusi cluster
        st.subheader("Distribusi Cluster")
        st.write(gempa_sample['dbscan_cluster'].value_counts())

        # Tampilkan rata-rata fitur per cluster
        st.subheader("Rata-rata Fitur per Cluster")
        st.dataframe(gempa_sample.groupby('dbscan_cluster').mean())

else:
    st.warning("Silakan upload file CSV dataset gempa terlebih dahulu!")

# Opsi load model lama
if st.sidebar.checkbox("Load model DBSCAN sebelumnya"):
    if os.path.exists(MODEL_PATH):
        dbscan_loaded = joblib.load(MODEL_PATH)
        st.success("Model DBSCAN berhasil diload dari file")
        st.write("Cluster dari dataset terakhir (jika dataset sama dengan sebelumnya):")
        st.write(dbscan_loaded.labels_)
    else:
        st.error(f"Tidak ada file model '{MODEL_PATH}' ditemukan")
