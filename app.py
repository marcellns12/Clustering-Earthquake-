# app.py
import streamlit as st
import pandas as pd
from sklearn.cluster import DBSCAN
import pydeck as pdk

st.set_page_config(page_title="DBSCAN Clustering Gempa", layout="wide")
st.title("DBSCAN Clustering - Dataset Gempa")

# Sidebar: parameter DBSCAN
st.sidebar.header("Parameter DBSCAN")
eps = st.sidebar.slider("Eps", 0.1, 5.0, 0.5)
min_samples = st.sidebar.slider("Min Samples", 1, 20, 5)

# Upload CSV lengkap (latitude, longitude, magnitude, depth, dsb)
uploaded_file = st.file_uploader("Upload CSV dataset gempa", type="csv")

if uploaded_file is not None:
    gempa_sample = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(gempa_sample.head(10))

    # Pilih kolom numerik untuk clustering
    numeric_cols = gempa_sample.select_dtypes(include=['float64', 'int64']).columns
    X = gempa_sample[numeric_cols].fillna(0)  # handle NaN

    # Fit DBSCAN
    if st.button("Fit DBSCAN dan Simpan CSV"):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        gempa_sample['dbscan_cluster'] = dbscan.fit_predict(X)

        # Simpan hasil CSV
        output_csv = "gempa_clustered.csv"
        gempa_sample.to_csv(output_csv, index=False)
        st.success(f"Hasil clustering disimpan ke '{output_csv}'")

        # Distribusi cluster
        st.subheader("Distribusi Cluster")
        st.write(gempa_sample['dbscan_cluster'].value_counts())

        # Rata-rata fitur per cluster
        st.subheader("Rata-rata Fitur per Cluster")
        st.dataframe(gempa_sample.groupby('dbscan_cluster')[['magnitude','depth']].mean())

        # Visualisasi peta
        if 'latitude' in gempa_sample.columns and 'longitude' in gempa_sample.columns:
            st.subheader("Visualisasi Peta Gempa Berdasarkan Cluster")
            # Tambahkan warna untuk tiap cluster
            gempa_sample['color'] = gempa_sample['dbscan_cluster'].apply(
                lambda x: (int((x+1)*50)%256, int((x+2)*80)%256, int((x+3)*110)%256) if x != -1 else (0,0,0)
            )

            # Peta sederhana
            st.map(gempa_sample[['latitude','longitude']])

            # Peta interaktif pydeck
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=gempa_sample,
                get_position='[longitude, latitude]',
                get_fill_color='color',
                get_radius=5000,
                pickable=True
            )
            view_state = pdk.ViewState(
                latitude=gempa_sample['latitude'].mean(),
                longitude=gempa_sample['longitude'].mean(),
                zoom=4,
                pitch=0
            )
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))
        else:
            st.warning("Dataset tidak memiliki kolom 'latitude' dan 'longitude'")

else:
    st.warning("Silakan upload file CSV dataset gempa terlebih dahulu!")
