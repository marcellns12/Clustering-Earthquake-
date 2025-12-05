import streamlit as st
import pandas as pd
import joblib
import numpy as np
import folium
from streamlit_folium import folium_static

# --- Konfigurasi Berkas Model ---
SCALER_FILE = 'scaler.pkl'
MODEL_FILE = 'model_clustering_optimal.pkl'
FEATURES = ['latitude', 'longitude']

# --- Fungsi untuk Memuat Model dan Scaler ---
@st.cache_resource
def load_assets():
    """Memuat model clustering dan scaler yang telah disimpan."""
    try:
        # Memuat scaler
        scaler = joblib.load(SCALER_FILE)
        # Memuat model
        model = joblib.load(MODEL_FILE)
        return scaler, model
    except FileNotFoundError:
        st.error(f"Berkas model/scaler tidak ditemukan. Pastikan {SCALER_FILE} dan {MODEL_FILE} tersedia.")
        st.stop()
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

# --- Main App ---
def main():
    st.title("🌎 Prediksi Klaster Lokasi Gempa (Clustering)")
    st.markdown("Aplikasi untuk memprediksi klaster lokasi gempa berdasarkan Latitude dan Longitude.")

    # Memuat aset
    scaler, model = load_assets()

    st.sidebar.header("Input Lokasi Gempa")
    
    # Input dari pengguna
    latitude = st.sidebar.number_input("Latitude", min_value=-90.0, max_value=90.0, value=38.0, step=0.01)
    longitude = st.sidebar.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-120.0, step=0.01)

    # Tombol Prediksi
    if st.sidebar.button("Prediksi Klaster"):
        
        # 1. Siapkan data input
        input_data = pd.DataFrame([[latitude, longitude]], columns=FEATURES)
        
        # 2. Scaling data input
        input_scaled = scaler.transform(input_data)
        
        # 3. Prediksi klaster
        # DBSCAN menggunakan .fit_predict(), tapi untuk prediksi data baru
        # pada model DBSCAN/KMeans/Agglomerative, umumnya kita menggunakan fungsi ini
        # untuk DBSCAN murni, data baru mungkin akan dilabeli -1 jika tidak ada Core/Border point terdekat.
        # Untuk K-Means/Agglomerative, kita pakai .predict()
        
        # Karena model clustering tidak memiliki metode .predict() yang universal, 
        # kita asumsikan model Anda adalah K-Means/Agglomerative.
        # Jika Anda menggunakan DBSCAN, Anda harus menggunakan logic jarak terdekat
        # ke centroid/cluster point atau asumsikan bahwa model DBSCAN Anda 
        # sudah dikonversi menjadi model klasifikasi yang dapat memprediksi.
        
        try:
            cluster_label = model.predict(input_scaled)[0]
        except AttributeError:
            st.warning("Model DBSCAN/Agglomerative standar mungkin tidak memiliki metode `.predict()` untuk data baru. Akan menggunakan metode prediksi alternatif jika memungkinkan.")
            # Untuk DBSCAN, prediksi data baru lebih kompleks. Untuk K-Means/Agglomerative, .predict() harusnya bekerja.
            cluster_label = "Tidak dapat diprediksi" # Placeholder

        # Coba lagi dengan asumsi K-Means/Agglomerative
        if cluster_label == "Tidak dapat diprediksi":
             # Menggunakan logika K-Means/Agglomerative
            from scipy.spatial.distance import cdist
            
            # Asumsi: model DBSCAN terbaik tidak disimpan. Jika menggunakan DBSCAN,
            # Anda harus melatih model klasifikasi terpisah untuk memprediksi.
            # Kita akan mengasumsikan model yang dimuat adalah K-Means atau model yang memiliki 'cluster_centers_'.
            
            if hasattr(model, 'cluster_centers_'):
                # Untuk K-Means
                distances = cdist(input_scaled, model.cluster_centers_)
                cluster_label = np.argmin(distances)
            elif 'DBSCAN' in str(type(model)):
                cluster_label = "DBSCAN: Prediksi untuk data baru tidak langsung, memerlukan klasifikasi terpisah."
            else:
                 cluster_label = f"Klaster {model.predict(input_scaled)[0]}"
        else:
             cluster_label = f"Klaster {cluster_label}"
             
        # 4. Tampilkan Hasil
        st.subheader("Hasil Prediksi")
        st.metric(label="Klaster Prediksi", value=cluster_label)
        st.balloons()
        
        # 5. Visualisasi Peta
        st.subheader("Visualisasi Lokasi")
        m = folium.Map(location=[latitude, longitude], zoom_start=6)
        
        # Menambahkan marker lokasi input
        folium.Marker(
            [latitude, longitude], 
            tooltip="Lokasi Input",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
        
        # Tampilkan peta
        folium_static(m)

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Aplikasi ini menggunakan Model: {MODEL_FILE}")

if __name__ == "__main__":
    main()
