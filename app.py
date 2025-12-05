import streamlit as st
import pandas as pd
import plotly.express as px

# Konfigurasi halaman
st.set_page_config(
    page_title="Visualisasi Data Gempa Bumi",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Judul Aplikasi ---
st.title("Visualisasi Data Gempa Bumi 🌍")
st.markdown("Unggah file CSV Anda (`gempa_clustered_full.csv`) untuk memulai visualisasi.")

# --- Sidebar untuk Unggah dan Filter ---
st.sidebar.header("Unggah File Data")
uploaded_file = st.sidebar.file_uploader(
    "Pilih file CSV gempa Anda (harus memiliki kolom latitude, longitude, mag, depth, cluster, dbscan_cluster)",
    type=['csv']
)

df = pd.DataFrame()
if uploaded_file is not None:
    # Memuat Data dari file yang diunggah
    try:
        # Menampilkan indikator loading saat memuat data
        with st.spinner("Memuat dan memproses data..."):
            df = pd.read_csv(uploaded_file)

            # Konversi kolom cluster dan dbscan_cluster menjadi string agar mudah difilter
            # Mengganti nilai kosong (NaN) dengan string 'N/A'
            for col in ['cluster', 'dbscan_cluster']:
                if col in df.columns:
                    # Mengisi NaN dengan -1 (sementara), konversi ke int, lalu ke str
                    df[col] = df[col].fillna(-1).astype(int).astype(str)
                    df[col] = df[col].replace('-1', 'N/A')

    except Exception as e:
        st.error(f"Gagal memproses file. Pastikan format CSV sudah benar. Error: {e}")
        st.stop()

# Hanya jalankan visualisasi jika DataFrame sudah terisi dan memiliki kolom wajib
if not df.empty and all(col in df.columns for col in ['latitude', 'longitude']):

    st.sidebar.header("Opsi Filter Data")

    # Ambil semua nilai unik (termasuk 'N/A') untuk cluster
    cluster_options = ['Semua'] + sorted(df['cluster'].unique().tolist())
    selected_cluster = st.sidebar.selectbox("Filter berdasarkan 'cluster':", cluster_options)

    dbscan_options = ['Semua'] + sorted(df['dbscan_cluster'].unique().tolist())
    selected_dbscan = st.sidebar.selectbox("Filter berdasarkan 'dbscan_cluster':", dbscan_options)

    # Terapkan Filter
    filtered_df = df.copy()
    if selected_cluster != 'Semua':
        filtered_df = filtered_df[filtered_df['cluster'] == selected_cluster]

    if selected_dbscan != 'Semua':
        filtered_df = filtered_df[filtered_df['dbscan_cluster'] == selected_dbscan]

    st.sidebar.markdown(f"**Jumlah Data Setelah Filter:** {len(filtered_df)}")
    st.sidebar.markdown(f"**Total Data Awal:** {len(df)}")

    # ---------------------------------------------
    # --- Bagian Utama: Visualisasi ---
    # ---------------------------------------------
    
    col1, col2 = st.columns(2)

    with col1:
        st.header("Peta Persebaran Gempa")
        
        # Tentukan kolom warna
        color_col = 'dbscan_cluster' if 'dbscan_cluster' in filtered_df.columns else 'cluster'
        
        # Peta interaktif menggunakan Plotly Scatter Mapbox
        fig_map = px.scatter_mapbox(
            filtered_df,
            lat="latitude",
            lon="longitude",
            color=color_col if color_col in filtered_df.columns else None,
            hover_name="cluster",
            hover_data={
                "mag": True,
                "depth": True,
                "latitude": ':.2f',
                "longitude": ':.2f',
                "cluster": True,
                "dbscan_cluster": True
            },
            zoom=2,
            height=600,
            title="Persebaran Gempa Berdasarkan Lokasi dan Cluster"
        )

        fig_map.update_layout(mapbox_style="carto-positron")
        fig_map.update_layout(margin={"r":0,"t":30,"l":0,"b":0})

        st.plotly_chart(fig_map, use_container_width=True)

    with col2:
        st.header("Distribusi Magnitudo dan Kedalaman")

        # Visualisasi Distribusi Magnitudo (mag)
        if 'mag' in filtered_df.columns:
            st.subheader("Histogram Magnitudo (mag)")
            fig_mag = px.histogram(
                filtered_df.dropna(subset=['mag']),
                x="mag",
                nbins=20,
                title="Distribusi Magnitudo Gempa"
            )
            st.plotly_chart(fig_mag, use_container_width=True)
        else:
            st.warning("Kolom 'mag' tidak ditemukan.")

        # Visualisasi Distribusi Kedalaman (depth)
        if 'depth' in filtered_df.columns:
            st.subheader("Histogram Kedalaman (depth)")
            fig_depth = px.histogram(
                filtered_df.dropna(subset=['depth']),
                x="depth",
                nbins=20,
                title="Distribusi Kedalaman Gempa"
            )
            st.plotly_chart(fig_depth, use_container_width=True)
        else:
            st.warning("Kolom 'depth' tidak ditemukan.")

    # --- Tampilan Data Mentah ---
    st.header("Data Gempa (Tabel)")
    st.markdown("Tampilan data yang telah difilter.")
    st.dataframe(filtered_df)

else:
    # Pesan yang muncul jika file belum diunggah
    st.info("Silakan unggah file CSV Anda di sidebar untuk menampilkan visualisasi.")
    st.markdown("""
        **Petunjuk:**
        1.  Klik **'Browse files'** di sidebar.
        2.  Pilih file **`gempa_clustered_full.csv`** Anda.
        3.  Visualisasi akan muncul secara otomatis.
    """)
