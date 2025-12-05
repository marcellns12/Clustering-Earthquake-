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
st.markdown("Aplikasi untuk menjelajahi data gempa yang telah dikelompokkan (clustered).")

# --- Muat Data ---
@st.cache_data
def load_data(file_path):
    """Memuat data dari file CSV."""
    try:
        data = pd.read_csv(file_path)
        # Konversi kolom cluster dan dbscan_cluster menjadi string agar mudah difilter
        for col in ['cluster', 'dbscan_cluster']:
            if col in data.columns:
                # Ganti nilai kosong dengan string 'NaN' atau 'None' sebelum konversi ke integer
                data[col] = data[col].fillna(-1).astype(int).astype(str)
                data[col] = data[col].replace('-1', 'N/A')
        return data
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        return pd.DataFrame()

# Pastikan nama file sesuai dengan yang Anda unggah
FILE_NAME = "gempa_clustered_full.csv"
df = load_data(FILE_NAME)

if not df.empty and 'latitude' in df.columns and 'longitude' in df.columns:

    # --- Sidebar untuk Filter ---
    st.sidebar.header("Opsi Filter Data")

    # Ambil semua nilai unik (termasuk 'N/A') untuk cluster
    cluster_options = ['Semua'] + sorted([c for c in df['cluster'].unique() if c != 'N/A']) + ['N/A']
    selected_cluster = st.sidebar.selectbox("Filter berdasarkan 'cluster':", cluster_options)

    dbscan_options = ['Semua'] + sorted([c for c in df['dbscan_cluster'].unique() if c != 'N/A']) + ['N/A']
    selected_dbscan = st.sidebar.selectbox("Filter berdasarkan 'dbscan_cluster':", dbscan_options)

    # Terapkan Filter
    filtered_df = df.copy()
    if selected_cluster != 'Semua':
        filtered_df = filtered_df[filtered_df['cluster'] == selected_cluster]

    if selected_dbscan != 'Semua':
        filtered_df = filtered_df[filtered_df['dbscan_cluster'] == selected_dbscan]

    st.sidebar.markdown(f"**Jumlah Data Setelah Filter:** {len(filtered_df)}")
    st.sidebar.markdown(f"**Total Data Awal:** {len(df)}")


    # --- Bagian Utama ---

    col1, col2 = st.columns(2)

    with col1:
        st.header("Peta Persebaran Gempa")
        st.markdown("Visualisasi lokasi gempa berdasarkan Longitude dan Latitude.")

        # Buat kolom warna berdasarkan cluster yang paling detail (dbscan_cluster)
        color_col = 'dbscan_cluster' if 'dbscan_cluster' in filtered_df.columns else 'cluster'
        
        if 'latitude' in filtered_df.columns and 'longitude' in filtered_df.columns:
            # Peta interaktif menggunakan Plotly Scatter Mapbox
            fig_map = px.scatter_mapbox(
                filtered_df,
                lat="latitude",
                lon="longitude",
                color=color_col if color_col in filtered_df.columns else None, # Beri warna berdasarkan cluster
                hover_name="cluster", # Tampilkan cluster pada hover
                hover_data={
                    "mag": True,
                    "depth": True,
                    "latitude": ':.2f',
                    "longitude": ':.2f',
                    "cluster": True,
                    "dbscan_cluster": True
                },
                color_continuous_scale=px.colors.cyclical.IceFire,
                zoom=2,
                height=600,
                title="Persebaran Gempa Berdasarkan Lokasi dan Cluster"
            )

            # Set Mapbox style (Anda mungkin perlu token Mapbox, tapi 'carto-positron' seringkali bekerja tanpa)
            fig_map.update_layout(mapbox_style="carto-positron")
            fig_map.update_layout(margin={"r":0,"t":30,"l":0,"b":0})

            st.plotly_chart(fig_map, use_container_width=True)

    with col2:
        st.header("Distribusi Magnitudo dan Kedalaman")

        # Visualisasi Distribusi Magnitudo (mag)
        if 'mag' in filtered_df.columns:
            st.subheader("Histogram Magnitudo (mag)")
            fig_mag = px.histogram(
                filtered_df.dropna(subset=['mag']), # Hapus NaN di mag
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
                filtered_df.dropna(subset=['depth']), # Hapus NaN di depth
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
    st.error("Data tidak dapat dimuat atau tidak memiliki kolom 'latitude' atau 'longitude' yang diperlukan untuk pemetaan.")

