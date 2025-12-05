import streamlit as st
import pandas as pd
import plotly.express as px

# Konfigurasi halaman
st.set_page_config(
    page_title="Visualisasi Data Gempa Bumi",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Visualisasi Data Gempa Bumi 🌍")
st.markdown("Visualisasi menggunakan koordinat dunia nyata berdasarkan `latitude` dan `longitude` serta hasil clustering (`cluster`, `dbscan_cluster`).")

# Sidebar untuk upload
st.sidebar.header("Unggah File Data")
uploaded_file = st.sidebar.file_uploader(
    "Pilih file CSV gempa Anda",
    type=['csv']
)

df = pd.DataFrame()
if uploaded_file is not None:
    try:
        with st.spinner("Memuat dan memproses data..."):
            df = pd.read_csv(uploaded_file)

            # Pastikan cluster menjadi string dan NaN -> 'N/A'
            for col in ['cluster', 'dbscan_cluster']:
                if col in df.columns:
                    df[col] = df[col].fillna(-1).astype(int).astype(str)
                    df[col] = df[col].replace('-1', 'N/A')

    except Exception as e:
        st.error(f"Gagal memproses file. Pastikan format CSV sudah benar. Error: {e}")
        st.stop()

if not df.empty and all(col in df.columns for col in ['latitude', 'longitude']):

    st.sidebar.header("Opsi Filter Data")
    cluster_options = ['Semua'] + sorted(df['cluster'].unique().tolist())
    selected_cluster = st.sidebar.selectbox("Filter 'cluster' (K-Means):", cluster_options)

    dbscan_options = ['Semua'] + sorted(df['dbscan_cluster'].unique().tolist())
    selected_dbscan = st.sidebar.selectbox("Filter 'dbscan_cluster' (DBSCAN):", dbscan_options)

    filtered_df = df.copy()
    if selected_cluster != 'Semua':
        filtered_df = filtered_df[filtered_df['cluster'] == selected_cluster]
    if selected_dbscan != 'Semua':
        filtered_df = filtered_df[filtered_df['dbscan_cluster'] == selected_dbscan]

    st.sidebar.markdown(f"**Jumlah Data Setelah Filter:** {len(filtered_df)}")
    st.sidebar.markdown(f"**Total Data Awal:** {len(df)}")

    # --- Peta dunia nyata ---
    st.header("1. Peta Persebaran Gempa Bumi Dunia Nyata")
    
    color_col = 'dbscan_cluster' if 'dbscan_cluster' in filtered_df.columns else 'cluster'

    fig_map = px.scatter_mapbox(
        filtered_df,
        lat="latitude",
        lon="longitude",
        color=color_col,
        hover_name="cluster",
        hover_data={
            "latitude": ':.2f',
            "longitude": ':.2f',
            "cluster": True,
            "dbscan_cluster": True,
            "mag": True if 'mag' in filtered_df.columns else False,
            "depth": True if 'depth' in filtered_df.columns else False
        },
        zoom=2,
        height=800,
        title="Distribusi Gempa Bumi Berdasarkan Cluster di Dunia Nyata"
    )

    fig_map.update_layout(mapbox_style="open-street-map")
    fig_map.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

    # --- Distribusi Cluster (Bar Chart) ---
    st.header("2. Distribusi Frekuensi Cluster")
    col_bar1, col_bar2 = st.columns(2)

    with col_bar1:
        if 'cluster' in filtered_df.columns:
            st.subheader("Hitungan per Cluster (K-Means)")
            cluster_counts = filtered_df['cluster'].value_counts().reset_index()
            cluster_counts.columns = ['Cluster', 'Count']
            fig_cluster = px.bar(cluster_counts, x='Cluster', y='Count', title="Distribusi Cluster K-Means")
            st.plotly_chart(fig_cluster, use_container_width=True)

    with col_bar2:
        if 'dbscan_cluster' in filtered_df.columns:
            st.subheader("Hitungan per Cluster (DBSCAN)")
            dbscan_counts = filtered_df['dbscan_cluster'].value_counts().reset_index()
            dbscan_counts.columns = ['DBSCAN_Cluster', 'Count']
            fig_dbscan = px.bar(dbscan_counts, x='DBSCAN_Cluster', y='Count', title="Distribusi Cluster DBSCAN")
            st.plotly_chart(fig_dbscan, use_container_width=True)

    # --- Data Mentah ---
    st.header("3. Data Mentah (Tabel)")
    st.dataframe(filtered_df)

else:
    st.info("Silakan unggah file CSV Anda di sidebar untuk menampilkan visualisasi.")
