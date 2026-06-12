import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path

# CONFIG
st.set_page_config(page_title="Toyota Price Estimator", page_icon="🚗", layout="wide")

# LOAD CSS
css_path = Path("app/style.css")
if css_path.exists():
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# LOAD MODEL & DATA
model = joblib.load("models/toyota_price_model.pkl")
# Load data untuk visualisasi (Asumsi file ada di folder yang sama)
df = pd.read_csv("data/toyota.csv") 

# HIDE STREAMLIT
st.markdown("""<style>#MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}</style>""", unsafe_allow_html=True)

# HERO
st.markdown("""
<div class="hero-section">
    <h1>🚗 Toyota Used Car Price Prediction</h1>
    <p>Analisis cerdas harga mobil bekas dengan Machine Learning.</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 2], gap="large")

with left:
    with st.container():
        # Kolom logo
        st.image("logo.png", width=150)
        st.markdown("---")
        st.markdown("### 📊 Dashboard")
        st.metric("Total Data", f"{len(df):,}")
        st.metric("Akurasi Model", "96.78%")
        st.metric("RMSE", "£1,171")
        st.markdown("---")
        st.markdown("### ℹ️ Informasi")
        st.write("Sistem ini memprediksi harga berdasarkan data historis. Grafik di samping akan menampilkan perbandingan harga setelah Anda melakukan prediksi.")

with right:
    st.markdown('<div class="form-title">Spesifikasi Kendaraan</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        car_model = st.selectbox("Model Mobil", df['model'].unique())
        year = st.number_input("Tahun Produksi", 2000, 2025, 2020)
        transmission = st.selectbox("Transmisi", df['transmission'].unique())
        fuel_type = st.selectbox("Fuel Type", df['fuelType'].unique())
    with c2:
        mileage = st.number_input("Mileage", 0, 300000, 30000)
        tax = st.number_input("Tax (£)", 0, 1000, 145)
        mpg = st.number_input("MPG", 0.0, 100.0, 55.4)
        engine_size = st.number_input("Engine Size", 0.0, 5.0, 1.5)

    if st.button("🚀 Prediksi Harga Mobil", use_container_width=True):
        input_data = pd.DataFrame({"model": [car_model], "year": [year], "transmission": [transmission], 
                                   "mileage": [mileage], "fuelType": [fuel_type], "tax": [tax], 
                                   "mpg": [mpg], "engineSize": [engine_size]})
        prediction = model.predict(input_data)[0]

        # Hasil
        st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-label">Estimasi Harga Mobil</div>
            <div class="prediction-price">£{prediction:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

        # Visualisasi
        st.subheader("📈 Analisis Perbandingan")
        avg_price = df[df['model'] == car_model]['price'].mean()
        
        chart_data = pd.DataFrame({
            'Kategori': ['Harga Prediksi Anda', 'Rata-rata Model Terpilih'],
            'Harga': [prediction, avg_price]
        })
        
        fig = px.bar(chart_data, x='Kategori', y='Harga', color='Kategori', 
                     color_discrete_map={'Harga Prediksi Anda': '#64ffda', 'Rata-rata Model Terpilih': '#8892b0'})
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig, use_container_width=True)