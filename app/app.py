import streamlit as st
import pandas as pd
import joblib

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================

st.set_page_config(
    page_title="Toyota Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load("models/toyota_price_model.pkl")

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🚗 Toyota Price Predictor")

st.sidebar.markdown("""
### Tentang Aplikasi

Aplikasi ini digunakan untuk memprediksi harga mobil Toyota bekas menggunakan model Machine Learning.

### Algoritma
- Random Forest Regressor

### Dataset
- Toyota Used Car Dataset

### Evaluasi Model
- RMSE: 1171.01
- R² Score: 0.9678
""")

# ==========================================
# HEADER
# ==========================================

st.title("🚗 Toyota Used Car Price Prediction")

st.markdown("""
Prediksi harga mobil Toyota bekas berdasarkan spesifikasi kendaraan menggunakan model **Machine Learning Random Forest Regressor**.

Silakan masukkan data kendaraan pada form di bawah ini.
""")

st.divider()

# ==========================================
# FORM INPUT
# ==========================================

st.subheader("📋 Spesifikasi Kendaraan")

col1, col2 = st.columns(2)

with col1:

    car_model = st.selectbox(
        "Model Mobil",
        [
            " Auris",
            " Avensis",
            " Aygo",
            " C-HR",
            " Camry",
            " Corolla",
            " GT86",
            " Hilux",
            " IQ",
            " Land Cruiser",
            " PROACE VERSO",
            " Prius",
            " RAV4",
            " Supra",
            " Urban Cruiser",
            " Verso",
            " Verso-S",
            " Yaris"
        ]
    )

    year = st.number_input(
        "Tahun Produksi",
        min_value=2000,
        max_value=2025,
        value=2018
    )

    transmission = st.selectbox(
        "Jenis Transmisi",
        [
            "Automatic",
            "Manual",
            "Other",
            "Semi-Auto"
        ]
    )

    fuel_type = st.selectbox(
        "Jenis Bahan Bakar",
        [
            "Diesel",
            "Hybrid",
            "Other",
            "Petrol"
        ]
    )

with col2:

    mileage = st.number_input(
        "Mileage",
        min_value=0,
        value=30000,
        step=1000
    )

    tax = st.number_input(
        "Tax",
        min_value=0,
        value=145
    )

    mpg = st.number_input(
        "MPG",
        min_value=0.0,
        value=55.4,
        step=0.1
    )

    engine_size = st.number_input(
        "Engine Size",
        min_value=0.5,
        value=1.5,
        step=0.1
    )

st.divider()

# ==========================================
# PREDIKSI
# ==========================================

if st.button("🔍 Prediksi Harga", use_container_width=True):

    input_data = pd.DataFrame({
        "model": [car_model],
        "year": [year],
        "transmission": [transmission],
        "mileage": [mileage],
        "fuelType": [fuel_type],
        "tax": [tax],
        "mpg": [mpg],
        "engineSize": [engine_size]
    })

    prediction = model.predict(input_data)

    st.success("Prediksi berhasil dilakukan!")

    st.metric(
        label="💰 Estimasi Harga Mobil",
        value=f"£{prediction[0]:,.2f}"
    )

    st.info(
        "Hasil prediksi merupakan estimasi berdasarkan pola yang dipelajari model dari dataset Toyota Used Car Dataset."
    )

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "Machine Learning Project | Prediksi Harga Mobil Toyota | Python • Scikit-Learn • Streamlit"
)