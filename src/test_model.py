import joblib
import pandas as pd

# Load model
model = joblib.load(
    "models/toyota_price_model.pkl"
)

# Data contoh
sample = pd.DataFrame({
    "model": ["Yaris"],
    "year": [2018],
    "transmission": ["Manual"],
    "mileage": [30000],
    "fuelType": ["Petrol"],
    "tax": [145],
    "mpg": [55.4],
    "engineSize": [1.5]
})

# Prediksi
prediction = model.predict(sample)

print(f"Prediksi harga: £{prediction[0]:,.2f}")