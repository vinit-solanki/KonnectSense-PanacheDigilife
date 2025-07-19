import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained XGBoost model
with open("xgboost_aqi_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the scaler
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Define the UI
st.set_page_config(page_title="AQI Prediction", layout="centered")
st.title("🌍 Air Quality Index (AQI) Prediction App")
st.write("Enter pollutant values to predict AQI for the next day.")

# Sidebar for user input
st.sidebar.header("🔹 Enter Pollutant Values")

# Location selection
location = st.sidebar.selectbox(
    "Select a location:",
    ["Bandra Kurla Complex", "Kurla", "Colaba", "Chhatrapati Shivaji Intl Airport"],
)

# User input fields for pollutants
pm25 = st.sidebar.number_input("PM2.5 (µg/m³)", min_value=0.0, max_value=500.0, step=0.1, value=50.0)
pm10 = st.sidebar.number_input("PM10 (µg/m³)", min_value=0.0, max_value=500.0, step=0.1, value=80.0)
no2 = st.sidebar.number_input("NO2 (µg/m³)", min_value=0.0, max_value=500.0, step=0.1, value=40.0)
so2 = st.sidebar.number_input("SO2 (µg/m³)", min_value=0.0, max_value=500.0, step=0.1, value=20.0)
co = st.sidebar.number_input("CO (mg/m³)", min_value=0.0, max_value=50.0, step=0.1, value=0.5)
o3 = st.sidebar.number_input("O3 (µg/m³)", min_value=0.0, max_value=500.0, step=0.1, value=30.0)

# Convert input into a NumPy array and scale
input_data = np.array([[pm25, pm10, no2, so2, co, o3]])
input_scaled = scaler.transform(input_data)

# Predict AQI
if st.button("Predict AQI 🚀"):
    predicted_aqi = model.predict(input_scaled)[0]
    
    # Display prediction
    st.subheader(f"Predicted AQI: {predicted_aqi:.2f}")
    
    # AQI Category Display
    if predicted_aqi <= 50:
        st.success("Good (0-50) ✅")
    elif predicted_aqi <= 100:
        st.info("Moderate (51-100) 😐")
    elif predicted_aqi <= 150:
        st.warning("Unhealthy for Sensitive Groups (101-150) ⚠️")
    elif predicted_aqi <= 200:
        st.warning("Unhealthy (151-200) 🚨")
    elif predicted_aqi <= 300:
        st.error("Very Unhealthy (201-300) ❌")
    else:
        st.error("Hazardous (301+) ☠️")

# Footer
st.write("📍 **Location Selected:**", location)
st.write("🔍 Model trained on Mumbai AQI data.")
st.markdown("🔗 *Developed using Streamlit & XGBoost*")