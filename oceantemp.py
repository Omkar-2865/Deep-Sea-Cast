import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

# Load and preprocess dataset
df = pd.read_csv("Ocean_Temperature.csv")
df.drop(columns=["datetime"], inplace=True)

LE = LabelEncoder()
df["ocean_name"] = LE.fit_transform(df["ocean_name"])

x = df.iloc[:, 0:-1]
y = df["temperature_c"]

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(xtrain, ytrain)

# Evaluate model
ypred = model.predict(xtest)
r2 = r2_score(ytest, ypred)

# Ocean name options
OceanName_L = ['Arctic Ocean', 'Atlantic Ocean', 'Indian Ocean', 'Other', 'Pacific Ocean', 'Southern Ocean']

# Streamlit UI
st.title("🌊 Deep Sea Cast")
st.write("Enter oceanic parameters below to predict the ocean temperature (°C).")

# Input fields
OceanName = st.selectbox("Select Ocean", OceanName_L)
latitude = st.number_input("Latitude", format="%.6f")
longitude = st.number_input("Longitude", format="%.6f")
depth_m = st.number_input("Depth (m)", format="%.2f")
salinity_psu = st.number_input("Salinity (PSU)", format="%.2f")
ph = st.number_input("pH", format="%.2f")
chlorophyll_mg_m3 = st.number_input("Chlorophyll (mg/m³)", format="%.2f")
nitrate_umol_l = st.number_input("Nitrate (μmol/L)", format="%.2f")
phosphate_umol_l = st.number_input("Phosphate (μmol/L)", format="%.2f")
oxygen_ml_l = st.number_input("Oxygen (ml/L)", format="%.2f")
current_speed_m_s = st.number_input("Current Speed (m/s)", format="%.2f")
wind_speed_m_s = st.number_input("Wind Speed (m/s)", format="%.2f")

# Prediction
if st.button("Predict Temperature"):
    try:
        OceanName_F = OceanName_L.index(OceanName)
        features = np.array([[OceanName_F, latitude, longitude, depth_m, salinity_psu, ph,
                              chlorophyll_mg_m3, nitrate_umol_l, phosphate_umol_l, oxygen_ml_l,
                              current_speed_m_s, wind_speed_m_s]])
        pred = model.predict(features)
        st.success(f"🌡️ Predicted Ocean Temperature: {pred[0]:.2f} °C")
        st.info(f"📈 Model R-squared Accuracy: {r2:.4f}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
