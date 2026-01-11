# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. LOAD THE BRAIN
@st.cache_resource
def load_brain():
    return joblib.load('titan_brain.pkl')

brain = load_brain()

# 2. THE SIDEBAR (Inputs)
st.sidebar.header("🏡 Property Details")

# Location Sliders (Melbourne Coordinates)
lat = st.sidebar.slider("Latitude", -38.0, -37.5, -37.81)
lon = st.sidebar.slider("Longitude", 144.5, 145.5, 144.96)

# House Details
rooms = st.sidebar.slider("Rooms", 1, 6, 3)
bath = st.sidebar.slider("Bathrooms", 1, 4, 1)
car = st.sidebar.slider("Car Spots", 0, 3, 1)
land = st.sidebar.number_input("Landsize (sqm)", 100, 2000, 400)
build = st.sidebar.number_input("Building Area (sqm)", 50, 1000, 150)
year = st.sidebar.slider("Year Built", 1900, 2024, 1990)
dist = st.sidebar.slider("Distance from CBD (km)", 2.0, 15.0, 5.0)
region = st.sidebar.selectbox("Region", ["Region_0", "Region_1", "Region_2"])

# 3. MAIN DASHBOARD
st.title("🏙️ Real Estate Titan")
st.write("### 📍 Location Visualizer")

# --- 🗺️ THE NEW MAP FEATURE ---
# We create a dataframe with just the current Lat/Lon
# Streamlit detects 'lat' and 'lon' columns automatically
map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})
st.map(map_data, zoom=10)
# -----------------------------

# 4. PREDICTION LOGIC
if st.button("💰 Predict Value"):
    # Create Input DataFrame (Must match training order)
    input_data = pd.DataFrame({
        'Rooms': [rooms],
        'Distance': [dist],
        'Bedroom2': [rooms],
        'Bathroom': [bath],
        'Car': [car],
        'Landsize': [land],
        'BuildingArea': [build],
        'YearBuilt': [year],
        'Lattitude': [lat],
        'Longtitude': [lon],
        'Regionname': [region]
    })
    
    # Preprocessing
    input_data['Regionname'] = brain['le_region'].transform(input_data['Regionname'])
    input_data['BuildingArea'] = brain['imputer'].transform(input_data[['BuildingArea']])
    input_data['Location_Cluster'] = brain['kmeans'].predict(input_data[['Lattitude', 'Longtitude']])
    
    # Scaling
    X_input = input_data[brain['feature_names']]
    X_scaled = brain['scaler'].transform(X_input)
    
    # Inference
    price_pred = brain['regressor'].predict(X_scaled)[0]
    tier_idx = brain['classifier'].predict(X_scaled)[0]
    tier_pred = brain['le_tier'].inverse_transform([tier_idx])[0]
    
    # Results Display
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Estimated Price", value=f"${price_pred:,.2f}")
    
    with col2:
        if tier_pred == 'High':
            st.success("💎 Luxury Tier")
        elif tier_pred == 'Mid':
            st.info("🏠 Mid-Market Tier")
        else:
            st.warning("📉 Budget Tier")
            
    st.caption(f"AI Logic: This property belongs to Neighborhood Cluster #{input_data['Location_Cluster'][0]}")