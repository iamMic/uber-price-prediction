import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. Haversine Function (To turn coordinates into distance)
def haversine(lon1, lat1, lon2, lat2):
    r = 6371 # Earth's radius in km
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return r * c

# 2. Load the trained model
# Make sure your model file is named 'uber_model.pkl' in the same folder
model = joblib.load('lrmodel.pkl')

# 3. Streamlit UI Design
st.set_page_config(page_title="Uber Fare Predictor", page_icon="🚗")
st.title("🚗 Uber Price Prediction Engine")
st.markdown("Estimate your trip cost based on location and time.")

# 4. Input Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Pickup Location")
    p_lat = st.number_input("Pickup Latitude", value=6.5244) # Default: Lagos
    p_lon = st.number_input("Pickup Longitude", value=3.3792)

with col2:
    st.subheader("Dropoff Location")
    d_lat = st.number_input("Dropoff Latitude", value=6.4531) # Default: Lekki
    d_lon = st.number_input("Dropoff Longitude", value=3.4682)

st.divider()

# Time and Passenger inputs
col3, col4 = st.columns(2)
with col3:
    hour = st.slider("Hour of Day (24hr)", 0, 23, 12)
with col4:
    passengers = st.selectbox("Number of Passengers", [1, 2, 3, 4, 5, 6])

# 5. Prediction Logic
if st.button("Calculate Estimated Fare", use_container_width=True):
    # Calculate distance from inputs
    distance = haversine(p_lon, p_lat, d_lon, d_lat)
    
    # Prepare features for the model (matching the order in your training script)
    # [distance_km, hour, day_of_week, passenger_count]
    # Assuming day_of_week is Friday (4) for this example
    input_data = np.array([[distance, hour, 4, passengers]]) 
    
    prediction = model.predict(input_data)
    
    # 6. Display Results
    st.success(f"### Estimated Fare: ₦{prediction[0]:,.2f}")
    st.info(f"Calculated Distance: {distance:.2f} km")