import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# 1. Load the trained model
# Ensure 'lrmodel.pkl' is in the same folder as this script
try:
    model = joblib.load('lrmodel.pkl')
except FileNotFoundError:
    st.error("Model file 'lrmodel.pkl' not found. Please check the file path.")

# 2. Streamlit UI Design
st.set_page_config(page_title="Uber Fare Predictor", page_icon="🚗")
st.title("🚗 Uber Price Prediction Engine")
st.markdown("Enter trip details below to estimate the fare.")

# 3. Input Layout - Coordinates
st.subheader("📍 Location Coordinates")
col1, col2 = st.columns(2)

with col1:
    p_lon = st.number_input("Pickup Longitude", value=3.3792, format="%.6f")
    p_lat = st.number_input("Pickup Latitude", value=6.5244, format="%.6f")

with col2:
    d_lon = st.number_input("Dropoff Longitude", value=3.4682, format="%.6f")
    d_lat = st.number_input("Dropoff Latitude", value=6.4531, format="%.6f")

st.divider()

# 4. Input Layout - Trip Details
st.subheader("📅 Trip Details")
col3, col4, col5 = st.columns(3)

with col3:
    passengers = st.number_input("Passengers", min_value=1, max_value=6, value=1)
with col4:
    hour = st.slider("Pickup Hour (24h)", 0, 23, datetime.now().hour)
with col5:
    day = st.selectbox("Day of Week", 
                      options=[0, 1, 2, 3, 4, 5, 6], 
                      format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])

# 5. Prediction Logic
if st.button("Calculate Estimated Fare", use_container_width=True):
    try:
        # Features must be in a 2D array: [[feature1, feature2, ...]]
        features = np.array([[p_lon, p_lat, d_lon, d_lat, passengers, hour, day]])
        
        # Make the prediction
        prediction = model.predict(features)
        
        # THE FIX: Flatten the output and convert the first element to a float
        # This handles nested arrays like [[value]], Series, or single-item lists
        fare = float(np.array(prediction).ravel()[0])
        
        # 6. Display Results
        if fare < 0:
            st.warning("The model predicted a negative fare. Please check your coordinate inputs.")
        else:
            st.success(f"### Estimated Fare: ₦{fare:,.2f}")
        
        # Add a map for visual appeal
        map_df = pd.DataFrame({
            'lat': [p_lat, d_lat],
            'lon': [p_lon, d_lon]
        })
        st.map(map_df)
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.info("Technical Note: This usually happens if the model output shape doesn't match the expected scalar format.")
