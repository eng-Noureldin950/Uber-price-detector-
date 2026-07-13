import streamlit as st
import folium
import requests
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os

from database import PredictionLogger

st.title("Uber Price Predictor")
db_logger = PredictionLogger()

# Load trained ML model and scaler using absolute path resolution
@st.cache_resource
def load_assets():
    # Get the absolute path of the directory containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    model_path = os.path.join(base_dir, 'uber_xgb_model.pkl')
    scaler_path = os.path.join(base_dir, 'coordinates_scaler.pkl')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

try:
    model, scaler = load_assets()
    assets_loaded = True
except Exception as e:
    st.warning("Model or Scaler files not found. Please run 'eda_uber.py' first.")
    assets_loaded = False

# Calculate cyclical time features for the model
def get_time_features():
    now = datetime.now()
    hour = now.hour
    day = now.day
    day_of_week = now.weekday()
    month = now.month
    
    # Sine and cosine transformations
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_of_week_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_of_week_cos = np.cos(2 * np.pi * day_of_week / 7)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    return day, hour_sin, hour_cos, day_of_week_sin, day_of_week_cos, month_sin, month_cos

# Initialize session state for map clicks
if "points" not in st.session_state:
    st.session_state.points = []

if "last_clicked_stored" not in st.session_state:
    st.session_state.last_clicked_stored = None

# Create map centered around NYC
m = folium.Map(location=[40.7128, -74.0060], zoom_start=12)

# Draw current markers
for p in st.session_state.points:
    folium.Marker(p).add_to(m)

# Process route when two points are selected
if len(st.session_state.points) == 2:
    p1 = st.session_state.points[0] # Pickup [lat, lon]
    p2 = st.session_state.points[1] # Dropoff [lat, lon]
    
    url = (
        f"https://router.project-osrm.org/route/v1/driving/"
        f"{p1[1]},{p1[0]};{p2[1]},{p2[0]}"
        "?overview=full&geometries=geojson"
    )
    try:
        data = requests.get(url).json()
        if "routes" in data and len(data["routes"]) > 0:
            route = data["routes"][0]
            distance = route["distance"] / 1000 # meters to km
            time_min = route["duration"] / 60   # seconds to minutes
            
            st.success(f"📍 Road Distance: {distance:.2f} km")
            st.success(f"Estimated Time: {time_min:.1f} minutes")
            
            # Plot route line on the map
            points = [[c[1], c[0]] for c in route["geometry"]["coordinates"]]
            folium.PolyLine(points, color="blue", weight=5).add_to(m)
            
            # Feature engineering and inference
            if assets_loaded:
                coords_df = pd.DataFrame([{
                    'pickup_longitude': p1[1],
                    'pickup_latitude': p1[0],
                    'dropoff_longitude': p2[1],
                    'dropoff_latitude': p2[0]
                }])
                
                # Apply scaler
                scaled_coords = scaler.transform(coords_df)[0]
                day, h_sin, h_cos, dow_sin, dow_cos, m_sin, m_cos = get_time_features()
                
                # Construct feature vector
                feature_vector = np.array([[
                    scaled_coords[0], 
                    scaled_coords[1], 
                    scaled_coords[2], 
                    scaled_coords[3], 
                    1, # Default passenger count
                    distance, 
                    day,
                    h_sin, h_cos,
                    dow_sin, dow_cos,
                    m_sin, m_cos
                ]])
                
                # Predict fare
                predicted_fare = model.predict(feature_vector)[0]
                st.info(f" Estimated Uber fare is : ${predicted_fare:.2f}")
                
                # Log request to database
                db_logger.log_request(p1[0], p1[1], p2[0], p2[1], distance, predicted_fare)
                st.caption("Saved prediction request to the database module.")
            
        else:
            st.warning("Could not find a valid driving route between these points.")
    except Exception as e:
        st.error(f"Error fetching route or predicting: {e}")

# Render map
output = st_folium(m, width=700, height=500, key="uber_map")

# Handle click events
if output and output.get("last_clicked"):
    current_click = [output["last_clicked"]["lat"], output["last_clicked"]["lng"]]
    
    if current_click != st.session_state.last_clicked_stored:
        st.session_state.last_clicked_stored = current_click  
        
        if len(st.session_state.points) >= 2:
            st.session_state.points = []  
            
        st.session_state.points.append(current_click)
        st.rerun()