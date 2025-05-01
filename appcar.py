import streamlit as st
import joblib
import numpy as np

# Load trained model and label encoders
model = joblib.load("car_price_model.pkl")
label_encoders = joblib.load("labelcars.pkl")

brand_to_models =joblib.load("brandmodel.pkl")
# Get dropdown options
brand_options = label_encoders['Brand'].classes_
model_options = label_encoders['Model'].classes_
fuel_options = label_encoders['Fuel_Type'].classes_
trans_options = label_encoders['Transmission'].classes_

# UI
st.title("Car Price Predictor")
brand_names = list(brand_to_models.keys())
selected_brand = st.selectbox("Select Brand", brand_names)

# Step 2: Show only models for the selected brand
model_options = brand_to_models[selected_brand]
selected_model = st.selectbox("Select Model", model_options)
#brand = st.selectbox("Choose a Brand", list(brand_to_models.keys()))

# Dynamically update models for selected brand
#model_list = brand_to_models.get(brand, [])
#model_name = st.selectbox("Choose a Model", model_list)

#brand = st.selectbox("Brand", brand_options)
#model_name = st.selectbox("Model", model_options)
fuel_type = st.selectbox("Fuel Type", fuel_options)
transmission = st.selectbox("Transmission", trans_options)

year = st.number_input("Year", min_value=1990, max_value=2025, value=2020)
engine_size = st.number_input("Engine Size (L)", min_value=0.0, step=0.1, value=2.0)
mileage = st.number_input("Mileage", min_value=0, step=1000, value=50000)
doors = st.number_input("Number of Doors", min_value=2, max_value=5, value=4)
owners = st.number_input("Owner Count", min_value=0, max_value=10, value=1)

# Encode categorical values
brand_encoded = label_encoders['Brand'].transform([selected_brand])[0]
model_encoded = label_encoders['Model'].transform([selected_model])[0]
fuel_encoded = label_encoders['Fuel_Type'].transform([fuel_type])[0]
trans_encoded = label_encoders['Transmission'].transform([transmission])[0]

# Feature array
features = np.array([[brand_encoded, model_encoded, year, engine_size,
                      fuel_encoded, trans_encoded, mileage, doors, owners]])

# Predict
if st.button("Predict Price"):
    predicted_price = model.predict(features)[0]
    st.success(f"Estimated Price: ${predicted_price:,.0f}")
