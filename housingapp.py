
import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("house_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="House Price Prediction", page_icon="🏡")

st.title("🏡 House Price Prediction App")
st.write("Predict the **Price of a House** based on features using Linear Regression.")

# Input fields
area = st.number_input("Area (sq.ft):", min_value=500, max_value=10000, step=100)
bedrooms = st.number_input("Number of Bedrooms:", min_value=1, max_value=10, step=1)
bathrooms = st.number_input("Number of Bathrooms:", min_value=1, max_value=10, step=1)
stories = st.number_input("Number of Stories:", min_value=1, max_value=5, step=1)
parking = st.number_input("Parking spaces:", min_value=0, max_value=5, step=1)

# Prediction
if st.button("Predict Price"):
    input_data = [[area, bedrooms, bathrooms, stories, parking]]
    prediction = model.predict(input_data)
    st.success(f"Predicted House Price: ₹ {prediction[0]:,.2f}")

# Bulk prediction
st.subheader("📂 Upload dataset for bulk prediction")
uploaded_file = st.file_uploader("Upload CSV with columns: Area, Bedrooms, Bathrooms, Stories, Parking", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", data.head())
    preds = model.predict(data[["Area", "Bedrooms", "Bathrooms", "Stories", "Parking"]])
    data["PredictedPrice"] = preds
    st.write("Predictions:", data)
    st.download_button("Download Predictions", data.to_csv(index=False), "house_predictions.csv")