import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load saved models
knn = joblib.load("knn_model.pkl")
pca_2 = joblib.load("pca_model.pkl")
scaler = joblib.load("scaler_model.pkl")

# App Title
st.title("🍷 Wine Classification App (PCA + KNN)")

# Create layout columns
left_col, right_col = st.columns(2)

# Sidebar inputs (left column)
with left_col:
    st.sidebar.header("Input Wine Features")
    Alcohol = st.sidebar.slider("Alcohol", 10.0, 15.0, 13.0, step=0.1)
    Malic_acid = st.sidebar.slider("Malic acid", 0.7, 5.0, 2.5, step=0.1)
    Ash = st.sidebar.slider("Ash", 1.5, 3.5, 2.0, step=0.1)
    Alcalinity_of_ash = st.sidebar.slider("Alcalinity of ash", 10.0, 30.0, 17.0, step=0.5)
    Magnesium = st.sidebar.slider("Magnesium", 70.0, 160.0, 100.0, step=1.0)
    Total_phenols = st.sidebar.slider("Total phenols", 0.9, 4.0, 2.5, step=0.1)
    Flavanoids = st.sidebar.slider("Flavanoids", 0.3, 5.5, 3.0, step=0.1)
    Nonflavanoid_phenols = st.sidebar.slider("Nonflavanoid phenols", 0.0, 1.0, 0.3, step=0.01)
    Proanthocyanins = st.sidebar.slider("Proanthocyanins", 0.0, 4.0, 1.5, step=0.1)
    Color_intensity = st.sidebar.slider("Color intensity", 1.0, 13.0, 5.0, step=0.1)
    Hue = st.sidebar.slider("Hue", 0.4, 1.8, 1.0, step=0.01)
    OD280_OD315 = st.sidebar.slider("OD280/OD315 of diluted wines", 1.2, 4.0, 2.5, step=0.1)
    Proline = st.sidebar.slider("Proline", 300.0, 1700.0, 800.0, step=10.0)

# Prediction in right column
with right_col:
    if st.button("Predict"):
        input_features = np.array([[
            Alcohol, Malic_acid, Ash, Alcalinity_of_ash, Magnesium,
            Total_phenols, Flavanoids, Nonflavanoid_phenols,
            Proanthocyanins, Color_intensity, Hue, OD280_OD315, Proline
        ]])

        # Scale, transform with PCA, and predict
        scaled_input = scaler.transform(input_features)
        pca_input = pca_2.transform(scaled_input)
        prediction = knn.predict(pca_input)[0]

        # Display result
        st.subheader("🎯 Prediction Result")
        st.success(f"Predicted Wine Class: **{prediction}**")
