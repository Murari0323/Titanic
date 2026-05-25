import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_ann_model():
    model = tf.keras.models.load_model("titanic_survival_ann_model.keras")
    return model

@st.cache_resource
def load_scaler():
    scaler = joblib.load("scaler_minmax.pkl")
    return scaler

model = load_ann_model()
scaler = load_scaler()

# ---------------- TITLE ----------------
st.title("🚢 Titanic Survival Prediction")
st.subheader("Deep Learning ANN Model")

st.markdown("""
This application predicts whether a passenger survived the Titanic disaster
using an Artificial Neural Network (ANN).
""")

st.divider()

# ---------------- INPUT SECTION ----------------
st.header("Passenger Details")

pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

age = st.slider(
    "Age",
    min_value=1,
    max_value=80,
    value=25
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    max_value=600.0,
    value=50.0
)

st.divider()

# ---------------- PREDICTION ----------------
if st.button("Predict Survival"):

    # Create DataFrame
    input_df = pd.DataFrame({
        "Pclass": [pclass],
        "Age": [age],
        "Fare": [fare]
    })

    # Scale Input
    scaled_input = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(scaled_input)

    probability = float(prediction[0][0])

    st.header("Prediction Result")

    if probability >= 0.5:
        st.success("✅ Passenger is likely to SURVIVE")
    else:
        st.error("❌ Passenger is NOT likely to survive")

    st.metric(
        label="Survival Probability",
        value=f"{probability * 100:.2f}%"
    )

    # Chart
    chart_data = pd.DataFrame({
        "Result": ["Survive", "Not Survive"],
        "Probability": [probability, 1 - probability]
    })

    st.bar_chart(chart_data.set_index("Result"))