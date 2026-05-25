import streamlit as st
import tensorflow as tf
import pandas as pd
import joblib

# Page Config
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("titanic_survival_ann_model.keras")

# Load Scaler
@st.cache_resource
def load_scaler():
    return joblib.load("scaler_minmax.pkl")

model = load_model()
scaler = load_scaler()

# Title
st.title("🚢 Titanic Survival Prediction")
st.write("ANN Deep Learning Model")

st.divider()

# Inputs
pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

age = st.slider(
    "Age",
    1,
    80,
    25
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    max_value=600.0,
    value=50.0
)

# Prediction
if st.button("Predict"):

    input_df = pd.DataFrame({
        "Pclass": [pclass],
        "Age": [age],
        "Fare": [fare]
    })

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)

    probability = float(prediction[0][0])

    st.subheader("Result")

    if probability >= 0.5:
        st.success("✅ Passenger is likely to SURVIVE")
    else:
        st.error("❌ Passenger is NOT likely to survive")

    st.metric(
        "Survival Probability",
        f"{probability * 100:.2f}%"
    )

    chart_data = pd.DataFrame({
        "Result": ["Survive", "Not Survive"],
        "Probability": [probability, 1 - probability]
    })

    st.bar_chart(chart_data.set_index("Result"))