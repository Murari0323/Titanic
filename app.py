import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
import pandas as pd

# --- Configuration --- 
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Load Model and Scaler (Task 5) ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('titanic_survival_ann_model.keras')
    return model

@st.cache_resource
def load_scaler():
    scaler = joblib.load('scaler_minmax.pkl')
    return scaler

model = load_model()
scaler = load_scaler()

# --- SECTION 1: Header Area (Task 3) ---
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🚢 Titanic Survival Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #5DADE2;'><i>Deep Learning Based Passenger Survival Prediction</i></h3>", unsafe_allow_html=True)
st.image("https://www.freeiconspng.com/thumbs/titanic-png/titanic-transparent-png-1.png", width=150) # Example AI-related icon

st.markdown("---")

# --- SECTION 2: Project Description (Task 3) ---
st.header("Project Overview")
st.write(
    "This application predicts the survival probability of a Titanic passenger during the emergency situation "
    "using an Artificial Neural Network (ANN) model. The model was trained with key passenger information."
)
st.write(
    "**How it works:** The ANN processes your input data (Passenger Class, Age, Fare), normalizes it "
    "using the same Min-Max scaling applied during training, and then feeds it through its hidden layers "
    "to produce a survival probability. This demonstrates a simple TensorFlow model deployment with Streamlit."
)

st.markdown("---<br>", unsafe_allow_html=True)

# --- SECTION 3: Passenger Input Form (Task 3) ---
st.header("Passenger Information")

with st.container(border=True):
    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox(
            "Passenger Class (Pclass)",
            options=[1, 2, 3],
            format_func=lambda x: f"Class {x} (Upper/Middle/Lower)",
            index=2, # Default to 3rd class
            help="1st = Upper; 2nd = Middle; 3rd = Lower"
        )
    with col2:
        age = st.slider(
            "Age",
            min_value=0.5, max_value=80.0, value=30.0, step=0.5,
            help="Passenger's age"
        )

    fare = st.number_input(
        "Fare (in GBP)",
        min_value=0.0, max_value=500.0, value=30.0, step=0.01,
        help="Passenger's fare cost"
    )

st.markdown("---<br>", unsafe_allow_html=True)

# --- SECTION 4: Prediction Button (Task 3) ---
if st.button("Predict Survival", type="primary", use_container_width=True):
    # --- Data Preprocessing (Task 4) ---
    # Create a DataFrame for the input, matching the training features
    input_data = pd.DataFrame([[pclass, age, fare]], columns=['Pclass', 'Age', 'Fare'])

    # Apply the same Min-Max scaling
    scaled_input = scaler.transform(input_data)

    # --- Model Prediction (Task 6) ---
    prediction = model.predict(scaled_input)[0][0]
    survival_probability = float(prediction) * 100
    non_survival_probability = (1 - float(prediction)) * 100

    # Determine outcome
    if prediction > 0.5:
        outcome = "Survived"
        outcome_color = "green"
        emoji = "✅"
    else:
        outcome = "Not Survived"
        outcome_color = "red"
        emoji = "❌"

    # --- SECTION 5: Prediction Output Area (Task 3) ---
    st.header("Prediction Result")
    with st.container(border=True):
        st.markdown(f"<h3 style='text-align: center;'>{emoji} Passenger is likely to <span style='color:{outcome_color};'>{outcome.upper()}</span> {emoji}</h3>", unsafe_allow_html=True)

        st.metric(label="Survival Probability", value=f"{survival_probability:.2f}%")

        confidence_score = abs(prediction - 0.5) * 200 # Scale from 0-0.5 to 0-100 for confidence
        st.info(f"**Confidence Score:** {confidence_score:.2f}% - indicating the model's certainty in its prediction.")

    st.markdown("---<br>", unsafe_allow_html=True)

    # --- SECTION 6: Visualization Area (Task 3) ---
    st.header("Probability Distribution")
    # Using a simple bar chart or pie chart visualization
    chart_data = pd.DataFrame({
        'Category': ['Survived', 'Not Survived'],
        'Probability': [survival_probability, non_survival_probability]
    })

    # Customizing the bar chart for better visualization
    st.bar_chart(chart_data.set_index('Category'))

    # Alternatively, you can use a pie chart with plotly if preferred
    # import plotly.express as px
    # fig = px.pie(chart_data, values='Probability', names='Category', title='Survival vs Non-Survival Probability')
    # fig.update_traces(textinfo='percent+label', pull=[0.05 if outcome == 'Survived' else 0, 0.05 if outcome == 'Not Survived' else 0])
    # st.plotly_chart(fig)

# --- UI Styling Requirements (Task 7) ---
# Streamlit's default styling, containers, columns, metrics are used. Colors are managed via markdown for text.
