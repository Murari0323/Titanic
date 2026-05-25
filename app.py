import streamlit as st

# Page Config
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

# Title
st.title("🚢 Titanic Survival Prediction")

st.write(
    "Simple Titanic survival prediction system."
)

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

    # Simple logic
    score = 0

    if pclass == 1:
        score += 40
    elif pclass == 2:
        score += 25
    else:
        score += 10

    if age < 15:
        score += 25
    elif age < 40:
        score += 15

    if fare > 100:
        score += 25
    elif fare > 50:
        score += 15

    probability = min(score, 100)

    # Result
    st.subheader("Prediction Result")

    if probability >= 50:
        st.success("✅ Passenger is likely to SURVIVE")
    else:
        st.error("❌ Passenger is NOT likely to survive")

    st.metric(
        "Survival Probability",
        f"{probability}%"
    )