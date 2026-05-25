import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1 {
    color: #00BFFF;
    text-align: center;
    font-size: 50px;
}

h2, h3 {
    color: white;
}

.stButton>button {
    background-color: #00BFFF;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #1E90FF;
    color: white;
}

.css-1d391kg {
    background-color: #111827;
}

.metric-card {
    background-color: #1F2937;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<h1>🚢 Titanic Survival Prediction System</h1>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center; color:white; font-size:20px;'>
Predict survival chances of Titanic passengers using smart analysis.
</div>
""", unsafe_allow_html=True)

st.divider()

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙ Passenger Details")

pclass = st.sidebar.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

age = st.sidebar.slider(
    "Age",
    1,
    80,
    25
)

fare = st.sidebar.number_input(
    "Fare",
    min_value=0.0,
    max_value=600.0,
    value=50.0
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

# ---------------- PREDICTION LOGIC ----------------
if st.sidebar.button("Predict Survival"):

    score = 0

    # Class score
    if pclass == 1:
        score += 40
    elif pclass == 2:
        score += 25
    else:
        score += 10

    # Age score
    if age < 15:
        score += 25
    elif age < 40:
        score += 15

    # Fare score
    if fare > 100:
        score += 25
    elif fare > 50:
        score += 15

    # Gender score
    if gender == "Female":
        score += 20

    probability = min(score, 100)

    # ---------------- RESULT ----------------
    st.subheader("🎯 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        if probability >= 50:
            st.success("✅ Passenger is likely to SURVIVE")
        else:
            st.error("❌ Passenger is NOT likely to survive")

    with col2:
        st.metric(
            label="Survival Probability",
            value=f"{probability}%"
        )

    st.divider()

    # ---------------- PIE CHART ----------------
    pie_data = pd.DataFrame({
        "Category": ["Survive", "Not Survive"],
        "Value": [probability, 100 - probability]
    })

    pie_fig = px.pie(
        pie_data,
        names="Category",
        values="Value",
        title="Survival Probability Distribution",
        hole=0.5
    )

    st.plotly_chart(pie_fig, use_container_width=True)

    # ---------------- BAR CHART ----------------
    bar_data = pd.DataFrame({
        "Factors": ["Class", "Age", "Fare", "Gender"],
        "Score": [
            40 if pclass == 1 else 25 if pclass == 2 else 10,
            25 if age < 15 else 15 if age < 40 else 5,
            25 if fare > 100 else 15 if fare > 50 else 5,
            20 if gender == "Female" else 5
        ]
    })

    bar_fig = px.bar(
        bar_data,
        x="Factors",
        y="Score",
        title="Factor Contribution Analysis",
        text="Score"
    )

    st.plotly_chart(bar_fig, use_container_width=True)

    # ---------------- GAUGE CHART ----------------
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        title={'text': "Survival Chance"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "green"},
            'steps': [
                {'range': [0, 50], 'color': "red"},
                {'range': [50, 100], 'color': "lightgreen"}
            ],
        }
    ))

    st.plotly_chart(gauge_fig, use_container_width=True)

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<div style='text-align:center; color:gray;'>
Made with ❤️ using Streamlit
</div>
""", unsafe_allow_html=True)