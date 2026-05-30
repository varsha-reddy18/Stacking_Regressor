import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="Stacking Regressor",
    page_icon="🏠",
    layout="wide"
)

# ======================================
# LOAD CSS
# ======================================

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css("style.css")

# ======================================
# LOAD DATA
# ======================================

df = pd.read_csv("data/housing.csv")

# ======================================
# FEATURES & TARGET
# ======================================

X = df.drop("Price", axis=1)
y = df["Price"]

# ======================================
# SPLIT DATA
# ======================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ======================================
# LOAD MODEL & SCALER
# ======================================

import os
import joblib

if not os.path.exists(
    "models/stacking_regressor.pkl"
):
    from implementation.train_model import *

model = joblib.load(
    "models/stacking_regressor.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)

# ======================================
# SCALE TEST DATA
# ======================================

X_test_scaled = scaler.transform(X_test)

# ======================================
# PREDICTIONS
# ======================================

y_pred = model.predict(X_test_scaled)

# ======================================
# METRICS
# ======================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)

# ======================================
# HEADER
# ======================================

st.markdown("""
<div class="main-header">

<h1>🏠 Stacking Regressor Dashboard</h1>

<p>
California Housing Price Prediction using Ensemble Learning
</p>

</div>
""", unsafe_allow_html=True)

# ======================================
# DATASET OVERVIEW
# ======================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.subheader("📋 Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Rows",
    df.shape[0]
)

c2.metric(
    "Columns",
    df.shape[1]
)

c3.metric(
    "Average Price",
    f"{df['Price'].mean():.2f}"
)

c4.metric(
    "R² Score",
    f"{r2:.3f}"
)

st.dataframe(
    df.head(10)
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

# ======================================
# STATISTICAL SUMMARY
# ======================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.subheader("📈 Statistical Summary")

st.dataframe(
    df.describe()
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

# ======================================
# CORRELATION HEATMAP
# ======================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.subheader("🔥 Correlation Heatmap")

fig, ax = plt.subplots(
    figsize=(12,8)
)

sns.heatmap(
    df.corr(),
    cmap="coolwarm"
)

st.pyplot(fig)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

# ======================================
# PRICE DISTRIBUTION
# ======================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.subheader("🏠 House Price Distribution")

fig, ax = plt.subplots(
    figsize=(8,5)
)

sns.histplot(
    df["Price"],
    kde=True,
    color="orange"
)

st.pyplot(fig)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

# ======================================
# ACTUAL VS PREDICTED
# ======================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.subheader("📊 Actual vs Predicted")

fig, ax = plt.subplots(
    figsize=(8,5)
)

ax.scatter(
    y_test,
    y_pred,
    color="deeppink"
)

ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")

st.pyplot(fig)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

# ======================================
# EVALUATION METRICS
# ======================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.subheader("📊 Evaluation Metrics")

m1, m2, m3 = st.columns(3)

m1.metric(
    "MAE",
    f"{mae:.3f}"
)

m2.metric(
    "RMSE",
    f"{rmse:.3f}"
)

m3.metric(
    "R² Score",
    f"{r2:.3f}"
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

# ======================================
# HYPERPARAMETERS
# ======================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.subheader("⚙️ Stacking Regressor Hyperparameters")

col1, col2 = st.columns(2)

with col1:

    rf_estimators = st.slider(
        "RF Estimators",
        10,
        500,
        100
    )

    rf_depth = st.slider(
        "RF Max Depth",
        1,
        30,
        10
    )

with col2:

    gb_estimators = st.slider(
        "GB Estimators",
        10,
        500,
        100
    )

    gb_learning_rate = st.slider(
        "GB Learning Rate",
        0.01,
        1.0,
        0.1
    )

cv_folds = st.slider(
    "CV Folds",
    2,
    10,
    5
)

st.info(
    "Hyperparameters are displayed for educational purposes. "
    "The saved Stacking Regressor model is used for prediction."
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

# ======================================
# PREDICTION SECTION
# ======================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.subheader("🏡 Predict House Price")

col1, col2 = st.columns(2)

with col1:

    medinc = st.number_input(
        "Median Income",
        value=3.0
    )

    houseage = st.number_input(
        "House Age",
        value=20.0
    )

    averooms = st.number_input(
        "Average Rooms",
        value=5.0
    )

    avebedrms = st.number_input(
        "Average Bedrooms",
        value=1.0
    )

with col2:

    population = st.number_input(
        "Population",
        value=1000.0
    )

    aveoccup = st.number_input(
        "Average Occupancy",
        value=3.0
    )

    latitude = st.number_input(
        "Latitude",
        value=34.0
    )

    longitude = st.number_input(
        "Longitude",
        value=-118.0
    )

if st.button("🚀 Predict House Price"):

    input_data = np.array([[
        medinc,
        houseage,
        averooms,
        avebedrms,
        population,
        aveoccup,
        latitude,
        longitude
    ]])

    input_scaled = scaler.transform(
        input_data
    )

    prediction = model.predict(
        input_scaled
    )[0]

    st.markdown(
        f"""
        <div class="prediction-box">

        <h2>Predicted House Price</h2>

        <h1>$ {prediction:.2f}</h1>

        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    '</div>',
    unsafe_allow_html=True
)