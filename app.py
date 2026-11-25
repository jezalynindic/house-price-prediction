
import os
import gdown
import joblib
import pandas as pd
import streamlit as st

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏡",
    layout="wide"
)

# ==========================================
# GOOGLE DRIVE MODEL FILES
# ==========================================

MODEL_FILES = {
    "best_house_model.pkl": "12Dt1dlPYATp4V9tdwDQPa83bBfk2S7cl",
    "house_imputer.pkl": "1MYR--qJ5GcckScyHzIoti_L5X0_fCURj",
    "house_features.pkl": "1-6daVESeOG0XQuOOht_ild02XxT-JQKU"
}

# Download model files from Google Drive
@st.cache_resource
def load_models():

    for filename, file_id in MODEL_FILES.items():

        if not os.path.exists(filename):

            if file_id.startswith("YOUR_"):
                st.error(
                    f"Please add the Google Drive file ID "
                    f"for {filename} in app.py."
                )
                st.stop()

            with st.spinner(f"Downloading {filename}..."):

                output = gdown.download(
                    id=file_id,
                    output=filename,
                    quiet=False
                )

                if output is None:
                    st.error(f"Failed to download {filename}.")
                    st.stop()

    model = joblib.load("best_house_model.pkl")
    imputer = joblib.load("house_imputer.pkl")
    features = joblib.load("house_features.pkl")

    return model, imputer, features


model, imputer, features = load_models()

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #f4f6fb;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #263c72;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 16px;
    margin-bottom: 30px;
}

.section-title {
    color: #263c72;
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 15px;
}

.prediction-box {
    background: linear-gradient(135deg, #263c72, #536dba);
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    color: white;
    margin-top: 20px;
}

.prediction-label {
    font-size: 17px;
    color: #e2e8f0;
}

.prediction-price {
    font-size: 38px;
    font-weight: 800;
    color: white;
}

.info-box {
    background-color: #e8eefc;
    padding: 18px;
    border-radius: 12px;
    color: #263c72;
    margin-top: 15px;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    margin-top: 40px;
}

div.stButton > button {
    background-color: #263c72;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 12px 25px;
    font-weight: bold;
    width: 100%;
}

div.stButton > button:hover {
    background-color: #405b9b;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="main-title">🏡 House Price Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning-Based House Price Estimation'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("🏠 About the System")

    st.write("""
    This application predicts house prices using
    machine learning models trained on the
    Kaggle House Sales in King County dataset.
    """)

    st.divider()

    st.subheader("🤖 Best Performing Model")

    st.success("Random Forest Regressor")

    st.write("**R² Score:** 68.49%")
    st.write("**MAE:** $127,670.18")
    st.write("**RMSE:** $218,255.82")

    st.divider()

    st.caption("Artificial Intelligence 5.0")
    st.caption("Regression Model Evaluation")

# ==========================================
# HOUSE INFORMATION INPUTS
# ==========================================

st.markdown(
    '<div class="section-title">🏠 Enter House Information</div>',
    unsafe_allow_html=True
)

st.write("Provide the details of the house to estimate its price.")

col1, col2, col3 = st.columns(3)

with col1:

    st.subheader("🏡 Basic Information")

    bedrooms = st.number_input(
        "Number of Bedrooms",
        min_value=0,
        max_value=20,
        value=3
    )

    bathrooms = st.number_input(
        "Number of Bathrooms",
        min_value=0.0,
        max_value=20.0,
        value=2.0,
        step=0.25
    )

    floors = st.number_input(
        "Number of Floors",
        min_value=1.0,
        max_value=5.0,
        value=1.0,
        step=0.5
    )

    waterfront = st.selectbox(
        "Waterfront View",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

with col2:

    st.subheader("📐 Property Size")

    sqft_living = st.number_input(
        "Living Area (sqft)",
        min_value=100,
        max_value=20000,
        value=1800
    )

    sqft_lot = st.number_input(
        "Lot Area (sqft)",
        min_value=100,
        max_value=1000000,
        value=5000
    )

    sqft_above = st.number_input(
        "Above Ground Area (sqft)",
        min_value=0,
        max_value=20000,
        value=1500
    )

    sqft_basement = st.number_input(
        "Basement Area (sqft)",
        min_value=0,
        max_value=10000,
        value=300
    )

with col3:

    st.subheader("⭐ House Quality")

    view = st.selectbox(
        "View Rating",
        options=[0, 1, 2, 3, 4]
    )

    condition = st.selectbox(
        "House Condition",
        options=[1, 2, 3, 4, 5],
        index=2
    )

    grade = st.selectbox(
        "Construction Grade",
        options=list(range(1, 14)),
        index=6
    )

    yr_built = st.number_input(
        "Year Built",
        min_value=1800,
        max_value=2015,
        value=2000
    )

    yr_renovated = st.number_input(
        "Year Renovated (0 if never renovated)",
        min_value=0,
        max_value=2015,
        value=0
    )

# ==========================================
# PREDICTION SETTINGS
# ==========================================

st.divider()

st.markdown(
    '<div class="section-title">💰 Price Prediction</div>',
    unsafe_allow_html=True
)

# Illustrative conversion rate.
# Update this value when you want to use another rate.
USD_TO_PHP = 58.50

st.info(
    f"Currency conversion rate used: "
    f"$1 USD = ₱{USD_TO_PHP:.2f} PHP"
)

# ==========================================
# PREDICTION
# ==========================================

if st.button("🔍 PREDICT HOUSE PRICE"):

    input_values = {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft_living": sqft_living,
        "sqft_lot": sqft_lot,
        "floors": floors,
        "waterfront": waterfront,
        "view": view,
        "condition": condition,
        "grade": grade,
        "sqft_above": sqft_above,
        "sqft_basement": sqft_basement,
        "yr_built": yr_built,
        "yr_renovated": yr_renovated
    }

    # Arrange input in the same feature order used during training
    input_data = pd.DataFrame(
        [[input_values[feature] for feature in features]],
        columns=features
    )

    # Apply the saved imputer
    input_data = pd.DataFrame(
        imputer.transform(input_data),
        columns=features
    )

    # Predict house price in USD
    prediction_usd = float(model.predict(input_data)[0])

    # Convert USD to PHP
    prediction_php = prediction_usd * USD_TO_PHP

    st.markdown(
        f"""
        <div class="prediction-box">
            <div class="prediction-label">
                Estimated House Price (Philippine Peso)
            </div>
            <div class="prediction-price">
                ₱{prediction_php:,.2f}
            </div>
            <div class="prediction-label">
                Original Prediction: ${prediction_usd:,.2f} USD
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.success("House price prediction completed successfully!")

    st.markdown("### 📊 Prediction Summary")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            "Predicted Price (USD)",
            f"${prediction_usd:,.2f}"
        )

    with result_col2:
        st.metric(
            "Converted Price (PHP)",
            f"₱{prediction_php:,.2f}"
        )

    st.caption(
        "Note: The dataset contains house sales from King County, USA. "
        "The Philippine peso amount is only a currency conversion, "
        "not an estimate of actual Philippine property market value."
    )

# ==========================================
# MODEL INFORMATION
# ==========================================

st.divider()

with st.expander("📈 View Model Performance"):

    st.subheader("Regression Model Comparison")

    performance_data = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Decision Tree",
            "Random Forest"
        ],
        "MAE": [
            143679.664339,
            172087.544683,
            127670.175417
        ],
        "RMSE": [
            229309.029437,
            285767.251239,
            218255.823442
        ],
        "R² Score": [
            0.652178,
            0.459818,
            0.684901
        ]
    })

    st.dataframe(
        performance_data,
        use_container_width=True,
        hide_index=True
    )

    st.success(
        "Random Forest is the best-performing model "
        "among the three models based on the test results."
    )

    st.write("""
    **Evaluation Metrics:**

    - **MAE:** Mean Absolute Error measures the average absolute difference between actual and predicted prices.
    - **RMSE:** Root Mean Squared Error measures prediction errors while giving larger errors more weight.
    - **R² Score:** Measures how much of the variation in house prices is explained by the model.
    """)

# ==========================================
# FOOTER
# ==========================================

st.markdown(
    """
    <div class="footer">
        House Price Prediction System |
        Artificial Intelligence 5.0 |
        Powered by Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)
