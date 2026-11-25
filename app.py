
import streamlit as st
import pandas as pd
import joblib

# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# =====================================
# LOAD TRAINED MODEL
# =====================================

model = joblib.load("best_house_model.pkl")
features = joblib.load("house_features.pkl")

# Illustrative USD to PHP conversion rate
USD_TO_PHP = 58.50

# =====================================
# CUSTOM DESIGN
# =====================================

st.markdown("""
<style>
    .main {
        background-color: #f8fafc;
    }

    .main-title {
        color: #166534;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
    }

    .subtitle {
        color: #64748b;
        text-align: center;
        font-size: 16px;
    }

    .price-box {
        background-color: #f0fdf4;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #bbf7d0;
        text-align: center;
    }

    .price {
        color: #15803d;
        font-size: 38px;
        font-weight: bold;
    }

    div.stButton > button {
        background-color: #15803d;
        color: white;
        border-radius: 10px;
        height: 50px;
        font-weight: bold;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #166534;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# =====================================
# HEADER
# =====================================

st.markdown(
    '<p class="main-title">🏠 House Price Prediction System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Predict estimated house prices using Machine Learning</p>',
    unsafe_allow_html=True
)

st.divider()

st.info(
    "Enter the house information below to estimate its selling price "
    "in Philippine Peso (PHP)."
)

# =====================================
# HOUSE INFORMATION
# =====================================

st.subheader("🏡 House Information")

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown("#### Basic Information")

    bedrooms = st.number_input(
        "Number of Bedrooms",
        min_value=0,
        max_value=20,
        value=3
    )

    bathrooms = st.number_input(
        "Number of Bathrooms",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.5
    )

    sqft_living = st.number_input(
        "Living Area (sqft)",
        min_value=100,
        value=1800
    )

    sqft_lot = st.number_input(
        "Lot Area (sqft)",
        min_value=100,
        value=5000
    )

    floors = st.number_input(
        "Number of Floors",
        min_value=1.0,
        max_value=4.0,
        value=1.0,
        step=0.5
    )

with col2:

    st.markdown("#### House Features")

    waterfront = st.selectbox(
        "Waterfront View",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    view = st.selectbox(
        "View Rating",
        [0, 1, 2, 3, 4]
    )

    condition = st.selectbox(
        "House Condition",
        [1, 2, 3, 4, 5],
        index=2
    )

    grade = st.selectbox(
        "House Grade",
        list(range(1, 14)),
        index=6
    )

with col3:

    st.markdown("#### Property Details")

    sqft_above = st.number_input(
        "Above Ground Area (sqft)",
        min_value=100,
        value=1500
    )

    sqft_basement = st.number_input(
        "Basement Area (sqft)",
        min_value=0,
        value=300
    )

    yr_built = st.number_input(
        "Year Built",
        min_value=1800,
        max_value=2026,
        value=2000
    )

    yr_renovated = st.number_input(
        "Year Renovated (0 if never renovated)",
        min_value=0,
        max_value=2026,
        value=0
    )

# =====================================
# PREDICTION
# =====================================

st.divider()

st.subheader("📊 House Price Prediction")

if st.button("🔍 Predict House Price", use_container_width=True):

    input_data = pd.DataFrame([{
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
    }], columns=features)

    # Predict using Random Forest
    prediction_usd = model.predict(input_data)[0]

    # Convert USD to Philippine Peso
    prediction_php = prediction_usd * USD_TO_PHP

    # Display result
    st.success("House Price Prediction Completed!")

    st.markdown(
        f"""
        <div class="price-box">
            <p style="color:#166534;font-size:18px;">
                Estimated House Selling Price
            </p>
            <p class="price">
                ₱{prediction_php:,.2f}
            </p>
            <p style="color:#64748b;">
                Philippine Peso (PHP)
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # Prediction details
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Original Prediction (USD)",
            f"${prediction_usd:,.2f}"
        )

    with col2:
        st.metric(
            "Converted Prediction (PHP)",
            f"₱{prediction_php:,.2f}"
        )

    st.caption(
        f"Conversion rate: 1 USD = ₱{USD_TO_PHP:,.2f}"
    )

    st.warning(
        "Note: This conversion is for demonstration purposes. "
        "The model was trained using US housing data, so the converted "
        "amount is not a Philippine market valuation."
    )

# =====================================
# MODEL INFORMATION
# =====================================

st.divider()

with st.expander("📌 About the Machine Learning Model"):

    st.write("**Selected Model:** Random Forest Regressor")

    st.write("**Dataset:** House Sales in King County, USA (Kaggle)")

    st.write("**Target Variable:** price")

    st.write("**R² Score:** 68.49%")

    st.write("**MAE:** $127,670.18")

    st.write("**RMSE:** $218,255.82")

    st.write(
        "The Random Forest model was selected because it achieved "
        "the highest R² score and lowest prediction errors among "
        "the three evaluated regression models."
    )

# =====================================
# FOOTER
# =====================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;color:#64748b;">
        <p><b>House Price Prediction System</b></p>
        <p>Artificial Intelligence 5.0 | Regression</p>
        <p>Powered by Python, Scikit-learn and Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
