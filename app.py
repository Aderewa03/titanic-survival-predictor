"""
🚢 TITANIC SURVIVAL PREDICTION SYSTEM
=====================================
A machine learning web application that predicts passenger survival
on the Titanic using a trained neural network model.

Author: Temmy
Date: January 2026
Model: Neural Network (Keras)
"""

# ============================================
# IMPORTS
# ============================================
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from keras.models import load_model
import time

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================
# CUSTOM CSS STYLING
# ============================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@400;500;600&family=Roboto+Mono&display=swap');
    
    /* Main Background */
    .main {
        background-color: #f5f9fc;
    }
    
    /* Header Styling */
    .header-container {
        background: linear-gradient(135deg, #1a4d6d 0%, #0a2239 100%);
        padding: 2.5rem 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .header-title {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        color: #ffffff;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .header-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        color: #d4af37;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Card Styling */
    .card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
    }
    
    .card-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        color: #0a2239;
        font-weight: 600;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #d4af37;
        padding-bottom: 0.5rem;
    }
    
    /* Result Cards */
    .result-survived {
        background: linear-gradient(135deg, #2d7a4f 0%, #1e5a3a 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        animation: fadeIn 0.5s ease-in;
    }
    
    .result-not-survived {
        background: linear-gradient(135deg, #c41e3a 0%, #8b1528 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        animation: fadeIn 0.5s ease-in;
    }
    
    .result-status {
        font-family: 'Playfair Display', serif;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 1rem 0;
    }
    
    .result-probability {
        font-family: 'Roboto Mono', monospace;
        font-size: 1.8rem;
        margin: 1rem 0;
    }
    
    /* Animation */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #d4af37 0%, #b8941f 100%);
        color: white;
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border: none;
        border-radius: 8px;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #b8941f 0%, #9a7a1a 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(212, 175, 55, 0.4);
    }
    
    /* Input Labels */
    label {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        color: #0a2239;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #536878;
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
    }
    
    /* Info Box */
    .info-box {
        background-color: #e8f4f8;
        border-left: 4px solid #1a4d6d;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODEL AND SCALER
# ============================================
@st.cache_resource  # Cache the model to avoid reloading
def load_ml_model():
    """Load the trained model and scaler"""
    try:
        model = load_model("model.h5")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"❌ Error loading model files: {str(e)}")
        st.stop()

# Load model at startup
model, scaler = load_ml_model()

# ============================================
# HEADER SECTION
# ============================================
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🚢 Titanic Survival Predictor</h1>
    <p class="header-subtitle">Will You Make It to the Lifeboats?</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# INFORMATION SECTION
# ============================================
with st.expander("ℹ️ About This Application"):
    st.markdown("""
    <div class="info-box">
    This application uses a <strong>Neural Network</strong> trained on historical Titanic passenger data 
    to predict survival probability. The model considers factors like passenger class, gender, age, 
    family members aboard, ticket fare, and port of embarkation.
    
    <br><br>
    <strong>Accuracy:</strong> ~81% on test data
    <br>
    <strong>Model:</strong> Keras Sequential Neural Network
    <br>
    <strong>Features:</strong> 8 input features with StandardScaler preprocessing
    </div>
    """, unsafe_allow_html=True)

# ============================================
# INPUT FORM SECTION
# ============================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<h2 class="card-title">👤 Passenger Information</h2>', unsafe_allow_html=True)

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    # Passenger Class
    pclass = st.selectbox(
        "🎫 Passenger Class",
        options=[1, 2, 3],
        format_func=lambda x: f"{'First' if x==1 else 'Second' if x==2 else 'Third'} Class",
        help="First class = wealthy, Third class = lower income"
    )
    
    # Age
    age = st.number_input(
        "🎂 Age (years)",
        min_value=0.0,
        max_value=100.0,
        value=28.0,
        step=1.0,
        help="Age in years (0-100)"
    )
    
    # Siblings/Spouses
    sibsp = st.slider(
        "👫 Siblings/Spouses Aboard",
        min_value=0,
        max_value=8,
        value=0,
        help="Number of siblings or spouses traveling with you"
    )

with col2:
    # Gender
    sex = st.radio(
        "⚧ Gender",
        options=["Female", "Male"],
        help="Females had higher survival rates"
    )
    sex_encoded = 0 if sex == "Female" else 1
    
    # Fare
    fare = st.number_input(
        "💰 Fare (£)",
        min_value=0.0,
        max_value=600.0,
        value=32.0,
        step=0.5,
        help="Ticket price in British Pounds"
    )
    
    # Parents/Children
    parch = st.slider(
        "👨‍👩‍👧 Parents/Children Aboard",
        min_value=0,
        max_value=6,
        value=0,
        help="Number of parents or children traveling with you"
    )

# Full width for Embarked
embarked = st.selectbox(
    "⚓ Port of Embarkation",
    options=["Southampton", "Cherbourg", "Queenstown"],
    help="Where you boarded the Titanic"
)

# One-hot encode embarked
embarked_C = 1 if embarked == "Cherbourg" else 0
embarked_Q = 1 if embarked == "Queenstown" else 0

st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# PREDICTION SECTION
# ============================================

# Center the predict button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔮 PREDICT SURVIVAL", use_container_width=True)

if predict_button:
    # Show loading animation
    with st.spinner("🌊 Analyzing passenger data..."):
        time.sleep(1)  # Dramatic pause for effect
        
        # Prepare features array
        features = np.array([[
            pclass,
            sex_encoded,
            age,
            sibsp,
            parch,
            fare,
            embarked_C,
            embarked_Q
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        probability = model.predict(features_scaled, verbose=0)[0][0]
        
        # Determine survival status
        survived = probability >= 0.5
    
    # ============================================
    # DISPLAY RESULTS
    # ============================================
    st.markdown("---")
    
    if survived:
        # Survived Result
        st.markdown(f"""
        <div class="result-survived">
            <div style="font-size: 4rem;">✅</div>
            <div class="result-status">SURVIVED</div>
            <div class="result-probability">Survival Probability: {probability*100:.1f}%</div>
            <p style="font-size: 1rem; margin-top: 1rem; opacity: 0.9;">
                Based on historical data, a passenger with your profile had a high chance of survival.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show balloons celebration
        st.balloons()
        
    else:
        # Did Not Survive Result
        st.markdown(f"""
        <div class="result-not-survived">
            <div style="font-size: 4rem;">❌</div>
            <div class="result-status">DID NOT SURVIVE</div>
            <div class="result-probability">Survival Probability: {probability*100:.1f}%</div>
            <p style="font-size: 1rem; margin-top: 1rem; opacity: 0.9;">
                Based on historical data, a passenger with your profile had a low chance of survival.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # ============================================
    # ADDITIONAL DETAILS
    # ============================================
    with st.expander("📊 View Detailed Analysis"):
        st.markdown("### Passenger Profile Summary")
        
        profile_data = {
            "Attribute": [
                "Passenger Class",
                "Gender",
                "Age",
                "Siblings/Spouses",
                "Parents/Children",
                "Fare",
                "Embarked",
                "Family Size"
            ],
            "Value": [
                f"{'First' if pclass==1 else 'Second' if pclass==2 else 'Third'} Class",
                sex,
                f"{age} years",
                sibsp,
                parch,
                f"£{fare:.2f}",
                embarked,
                sibsp + parch
            ]
        }
        
        df_profile = pd.DataFrame(profile_data)
        st.dataframe(df_profile, use_container_width=True, hide_index=True)
        
        st.markdown("### Historical Context")
        st.info(f"""
        **Survival Statistics:**
        - Overall Survival Rate: 38.4%
        - {'Female' if sex=='Female' else 'Male'} Survival Rate: {'74.2%' if sex=='Female' else '18.9%'}
        - {f"{'First' if pclass==1 else 'Second' if pclass==2 else 'Third'} Class"} Survival Rate: {'62.9%' if pclass==1 else '47.3%' if pclass==2 else '24.2%'}
        """)

# ============================================
# FOOTER SECTION
# ============================================
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🚢 <strong>Titanic Survival Prediction System</strong></p>
    <p>Built with Streamlit • Powered by TensorFlow/Keras</p>
    <p style="font-size: 0.8rem; margin-top: 1rem;">
        Model trained on historical Titanic passenger data (1912)
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR (OPTIONAL)
# ============================================
with st.sidebar:
    st.markdown("### 🎯 Quick Tips")
    st.markdown("""
    - **First Class** passengers had better survival rates
    - **Women and children** were prioritized
    - **Younger passengers** had higher survival chances
    - **Lower deck cabins** (cheaper fares) were harder to evacuate
    """)
    
    st.markdown("---")
    st.markdown("### 📚 Resources")
    st.markdown("""
    - [Titanic Dataset](https://www.kaggle.com/c/titanic)
    - [Model Documentation](#)
    - [GitHub Repository](#)
    """)