import streamlit as st
import pickle
import pandas as pd
import time

# --- 1. Page Configuration & Custom Styling ---
st.set_page_config(
    page_title="Student Analytics Dashboard", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Advanced CSS for a professional, high-tech UI
st.markdown("""
    <style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Style the main title */
    .main-title {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #1E3A8A;
        font-weight: 800;
        font-size: 3rem;
        text-align: center;
        margin-bottom: 0px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Subtitle text */
    .sub-title {
        color: #4B5563;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 30px;
    }

    /* Style the Predict Button */
    div.stButton > button {
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        background: linear-gradient(90deg, #3B82F6 0%, #1E3A8A 100%);
    }

    /* Card-like containers for form inputs */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- App Header ---
st.markdown('<h1 class="main-title">📊 Student Analytics Engine</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Advanced Support Vector Regression (SVR) Performance Prediction</p>', unsafe_allow_html=True)


# --- 2. Load the Model Safely ---
@st.cache_resource
def load_model():
    try:
        # Update this filename if yours is still "model (2).pkl"
        with open("model (2).pkl", "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        return None

model = load_model()

if model is None:
    st.error("🚨 **Error:** `model.pkl` not found. Please ensure the file is uploaded to GitHub and named exactly `model.pkl` (case-sensitive).")
    st.stop() # Stops the app from running further until the file is fixed


# --- 3. Dashboard Layout & Inputs ---
# We use containers and columns to create a clean, card-like grid
st.markdown("### 🧑‍🎓 Student Profile Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.markdown("#### 📋 Demographics & Academics")
        age = st.number_input("Age", min_value=10, max_value=100, value=20, step=1)
        gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male (0)" if x==0 else "Female (1)")
        course = st.number_input("Course ID", min_value=0, max_value=20, value=1, step=1, help="Numeric identifier for the registered course.")
        study_method = st.number_input("Study Method ID", min_value=0, max_value=10, value=1, help="Numeric identifier for the primary study technique.")

with col2:
    with st.container(border=True):
        st.markdown("#### ⏱️ Habits & Attendance")
        study_hours = st.number_input("Study Hours (Weekly)", min_value=0.0, max_value=100.0, value=15.5)
        class_attendance = st.slider("Class Attendance (%)", min_value=0.0, max_value=100.0, value=85.0)
        sleep_hours = st.number_input("Sleep Hours (Nightly)", min_value=0.0, max_value=24.0, value=7.5)
        sleep_quality = st.slider("Sleep Quality Rating", min_value=1, max_value=10, value=7, help="1 = Very Poor, 10 = Excellent")

with col3:
    with st.container(border=True):
        st.markdown("#### 🏢 Environment & Accessibility")
        internet_access = st.selectbox("Internet Access", options=[0, 1], format_func=lambda x: "No (0)" if x==0 else "Yes (1)")
        facility_rating = st.slider("Facility Rating", min_value=1, max_value=10, value=6, help="Student's rating of institutional facilities.")
        exam_difficulty = st.slider("Exam Difficulty Perception", min_value=1, max_value=10, value=5)


# --- 4. Prediction Execution ---
st.markdown("<br>", unsafe_allow_html=True) # Spacer

# Center the button using columns
_, btn_col, _ = st.columns([1, 2, 1])

with btn_col:
    if st.button("Initialize Prediction Model 🚀", use_container_width=True):
        
        # Loading bar effect
        progress_text = "Running SVR algorithms..."
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(0.5)
        my_bar.empty() # Clear the progress bar when done
        
        # Structure the DataFrame exactly as the model expects
        input_data = pd.DataFrame(
            [[age, gender, course, study_hours, class_attendance, internet_access, 
              sleep_hours, sleep_quality, study_method, facility_rating, exam_difficulty]],
            columns=['age', 'gender', 'course', 'study_hours', 'class_attendance', 
                     'internet_access', 'sleep_hours', 'sleep_quality', 'study_method', 
                     'facility_rating', 'exam_difficulty']
        )
        
        # Inference
        prediction = model.predict(input_data)
        final_score = prediction[0]
        
        # Celebration effect
        st.balloons()
        
        # Display the result in a highly visible metric card
        st.markdown(f"""
            <div style="background: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 5px solid #10B981;">
                <h3 style="color: #4B5563; margin-bottom: 0;">Predicted Performance Score</h3>
                <h1 style="color: #10B981; font-size: 3.5rem; margin-top: 10px;">{final_score:.2f}</h1>
            </div>
        """, unsafe_allow_html=True)
