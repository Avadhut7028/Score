import streamlit as st
import pickle
import pandas as pd
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Student Exam Predictor", 
    page_icon="📚", 
    layout="wide"
)

# --- Load the Model ---
@st.cache_resource
def load_model():
    # Loading your specific file verbatim
    with open("model (2).pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# --- App Header ---
st.title("📚 Student Performance Predictor")
st.markdown("Enter the student's details below to predict if they will **Pass** or **Fail**.")
st.divider()

# --- Input Fields Layout ---
# Using 3 columns for a clean, wide layout
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=20)
    gender = st.number_input("Gender (Encoded: e.g., 0=Male, 1=Female)", value=0)
    course = st.number_input("Course (Encoded ID)", value=0)
    study_hours = st.number_input("Study Hours per Week", min_value=0.0, value=10.0)

with col2:
    class_attendance = st.number_input("Class Attendance (%)", min_value=0.0, max_value=100.0, value=80.0)
    internet_access = st.number_input("Internet Access (0=No, 1=Yes)", min_value=0, max_value=1, value=1)
    sleep_hours = st.number_input("Sleep Hours per Night", min_value=0.0, value=7.0)
    sleep_quality = st.number_input("Sleep Quality (1-10)", min_value=1, max_value=10, value=7)

with col3:
    study_method = st.number_input("Study Method (Encoded ID)", value=0)
    facility_rating = st.number_input("Facility Rating (1-10)", min_value=1, max_value=10, value=5)
    exam_difficulty = st.number_input("Exam Difficulty (1-10)", min_value=1, max_value=10, value=5)

st.write("") # Spacing

# --- Predict Button & Effects ---
if st.button("Predict Outcome 🚀", use_container_width=True):
    
    with st.spinner("Analyzing student profile..."):
        time.sleep(1) # Short delay for the spinner effect
        
        # Format the data exactly as the SVR model expects it
        input_data = pd.DataFrame(
            [[age, gender, course, study_hours, class_attendance, internet_access, 
              sleep_hours, sleep_quality, study_method, facility_rating, exam_difficulty]],
            columns=['age', 'gender', 'course', 'study_hours', 'class_attendance', 
                     'internet_access', 'sleep_hours', 'sleep_quality', 'study_method', 
                     'facility_rating', 'exam_difficulty']
        )
        
        # Make the continuous prediction
        predicted_score = model.predict(input_data)[0]
        
        # --- Pass/Fail Logic ---
        # Assuming the model predicts a score out of 100, and 50 is passing.
        passing_threshold = 50.0 
        
        if predicted_score >= passing_threshold:
            st.balloons()
            st.success(f"### Prediction Result: PASS 🎉")
            st.write(f"**Predicted Score:** {predicted_score:.2f}")
        else:
            st.snow()
            st.error(f"### Prediction Result: FAIL 😔")
            st.write(f"**Predicted Score:** {predicted_score:.2f}")
