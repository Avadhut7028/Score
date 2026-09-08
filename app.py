import streamlit as st
import pickle
import pandas as pd
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Student Score Predictor", 
    page_icon="📚", 
    layout="wide" # Using wide layout for 11 features
)

# --- Load the Model ---
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# --- App Header ---
st.title("📚 Student Performance Predictor (SVR)")
st.markdown("Adjust the student's metrics below to predict their final score/performance.")
st.divider()

# --- Input Fields Layout ---
# We will use 3 columns to neatly organize the 11 inputs
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics & Course")
    age = st.number_input("Age", min_value=10, max_value=100, value=20, step=1)
    # Assuming gender was encoded as 0 and 1
    gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male (0)" if x==0 else "Female (1)")
    # Assuming course was encoded numerically (0, 1, 2, etc.)
    course = st.number_input("Course ID (Numeric)", min_value=0, max_value=20, value=1, step=1)
    study_method = st.number_input("Study Method ID (Numeric)", min_value=0, max_value=10, value=1)

with col2:
    st.subheader("Habits & Attendance")
    study_hours = st.number_input("Study Hours (per week)", min_value=0.0, max_value=100.0, value=10.0)
    class_attendance = st.slider("Class Attendance (%)", min_value=0.0, max_value=100.0, value=85.0)
    sleep_hours = st.number_input("Sleep Hours (per night)", min_value=0.0, max_value=24.0, value=7.0)
    sleep_quality = st.slider("Sleep Quality (1-10)", min_value=1, max_value=10, value=7)

with col3:
    st.subheader("Environment")
    # Assuming internet access is binary (0=No, 1=Yes)
    internet_access = st.selectbox("Internet Access", options=[0, 1], format_func=lambda x: "No (0)" if x==0 else "Yes (1)")
    facility_rating = st.slider("Facility Rating (1-10)", min_value=1, max_value=10, value=5)
    exam_difficulty = st.slider("Exam Difficulty (1-10)", min_value=1, max_value=10, value=5)

st.divider()

# --- Predict Button & Effects ---
# Centering the predict button
_, center_col, _ = st.columns([1, 2, 1])

with center_col:
    if st.button("Predict Performance 🚀", use_container_width=True):
        
        # 1. Visual Effect: Loading Spinner
        with st.spinner("Analyzing student metrics..."):
            time.sleep(1.5) # Slight delay for dramatic effect
            
            # 2. Format the data EXACTLY matching the SVR model's features
            input_data = pd.DataFrame(
                [[age, gender, course, study_hours, class_attendance, internet_access, 
                  sleep_hours, sleep_quality, study_method, facility_rating, exam_difficulty]],
                columns=['age', 'gender', 'course', 'study_hours', 'class_attendance', 
                         'internet_access', 'sleep_hours', 'sleep_quality', 'study_method', 
                         'facility_rating', 'exam_difficulty']
            )
            
            # 3. Make the prediction
            prediction = model.predict(input_data)
        
        # 4. Visual Effect: Celebration!
        st.balloons()
        
        # 5. Display the result
        # Because SVR returns a continuous value, we format it to 2 decimal places
        st.success(f"### Predicted Score: {prediction[0]:.2f} 🎯")
