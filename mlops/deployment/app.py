import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="Sandhya777/tourism_package_prediction_model",
    filename="best_tourism_package_prediction_v1.joblib"
)
model = joblib.load(model_path)

# Streamlit UI for Insurance Charges Prediction
st.title("Tourism Package Prediction App")
st.write("""
This application predicts the **Tourism Package Prediction** based on personal and lifestyle details.
Please enter the required information below to get a prediction.
""")

# User input
age= st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
typeofcontact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
citytier = st.selectbox("City Tier", [1, 2, 3])
durationofpitch = st.number_input("Duration of Pitch", min_value=1, max_value=100, value=10, step=1)
occupation= st.selectbox("Occupation", ["Salaried", "Freelancer"])
gender = st.selectbox("Gender", ["Male", "Female"])
numberofpersonvisiting = st.number_input("Number of People Visiting", min_value=1, max_value=10, value=2, step=1)
numberoffollowups = st.number_input("Number of Follow-ups", min_value=1, max_value=10, value=2, step=1)
productpitched= st.selectbox("Product Pitched", ["Basic", "Deluxe","Standard","King","Super Deluxe"])
preferredpropertystar= st.number_input("Preferred Property Star", min_value=1, max_value=5, value=3, step=1)
maritalstatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
numberoftrips = st.number_input("Number of Trips", min_value=1, max_value=10, value=2, step=1)
passport = st.selectbox("Passport", [0, 1])
pitchsatisfactionscore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3, step=1)
owncar = st.selectbox("Own Car", [0, 1])
numberofchildrenvisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0, step=1)
designation = st.selectbox("Designation", ["Executive", "Managerial", "Professional", "Other"])
monthlyincome = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=5000, step=100)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': typeofcontact,
    'CityTier': citytier,
    'DurationOfPitch': durationofpitch,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': numberofpersonvisiting,
    'NumberOfFollowups': numberoffollowups,
    'ProductPitched': productpitched,
    'PreferredPropertyStar': preferredpropertystar,
    'MaritalStatus': maritalstatus,
    'NumberOfTrips': numberoftrips,
    'Passport': passport,
    'PitchSatisfactionScore': pitchsatisfactionscore,
    'OwnCar': owncar,
    'NumberOfChildrenVisiting': numberofchildrenvisiting,
    'Designation': designation,
    'MonthlyIncome': monthlyincome
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "Package Taken" if prediction == 1 else "Package not Taken"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
