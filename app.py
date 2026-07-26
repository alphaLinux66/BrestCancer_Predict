import streamlit as st
import pandas as pd
import joblib

model = joblib.load("cancer_model.pkl")
scaler = joblib.load("scaler.pkl")

features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
    'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
    'smoothness_worst', 'compactness_worst', 'concavity_worst',
    'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

st.title("🧠 Breast Cancer Prediction")
st.markdown("Fill the form below with patient test values.")

user_input = {}
for col in features:
    user_input[col] = st.number_input(col.replace('_', ' ').capitalize(), min_value=0.0, step=0.01)

if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]

    st.subheader("Result:")
    st.success("Benign (Non-Cancerous)") if prediction == 0 else st.error("Malignant (Cancer Detected)")
    st.write(f"Confidence (Benign): {proba[0]*100:.2f}%")
    st.write(f"Confidence (Malignant): {proba[1]*100:.2f}%")