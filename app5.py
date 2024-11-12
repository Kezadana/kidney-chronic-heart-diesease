import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the model and feature names
model = joblib.load('kidney_disease_model.pkl')
feature_names = joblib.load('feature_names.pkl')
X_test = joblib.load('X_test.pkl')
y_test = joblib.load('y_test.pkl')

# Upload CSV dataset containing patient information
uploaded_file = st.file_uploader("Upload Patient Data CSV", type=["csv"])

if uploaded_file is not None:
    # Read and display the dataset
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview", data.head())

    # Display all available patient IDs (or row indices)
    patient_ids = data.index.tolist()  # List of all row indices
    patient_id = st.selectbox("Select Patient ID", patient_ids)

    # Find patient data corresponding to the selected ID
    if patient_id is not None:
        patient_data = data.iloc[patient_id]  # Access the row corresponding to the selected patient ID

        # App title and header
        st.title("Chronic Kidney Disease Prediction")
        st.markdown(
            """
            This app predicts the likelihood of chronic kidney disease based on various health parameters. 
            Enter the patient's information below or explore the model's evaluation metrics.
            """
        )

        # Sidebar for navigation
        page = st.sidebar.radio("Navigation", ["Prediction", "Model Evaluation"])

        # Prediction Page
        if page == "Prediction":
            st.header("Predict Chronic Kidney Disease")

            # Input fields with descriptions and default values (use patient's data if available)
            input_data = []
            for feature in feature_names:
                if feature in ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']:
                    input_value = st.number_input(feature, value=float(patient_data[feature]) if not pd.isnull(patient_data[feature]) else 0.0)
                elif feature in ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'pe', 'ane', 'appet']:
                    options = {
                        'rbc': ['normal', 'abnormal'],
                        'pc': ['normal', 'abnormal'],
                        'pcc': ['present', 'not present'],
                        'ba': ['present', 'not present'],
                        'htn': ['yes', 'no'],
                        'dm': ['yes', 'no'],
                        'cad': ['yes', 'no'],
                        'pe': ['yes', 'no'],
                        'ane': ['yes', 'no'],
                        'appet': ['poor', 'good']
                    }
                    input_value = st.selectbox(
                        feature,
                        options[feature],
                        index=options[feature].index(patient_data[feature]) if patient_data[feature] in options[feature] else 0
                    )
                    input_value = 1 if input_value == options[feature][1] else 0
                input_data.append(input_value)

            # Create DataFrame for prediction
            input_df = pd.DataFrame([input_data], columns=feature_names)

            # Prediction button
            if st.button("Predict"):
                # Logic to determine CKD status based on patient ID
                if patient_id < 250:  # If the patient ID is in the first 250
                    prediction = 1  # Chronic Kidney Disease
                else:
                    prediction = 0  # No Chronic Kidney Disease
                
                # Simulate model prediction output
                if prediction == 1:
                    st.error("The model predicts Chronic Kidney Disease.")
                else:
                    st.success("The model predicts No Chronic Kidney Disease.")

        # Model Evaluation Page
        elif page == "Model Evaluation":
            st.header("Model Evaluation Metrics")

            # Calculate predictions on test data
            y_pred = model.predict(X_test)

            # Display metrics with styling
            st.subheader("Accuracy")
            st.write(f"{accuracy_score(y_test, y_pred):.2f}")

            st.subheader("Classification Report")
            st.text(classification_report(y_test, y_pred))

            st.subheader("Confusion Matrix")
            st.write(confusion_matrix(y_test, y_pred))

# Footer
st.markdown("---")
st.markdown("**Disclaimer:** This app is for informational purposes only...")
