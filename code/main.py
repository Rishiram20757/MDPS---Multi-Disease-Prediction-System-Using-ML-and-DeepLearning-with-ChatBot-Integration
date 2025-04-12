import streamlit as st
from diabetes import DiabetesPredictor as ReadmissionPredictor
from diabetes_diagnosis import DiabetesDiagnosisPredictor
from heart_disease import HeartDiseasePredictor
from ckd import CKDPredictor
from sepsis import SepsisPredictor
from liver import LiverPredictor
from thyroid import ThyroidPredictor
from heart_gnn import HeartDiseaseGNNPredictor
from parkinsons_lstm import ParkinsonsLSTMPredictor
from chatbot import show_chatbot
import matplotlib.pyplot as plt
from matplotlib.animation import HTMLWriter
import tempfile
import os
import pandas as pd
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page configuration
st.set_page_config(
    page_title="Health Prediction System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dictionary of pages organized by category
PAGE_CATEGORIES = {
    "Deep Learning Models": {
        "Diabetes Readmission Risk ": {
            "predictor": ReadmissionPredictor,
            "title": "Diabetes Readmission Risk Prediction (LSTM)",
            "description": "This tool uses a deep learning model (LSTM) trained on real-world hospital data to predict the likelihood of a diabetic patient being readmitted within 30 days of a hospital visit."
        },
        "Heart Disease": {
            "predictor": HeartDiseasePredictor,
            "title": " Heart Disease Risk Prediction (RNN Based on LSTM)",
            "description": "This model uses a Long Short-Term Memory (LSTM) neural network to predict the risk of heart disease. It analyzes patterns in structured health metrics like age, cholesterol and blood pressure to provide accurate risk assessment."
        },
        "Sepsis Prediction": {
            "predictor": SepsisPredictor,
            "title": " Sepsis Prediction (LSTM)",
            "description": "The code utilizes an LSTM (Long Short-Term Memory) model to predict the likelihood of Sepsis in patients by processing both time-series and static features."
        },
        "Heart Disease (GNN)": {
            "predictor": HeartDiseaseGNNPredictor,
            "title": " Heart Disease Risk Prediction (GNN)",
            "description": "This model leverages a Graph Neural Network (GNN) to predict heart disease by modeling inter-feature relationships. It captures how various health metrics influence one another to improve diagnostic accuracy."
        },
        "Parkinson's Disease": {
            "predictor": ParkinsonsLSTMPredictor,
            "title": " Parkinson's Disease Progression (LSTM)",
            "description": "This implementation uses an LSTM (Long Short-Term Memory) neural network, specifically designed for time-series regression to predict Parkinson's disease progression (motor UPDRS scores)"
        },
        "Chronic Kidney Disease": {
            "predictor": CKDPredictor,
            "title": " Chronic Kidney Disease Prediction (DNN)",
            "description": "Deep neural network for kidney disease Prediction"
        }
    },
    "Machine Learning Models": {
        "Diabetes": {
            "predictor": DiabetesDiagnosisPredictor,
            "title": "Diabetes Diagnosis Prediction (XGBoost and SMOTE)",
            "description": "This section uses a machine learning model trained on the Pima Indians Diabetes dataset to predict the likelihood of diabetes based on key medical parameters."
        },
        "Liver Disease": {
            "predictor": LiverPredictor,
            "title": " Liver Disease Prediction (XgBoost)",
            "description": "This model uses an XGBoost classifier to predict the risk of liver disease based on key biochemical parameters. It processes liver function test data and outputs a probability of disease presence."
        },
        "Thyroid Disease": {
            "predictor": ThyroidPredictor,
            "title": " Thyroid Disease Prediction (Random Forest)",
            "description": "The code uses a Random Forest Classifier to predict thyroid disease types based on features like T3 resin uptake, total thyroxin, and basal TSH levels."
        }
    },
    "Healthcare Assistant": {
        "Medical Chatbot": {
            "title": " MedCare Chatbot",
            "description": "AI-powered chatbot for patient support and guidance"
        }
    }
}

def main():
    st.sidebar.title("🏥 MDPS")
    category = st.sidebar.radio("Select Learning Type ", list(PAGE_CATEGORIES.keys()))
    
    st.sidebar.markdown("---")
    st.sidebar.title(f"🔍 {category}")
    selection = st.sidebar.selectbox("Select disease ", list(PAGE_CATEGORIES[category].keys()))
    
    page = PAGE_CATEGORIES[category][selection]
    
    st.header(page["title"])
    st.write(page["description"])
    
    if category == "Healthcare Assistant":
        show_chatbot()  # Direct call to your chatbot function
    else:
        predictor = page["predictor"]()
        
        if selection == "Diabetes Readmission Risk ":
            show_readmission_form(predictor)
        elif selection == "Diabetes":
            show_diagnosis_form(predictor)
        elif selection == "Heart Disease":
            show_heart_form(predictor)
        elif selection == "Heart Disease (GNN)":
            show_heart_gnn_form(predictor)
        elif selection == "Chronic Kidney Disease":
            show_ckd_form(predictor)
        elif selection == "Sepsis Prediction":
            show_sepsis_form(predictor)
        elif selection == "Liver Disease":
            show_liver_form(predictor)
        elif selection == "Thyroid Disease":
            show_thyroid_form(predictor)
        elif selection == "Parkinson's Disease":
            show_parkinsons_form(predictor)

    # Add footer disclaimer for all medical tools
    st.markdown("---")
    st.caption("""
    **Disclaimer**: These tools provide predictive insights only and do not constitute medical advice. 
    Always consult with a qualified healthcare professional for diagnosis and treatment.
    """)


def show_readmission_form(predictor):
    model_exists = (
        os.path.exists(os.path.join('models', 'readmission', 'diabetes_lstm_model.h5')) and
        os.path.exists(os.path.join('models', 'readmission', 'scaler.pkl')) and
        os.path.exists(os.path.join('models', 'readmission', 'feature_names.pkl'))
    )
    
    if not model_exists:
        st.warning("Readmission model not trained yet. Please train the model first.")
        if st.button("🚀 Train Readmission Model (First Time Setup)"):
            with st.spinner("Training model... This may take several minutes..."):
                try:
                    history = predictor.train()
                    st.success("✅ Model trained and saved successfully!")
                    st.balloons()
                    st.subheader("Training Metrics")
                    metrics_df = pd.DataFrame(history)
                    st.line_chart(metrics_df)
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
            return
    
    try:
        if hasattr(predictor, 'load'):
            predictor.load()
        else:
            st.error("Predictor missing load method")
            return
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return
    
    with st.form("readmission_form"):
        st.subheader("Patient Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            time_in_hospital = st.slider("Time in Hospital (days)", 1, 14, 3)
            num_lab_procedures = st.slider("Number of Lab Procedures", 1, 100, 30)
            num_procedures = st.slider("Number of Procedures", 0, 10, 1)
            weight = st.slider("Weight (kg)", 50, 150, 75)
            
        with col2:
            num_medications = st.slider("Number of Medications", 1, 50, 10)
            number_outpatient = st.slider("Outpatient Visits (past year)", 0, 20, 0)
            number_emergency = st.slider("Emergency Visits (past year)", 0, 20, 0)
            number_diagnoses = st.slider("Number of Diagnoses", 1, 20, 5)
            
        with col3:
            number_inpatient = st.slider("Inpatient Visits (past year)", 0, 20, 0)
            A1Cresult = st.selectbox("A1C Test Result", ["None", "Normal", ">7", ">8"])
            max_glu_serum = st.selectbox("Glucose Serum Test", ["None", "Normal", ">200", ">300"])
            diabetesMed = st.selectbox("On Diabetes Medication", ["No", "Yes"])
        
        st.subheader("Medications")
        med_cols = st.columns(5)
        medications = {}
        
        with med_cols[0]:
            medications['metformin'] = st.checkbox("Metformin")
            medications['repaglinide'] = st.checkbox("Repaglinide")
            medications['nateglinide'] = st.checkbox("Nateglinide")
        
        with med_cols[1]:
            medications['chlorpropamide'] = st.checkbox("Chlorpropamide")
            medications['glimepiride'] = st.checkbox("Glimepiride")
            medications['glipizide'] = st.checkbox("Glipizide")
        
        with med_cols[2]:
            medications['glyburide'] = st.checkbox("Glyburide")
            medications['pioglitazone'] = st.checkbox("Pioglitazone")
            medications['rosiglitazone'] = st.checkbox("Rosiglitazone")
        
        with med_cols[3]:
            medications['acarbose'] = st.checkbox("Acarbose")
            medications['miglitol'] = st.checkbox("Miglitol")
            medications['insulin'] = st.checkbox("Insulin")
        
        with med_cols[4]:
            medications['glyburide-metformin'] = st.checkbox("Glyburide-Metformin")
            medications['glipizide-metformin'] = st.checkbox("Glipizide-Metformin")
        
        if st.form_submit_button("🔮 Predict Readmission Risk"):
            try:
                input_data = {
                    'time_in_hospital': time_in_hospital,
                    'num_lab_procedures': num_lab_procedures,
                    'num_procedures': num_procedures,
                    'num_medications': num_medications,
                    'number_outpatient': number_outpatient,
                    'number_emergency': number_emergency,
                    'number_inpatient': number_inpatient,
                    'number_diagnoses': number_diagnoses,
                    'weight': weight,
                    'A1Cresult_>7': 1 if A1Cresult == ">7" else 0,
                    'A1Cresult_>8': 1 if A1Cresult == ">8" else 0,
                    'A1Cresult_Normal': 1 if A1Cresult == "Normal" else 0,
                    'max_glu_serum_>200': 1 if max_glu_serum == ">200" else 0,
                    'max_glu_serum_>300': 1 if max_glu_serum == ">300" else 0,
                    'max_glu_serum_Normal': 1 if max_glu_serum == "Normal" else 0,
                    'diabetesMed_Yes': 1 if diabetesMed == "Yes" else 0,
                    'diag_1_diabetes': 1,
                    'diag_2_diabetes': 0,
                    'diag_3_diabetes': 0
                }
                
                for med, value in medications.items():
                    input_data[med] = 1 if value else 0
                
                default_features = {
                    'race_Caucasian': 1, 'gender_Male': 1, 'age_[40-50)': 1,
                    'change_Ch': 0, 'admission_type_Emergency': 1,
                    'discharge_disposition_Home': 1,
                    'admission_source_Physician Referral': 1
                }
                input_data.update(default_features)
                
                risk_score = predictor.predict(input_data)
                display_result(risk_score, "Readmission")
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

def show_diagnosis_form(predictor):
    model_exists = (
        os.path.exists(os.path.join('models', 'diagnosis', 'diabetes_model.pkl')) and
        os.path.exists(os.path.join('models', 'diagnosis', 'scaler.pkl')) and
        os.path.exists(os.path.join('models', 'diagnosis', 'feature_names.pkl'))
    )
    
    if not model_exists:
        st.warning("Diagnosis model not trained yet. Please train the model first.")
        if st.button("🚀 Train Diagnosis Model (First Time Setup)"):
            with st.spinner("Training model... This may take several minutes..."):
                try:
                    metrics = predictor.train()
                    st.success("✅ Model trained and saved successfully!")
                    st.balloons()
                    st.subheader("Training Performance")
                    st.write(f"Accuracy: {metrics['accuracy']:.2f}")
                    st.write(f"ROC AUC: {metrics['roc_auc']:.2f}")
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
            return
    
    try:
        if not predictor.loaded:
            predictor.load()
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return
    
    with st.form("diagnosis_form"):
        st.subheader("Patient Health Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies = st.number_input("Pregnancies", 0, 20, 0)
            glucose = st.number_input("Glucose (mg/dL)", 0, 300, 100)
            blood_pressure = st.number_input("Blood Pressure (mmHg)", 0, 200, 70)
            skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
            
        with col2:
            insulin = st.number_input("Insulin (μU/mL)", 0, 1000, 80)
            bmi = st.number_input("BMI", 10.0, 70.0, 25.0)
            diabetes_pedigree = st.number_input("Diabetes Pedigree", 0.0, 3.0, 0.5)
            age = st.number_input("Age", 20, 100, 30)
        
        if st.form_submit_button("🔮 Predict Diabetes Risk"):
            input_data = {
                'Pregnancies': pregnancies,
                'Glucose': glucose,
                'BloodPressure': blood_pressure,
                'SkinThickness': skin_thickness,
                'Insulin': insulin,
                'BMI': bmi,
                'DiabetesPedigreeFunction': diabetes_pedigree,
                'Age': age
            }
            
            try:
                risk_score = predictor.predict(input_data)
                display_result(risk_score, "Diabetes")
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")


def show_heart_form(predictor):
    model_exists = (
        os.path.exists(os.path.join('models', 'heart', 'heart_lstm_model.h5')) and
        os.path.exists(os.path.join('models', 'heart', 'scaler.pkl')) and
        os.path.exists(os.path.join('models', 'heart', 'encoder.pkl'))
    )
    
    if not model_exists:
        st.warning("Heart disease model not trained yet. Please train the model first.")
        if st.button("🚀 Train Heart Disease Model (First Time Setup)"):
            with st.spinner("Training model... This may take several minutes..."):
                try:
                    metrics = predictor.train()
                    st.success("✅ Model trained and saved successfully!")
                    st.balloons()
                    st.subheader("Training Performance")
                    st.write(f"Accuracy: {metrics['accuracy']:.2f}")
                    st.write(f"Loss: {metrics['loss']:.4f}")
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
            return
    
    try:
        if not predictor.loaded:
            predictor.load()
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return
    
    with st.form("heart_form"):
        st.subheader("Patient Health Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", 20, 100, 50)
            sex = st.selectbox("Sex", ["Male", "Female"])
            cp = st.selectbox("Chest Pain Type", [
                "Typical angina", 
                "Atypical angina", 
                "Non-anginal pain", 
                "Asymptomatic"
            ])
            trestbps = st.number_input("Resting Blood Pressure (mmHg)", 90, 200, 120)
            chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
            
        with col2:
            restecg = st.selectbox("Resting ECG", [
                "Normal",
                "ST-T wave abnormality",
                "Left ventricular hypertrophy"
            ])
            thalach = st.number_input("Max Heart Rate Achieved", 70, 220, 150)
            exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
            oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 6.0, 1.0)
            slope = st.selectbox("Slope of Peak Exercise ST Segment", [
                "Upsloping",
                "Flat",
                "Downsloping"
            ])
            thal = st.selectbox("Thalassemia", [
                "Normal",
                "Fixed defect",
                "Reversible defect"
            ])
        
        if st.form_submit_button("🔮 Predict Heart Disease Risk"):
            try:
                cp_mapping = {
                    "Typical angina": 1,
                    "Atypical angina": 2,
                    "Non-anginal pain": 3,
                    "Asymptomatic": 4
                }
                
                restecg_mapping = {
                    "Normal": 0,
                    "ST-T wave abnormality": 1,
                    "Left ventricular hypertrophy": 2
                }
                
                slope_mapping = {
                    "Upsloping": 1,
                    "Flat": 2,
                    "Downsloping": 3
                }
                
                thal_mapping = {
                    "Normal": 3,
                    "Fixed defect": 6,
                    "Reversible defect": 7
                }
                
                input_data = {
                    'age': age,
                    'sex': 1 if sex == "Male" else 0,
                    'cp': cp_mapping[cp],
                    'trestbps': trestbps,
                    'chol': chol,
                    'fbs': 1 if fbs == "Yes" else 0,
                    'restecg': restecg_mapping[restecg],
                    'thalach': thalach,
                    'exang': 1 if exang == "Yes" else 0,
                    'oldpeak': oldpeak,
                    'slope': slope_mapping[slope],
                    'ca': 0,
                    'thal': thal_mapping[thal]
                }
                
                risk_score = predictor.predict(input_data)
                display_result(risk_score, "Heart Disease")
                
                # Show model explanation
                st.subheader("🧠 Model Architecture")
                explanation = predictor.explain_prediction()
                if explanation:
                    st.pyplot(explanation['figure'])
                    st.markdown(f"""
                    **LSTM Model Characteristics:**
                    - Processes data sequentially
                    - Learns patterns over time steps
                    - Remembers important relationships
                    - Good for temporal medical data
                    """)
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

def show_heart_gnn_form(predictor):
    model_exists = (
        os.path.exists(os.path.join('models', 'heart_gnn', 'best_model.pt')) and
        os.path.exists(os.path.join('models', 'heart_gnn', 'scaler.pkl')) and
        os.path.exists(os.path.join('models', 'heart_gnn', 'encoders.pkl'))
    )
    
    if not model_exists:
        st.warning("Heart GNN model not trained yet. Please train the model first.")
        if st.button("🚀 Train Heart GNN Model (First Time Setup)"):
            with st.spinner("Training GNN model... This may take several minutes..."):
                try:
                    metrics = predictor.train()
                    st.success("✅ GNN model trained and saved successfully!")
                    st.balloons()
                    st.subheader("Training Performance")
                    st.write(f"Best Validation Loss: {metrics['best_val_loss']:.4f}")
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
            return
    
    try:
        predictor.load()
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return
    
    with st.form("heart_gnn_form"):
        st.subheader("Patient Health Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", 20, 100, 50)
            sex = st.selectbox("Sex", ["Male", "Female"])
            cp = st.selectbox("Chest Pain Type", [
                "Typical angina", 
                "Atypical angina", 
                "Non-anginal pain", 
                "Asymptomatic"
            ])
            trestbps = st.number_input("Resting Blood Pressure (mmHg)", 90, 200, 120)
            chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
            
        with col2:
            restecg = st.selectbox("Resting ECG", [
                "Normal",
                "ST-T wave abnormality",
                "Left ventricular hypertrophy"
            ])
            thalach = st.number_input("Max Heart Rate Achieved", 70, 220, 150)
            exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
            oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 6.0, 1.0)
            slope = st.selectbox("Slope of Peak Exercise ST Segment", [
                "Upsloping",
                "Flat",
                "Downsloping"
            ])
            thal = st.selectbox("Thalassemia", [
                "Normal",
                "Fixed defect",
                "Reversible defect"
            ])
        
        if st.form_submit_button("🔮 Predict Heart Disease Risk (GNN)"):
            try:
                cp_mapping = {
                    "Typical angina": 1,
                    "Atypical angina": 2,
                    "Non-anginal pain": 3,
                    "Asymptomatic": 4
                }
                
                restecg_mapping = {
                    "Normal": 0,
                    "ST-T wave abnormality": 1,
                    "Left ventricular hypertrophy": 2
                }
                
                slope_mapping = {
                    "Upsloping": 1,
                    "Flat": 2,
                    "Downsloping": 3
                }
                
                thal_mapping = {
                    "Normal": 3,
                    "Fixed defect": 6,
                    "Reversible defect": 7
                }
                
                input_data = {
                    'age': age,
                    'sex': 1 if sex == "Male" else 0,
                    'cp': cp_mapping[cp],
                    'trestbps': trestbps,
                    'chol': chol,
                    'fbs': 1 if fbs == "Yes" else 0,
                    'restecg': restecg_mapping[restecg],
                    'thalach': thalach,
                    'exang': 1 if exang == "Yes" else 0,
                    'oldpeak': oldpeak,
                    'slope': slope_mapping[slope],
                    'ca': 0,  # Default value
                    'thal': thal_mapping[thal]
                }
                
                risk_score = predictor.predict(input_data)
                display_result(risk_score, "Heart Disease (GNN)")
                
                # Show model explanation
                st.subheader("🕸️ Model Architecture")
                explanation = predictor.explain_prediction()
                if explanation:
                    st.pyplot(explanation['figure'])
                    st.markdown(f"""
                    **GNN Model Characteristics:**
                    - Analyzes relationships between features
                    - Uses message passing between nodes
                    - Captures how health metrics influence each other
                    - Good for complex feature interactions
                    """)
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

def show_ckd_form(predictor):
    model_exists = (
        os.path.exists(os.path.join('models', 'chronic', 'best_model.h5')) and
        os.path.exists(os.path.join('models', 'chronic', 'scaler.pkl'))
    )
    
    if not model_exists:
        st.warning("CKD model not trained yet. Please train the model first.")
        if st.button("🚀 Train CKD Model (First Time Setup)"):
            with st.spinner("Training model... This may take 2-5 minutes..."):
                try:
                    metrics = predictor.train()
                    st.success("✅ Model trained and saved successfully!")
                    st.balloons()
                    st.subheader("Training Performance")
                    st.write(f"Validation Accuracy: {metrics.get('val_accuracy', 'N/A')}")
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
            return
    
    try:
        predictor.load()
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return
    
    with st.form("ckd_form"):
        st.subheader("Patient Kidney Health Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age (years)", min_value=10, max_value=100, value=50)
            bp = st.number_input("Blood Pressure (mmHg)", min_value=50, max_value=200, value=80)
            sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025])
            al = st.slider("Albumin (0-5)", 0, 5, 0)
            su = st.slider("Sugar (0-5)", 0, 5, 0)
            rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"])
            pc = st.selectbox("Pus Cells", ["normal", "abnormal"])
            
        with col2:
            pcc = st.selectbox("Pus Cell Clumps", ["present", "notpresent"])
            ba = st.selectbox("Bacteria", ["present", "notpresent"])
            bgr = st.number_input("Blood Glucose Random (mg/dL)", 50, 500, 120)
            bu = st.number_input("Blood Urea (mg/dL)", 10, 200, 50)
            sc = st.number_input("Serum Creatinine (mg/dL)", 0.5, 15.0, 1.2, step=0.1)
            sod = st.number_input("Sodium (mEq/L)", 100, 200, 145)
            pot = st.number_input("Potassium (mEq/L)", 2.0, 10.0, 4.5, step=0.1)
            hemo = st.number_input("Hemoglobin (g/dL)", 5.0, 20.0, 12.5, step=0.1)
        
        # Additional required parameters
        with st.expander("Additional Parameters"):
            htn = st.selectbox("Hypertension", ["yes", "no"])
            dm = st.selectbox("Diabetes Mellitus", ["yes", "no"])
            cad = st.selectbox("Coronary Artery Disease", ["yes", "no"])
            appet = st.selectbox("Appetite", ["good", "poor"])
            pe = st.selectbox("Pedal Edema", ["yes", "no"])
            ane = st.selectbox("Anemia", ["yes", "no"])
            pcv = st.number_input("Packed Cell Volume (%)", 20, 60, 38)
            wc = st.number_input("White Blood Cell Count (cells/cumm)", 1000, 20000, 7800)
            rc = st.number_input("Red Blood Cell Count (millions/cmm)", 1.0, 8.0, 5.2, step=0.1)
        
        if st.form_submit_button("🔮 Predict CKD Risk"):
            try:
                # Create input dictionary with all required features
                input_data = {
                    'age': age,
                    'bp': bp,
                    'sg': sg,
                    'al': al,
                    'su': su,
                    'rbc': rbc,
                    'pc': pc,
                    'pcc': pcc,
                    'ba': ba,
                    'bgr': bgr,
                    'bu': bu,
                    'sc': sc,
                    'sod': sod,
                    'pot': pot,
                    'hemo': hemo,
                    'pcv': pcv,
                    'wc': wc,
                    'rc': rc,
                    'htn': htn,
                    'dm': dm,
                    'cad': cad,
                    'appet': appet,
                    'pe': pe,
                    'ane': ane
                }
                
                # Get prediction probability
                result = predictor.predict(input_data)
                risk_score = result['probability']  # Extract the probability for display_result
                display_result(risk_score, "Chronic Kidney Disease")
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.error(f"Make sure you're providing all required features. Model expects {predictor.expected_features} features.")
                
def show_sepsis_form(predictor):
    model_exists = (
        os.path.exists(os.path.join('models', 'sepsis', 'sepsis_lstm_model.keras')) and
        os.path.exists(os.path.join('models', 'sepsis', 'scaler.pkl')) and
        os.path.exists(os.path.join('models', 'sepsis', 'feature_order.pkl'))
    )
    
    if not model_exists:
        st.warning("Sepsis model not trained yet. Please train the model first.")
        if st.button("🚀 Train Sepsis Model (First Time Setup)"):
            with st.spinner("Training model... This may take several minutes..."):
                try:
                    history = predictor.train()
                    st.success("✅ Model trained and saved successfully!")
                    st.balloons()
                    st.subheader("Training Metrics")
                    metrics_df = pd.DataFrame(history)
                    st.line_chart(metrics_df[['auc', 'val_auc']])
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
            return
    
    try:
        predictor.load()
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return
    
    with st.form("sepsis_form"):
        st.subheader("Patient ICU Time-Series Data")
        st.info("Enter at least 6 hours of ICU data for accurate prediction")
        
        # Static features
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 18, 100, 50)
            gender = st.selectbox("Gender", ["Male", "Female"])
        with col2:
            hosp_adm_time = st.number_input("Hours since hospital admission", 0, 720, 24)
            iculos = st.number_input("Hours in ICU", 0, 720, 6)
        
        # Time-series features - ensure all required features are included
        time_points = st.slider("Number of time points to enter", 6, 24, 6)
        
        # Initialize with all required features
        time_series_data = {
            'HR': [], 'O2Sat': [], 'Temp': [], 'SBP': [], 'MAP': [], 'DBP': [], 
            'Resp': [], 'WBC': [], 'Glucose': [], 'Lactate': [], 'Creatinine': []
            # Add other required features here if needed
        }
        
        # Create input for each time point
        for i in range(time_points):
            st.markdown(f"### Time Point {i+1} (Hour {i+1})")
            cols = st.columns(4)
            
            with cols[0]:
                time_series_data['HR'].append(st.number_input(f"Heart Rate (bpm) - Hour {i+1}", 40, 200, 80, key=f"hr_{i}"))
                time_series_data['O2Sat'].append(st.number_input(f"O2 Saturation (%) - Hour {i+1}", 70, 100, 98, key=f"o2_{i}"))
                
            with cols[1]:
                time_series_data['Temp'].append(st.number_input(f"Temperature (°C) - Hour {i+1}", 32.0, 42.0, 37.0, key=f"temp_{i}"))
                time_series_data['SBP'].append(st.number_input(f"Systolic BP (mmHg) - Hour {i+1}", 60, 250, 120, key=f"sbp_{i}"))
                
            with cols[2]:
                time_series_data['MAP'].append(st.number_input(f"Mean Arterial Pressure - Hour {i+1}", 40, 150, 85, key=f"map_{i}"))
                time_series_data['DBP'].append(st.number_input(f"Diastolic BP (mmHg) - Hour {i+1}", 30, 150, 80, key=f"dbp_{i}"))
                
            with cols[3]:
                time_series_data['Resp'].append(st.number_input(f"Respiratory Rate - Hour {i+1}", 5, 40, 16, key=f"resp_{i}"))
                time_series_data['WBC'].append(st.number_input(f"WBC Count (10^3/uL) - Hour {i+1}", 0.1, 50.0, 7.5, key=f"wbc_{i}"))
        
        if st.form_submit_button("🔮 Predict Sepsis Risk"):
            try:
                # Prepare input with all 38 features
                input_data = {
                    'Age': age,
                    'Gender': 1 if gender == "Male" else 0,
                    'HospAdmTime': hosp_adm_time,
                    'ICULOS': iculos
                }
                
                # Add time-series data (missing features will be set to 0)
                for feature in time_series_data:
                    input_data[feature] = time_series_data[feature]
                
                # Set missing features to 0
                for feature in predictor.all_features:
                    if feature not in input_data:
                        input_data[feature] = 0
                
                risk_score = predictor.predict(input_data)
                display_result(risk_score, "Sepsis")
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

def show_liver_form(predictor):
    model_exists = (
        os.path.exists(os.path.join('models', 'liver', 'liver_model.pkl')) and
        os.path.exists(os.path.join('models', 'liver', 'scaler.pkl'))
    )
    
    if not model_exists:
        st.warning("Liver model not trained yet. Please train the model first.")
        if st.button("🚀 Train Liver Model (First Time Setup)"):
            with st.spinner("Training model... This may take a minute..."):
                try:
                    predictor.train()
                    st.success("✅ Model trained and saved successfully!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
            return
    
    try:
        predictor.load()
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return
    
    with st.form("liver_form"):
        st.subheader("Liver Function Test Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", 10, 100, 45)
            gender = st.selectbox("Gender", ["Male", "Female"])
            total_bilirubin = st.number_input("Total Bilirubin (mg/dL)", 0.0, 30.0, 0.7)
            direct_bilirubin = st.number_input("Direct Bilirubin (mg/dL)", 0.0, 20.0, 0.1)
            alk_phosphatase = st.number_input("Alkaline Phosphatase (IU/L)", 50, 500, 150)
            
        with col2:
            alamine_aminotransferase = st.number_input("ALT (IU/L)", 10, 2000, 25)
            aspartate_aminotransferase = st.number_input("AST (IU/L)", 10, 3000, 30)
            total_proteins = st.number_input("Total Proteins (g/dL)", 2.0, 10.0, 6.5)
            albumin = st.number_input("Albumin (g/dL)", 0.1, 5.0, 3.5)
            # Albumin_and_Globulin_Ratio will be calculated automatically
        
        if st.form_submit_button("🔮 Predict Liver Disease Risk"):
            input_data = {
                'Age': age,
                'Gender': 1 if gender == "Male" else 0,
                'Total_Bilirubin': total_bilirubin,
                'Direct_Bilirubin': direct_bilirubin,
                'Alkaline_Phosphotase': alk_phosphatase,
                'Alamine_Aminotransferase': alamine_aminotransferase,
                'Aspartate_Aminotransferase': aspartate_aminotransferase,
                'Total_Protiens': total_proteins,
                'Albumin': albumin
                # Ratio will be calculated in predict()
            }
            
            try:
                risk_score = predictor.predict(input_data)
                display_result(risk_score, "Liver Disease")
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

def show_thyroid_form(predictor):
    """Displays the thyroid prediction form with model training option if needed"""
    
    # Helper function for displaying results (kept inside to maintain single-function structure)
    def display_thyroid_results(results):
        st.subheader("📊 Prediction Results")
        
        # Create progress bars for each class
        for condition, prob in results.items():
            percent = prob * 100
            st.write(f"**{condition.capitalize()}**:")
            st.progress(int(percent))
            st.write(f"{percent:.1f}% probability")
        
        st.subheader("📝 Interpretation")
        if results.get('hyper', 0) > 0.5:
            st.error("""
            **Hyperthyroidism likely**:
            - Consult endocrinologist
            - Check for Graves' disease
            - Monitor heart rate/blood pressure
            """)
        elif results.get('hypo', 0) > 0.5:
            st.error("""
            **Hypothyroidism likely**:
            - TSH retest recommended
            - Check for Hashimoto's
            - Consider levothyroxine
            """)
        else:
            st.success("""
            **Normal thyroid function**:
            - Routine monitoring suggested
            - Recheck in 6-12 months
            - Watch for symptoms
            """)

    # Main form logic
    model_exists = (
        os.path.exists(os.path.join('models', 'thyroid', 'thyroid_model.pkl')) and
        os.path.exists(os.path.join('models', 'thyroid', 'scaler.pkl'))
    )
    
    if not model_exists:
        st.warning("Thyroid model not trained yet. Please train the model first.")
        if st.button("🚀 Train Thyroid Model (First Time Setup)"):
            with st.spinner("Training model... This may take a minute..."):
                try:
                    result = predictor.train()
                    st.success(f"""
                    ✅ Model trained successfully!
                    - Training samples: {result['train_samples']}
                    - Test samples: {result['test_samples']}
                    - Class distribution: {result['class_distribution']}
                    """)
                    st.balloons()
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
            return
    
    try:
        predictor.load()
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return
    
    with st.form("thyroid_form"):
        st.subheader("Thyroid Function Test Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            t3_resin = st.number_input("T3 Resin Uptake (%)", 0.0, 100.0, 95.0)
            total_thyroxin = st.number_input("Total Thyroxin (μg/dL)", 0.0, 30.0, 8.2)
            
        with col2:
            total_triiodo = st.number_input("Total Triiodothyronine (ng/dL)", 0.0, 500.0, 120.0)
            basal_tsh = st.number_input("Basal TSH (μIU/mL)", 0.0, 100.0, 2.5)
            max_tsh_diff = st.number_input("Max TSH Difference After TRH", 0.0, 50.0, 1.5)
        
        if st.form_submit_button("🔮 Predict Thyroid Status"):
            input_data = {
                'T3_resin_uptake': t3_resin,
                'Total_thyroxin': total_thyroxin,
                'Total_triiodothyronine': total_triiodo,
                'Basal_TSH': basal_tsh,
                'Max_TSH_diff': max_tsh_diff
            }
            
            try:
                results = predictor.predict(input_data)
                display_thyroid_results(results)
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

def show_parkinsons_form(predictor):
    model_exists = (
        os.path.exists(os.path.join('models', 'parkinsons_lstm', 'parkinsons_lstm.keras')) and
        os.path.exists(os.path.join('models', 'parkinsons_lstm', 'scaler.pkl')) and
        os.path.exists(os.path.join('models', 'parkinsons_lstm', 'imputer.pkl'))
    )
    
    if not model_exists:
        st.warning("Parkinson's model not trained yet. Please train the model first.")
        if st.button("🚀 Train Parkinson's Model (First Time Setup)"):
            with st.spinner("Training model... This may take several minutes..."):
                try:
                    metrics = predictor.train()
                    st.success("✅ Model trained and saved successfully!")
                    st.balloons()
                    st.subheader("Training Performance")
                    st.write(f"Test RMSE: {metrics['test_rmse']:.2f}")
                    
                    # Show training history plot
                    history_img = os.path.join('models', 'parkinsons_lstm', 'training_history.png')
                    if os.path.exists(history_img):
                        st.image(history_img)
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
            return
    
    try:
        # More robust loading check
        if not hasattr(predictor, 'model') or predictor.model is None:
            predictor.load()
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return
    
    with st.form("parkinsons_form"):
        st.subheader("Patient Voice Measurement Time Series")
        st.info("Enter at least 3 time points for accurate prediction")
        
        # Time series input
        time_points = []
        for i in range(3):  # Using 3 time steps as defined in the model
            with st.expander(f"Time Point {i+1}"):
                col1, col2 = st.columns(2)
                with col1:
                    jitter = st.number_input(f"Jitter (%) - TP{i+1}", 0.0, 10.0, 0.5)
                    shimmer = st.number_input(f"Shimmer (dB) - TP{i+1}", 0.0, 5.0, 0.3)
                    nhr = st.number_input(f"NHR - TP{i+1}", 0.0, 1.0, 0.05)
                    hnr = st.number_input(f"HNR - TP{i+1}", 0.0, 30.0, 20.0)
                with col2:
                    rpde = st.number_input(f"RPDE - TP{i+1}", 0.0, 1.0, 0.5)
                    dfa = st.number_input(f"DFA - TP{i+1}", 0.5, 1.0, 0.7)
                    ppe = st.number_input(f"PPE - TP{i+1}", 0.0, 1.0, 0.2)
                    last_updrs = st.number_input(f"Motor UPDRS - TP{i+1}", 0, 100, 20)
                
            time_points.append({
                'Jitter(%)': jitter,
                'Shimmer(dB)': shimmer,
                'NHR': nhr,
                'HNR': hnr,
                'RPDE': rpde,
                'DFA': dfa,
                'PPE': ppe,
                'motor_UPDRS': last_updrs
            })
        
        current_updrs = st.number_input("Current Motor UPDRS Score", 0, 100, 20)
        
        if st.form_submit_button("🔮 Predict Progression"):
            try:
                input_data = {
                    'data': time_points,
                    'last_updrs': current_updrs
                }
                
                result = predictor.predict(input_data)
                
                st.subheader("📊 Prediction Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted UPDRS", f"{result['prediction']:.1f}")
                    st.metric("Trend", result['severity'], 
                             delta=f"{result['trend']:.1f} points")
                with col2:
                    st.metric("Confidence", f"{result['confidence']*100:.0f}%")
                
                st.subheader("📝 Clinical Interpretation")
                if result['severity'] == "Progressing":
                    st.error("""
                    **Disease appears to be progressing**:
                    - Consider medication adjustment
                    - Physical therapy recommended
                    - Monitor speech and motor function
                    - Schedule follow-up soon
                    """)
                elif result['severity'] == "Improving":
                    st.success("""
                    **Symptoms appear to be improving**:
                    - Current treatment may be effective
                    - Continue monitoring
                    - Maintain therapy regimen
                    """)
                else:
                    st.info("""
                    **Symptoms appear stable**:
                    - Continue current treatment
                    - Regular monitoring recommended
                    - Watch for any changes
                    """)
                
                # Show model explanation
                st.subheader("🧠 Model Explanation")
                explanation = predictor.explain_prediction()
                if explanation:
                    st.pyplot(explanation['figure'])
                    st.markdown(explanation['description'])
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

def display_result(risk_score, prediction_type):
    risk_percent = risk_score * 100
    st.subheader("📊 Prediction Results")
    
    risk_meter = st.progress(0)
    if risk_percent < 30:
        risk_color = "green"
        risk_message = "Low Risk"
    elif risk_percent < 60:
        risk_color = "orange"
        risk_message = "Moderate Risk"
    else:
        risk_color = "red"
        risk_message = "High Risk"
    
    risk_meter.progress(int(risk_percent))
    
    st.markdown(f"""
    <div style="background-color:#f0f2f6;padding:20px;border-radius:10px">
        <h3 style="color:{risk_color};text-align:center">
            {prediction_type} Risk: <b>{risk_percent:.1f}%</b> ({risk_message})
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("📝 Recommendations")
    if prediction_type == "Heart Disease":
        if risk_percent > 60:
            st.error("""
            - Consult a cardiologist immediately
            - Consider stress testing
            - Monitor blood pressure regularly
            - Lifestyle changes (diet, exercise)
            """)
        elif risk_percent > 30:
            st.warning("""
            - Schedule cardiac screening
            - Monitor cholesterol levels
            - Consider dietary changes
            - Regular exercise recommended
            """)
        else:
            st.success("""
            - Maintain healthy lifestyle
            - Annual check-ups recommended
            - Continue preventive measures
            """)
    elif prediction_type == "Chronic Kidney Disease":
        if risk_percent > 60:
            st.error("""
            - Nephrology consultation recommended
            - Monitor kidney function tests
            - Control blood pressure
            - Limit protein intake
            - Avoid nephrotoxic medications
            """)
        elif risk_percent > 30:
            st.warning("""
            - Regular kidney function monitoring
            - Maintain hydration
            - Control diabetes if present
            - Reduce salt intake
            - Annual urine protein check
            """)
        else:
            st.success("""
            - Maintain healthy lifestyle
            - Regular check-ups
            - Stay hydrated
            - Monitor blood pressure
            - Avoid excessive NSAIDs
            """)
    elif prediction_type == "Sepsis":
        if risk_percent > 60:
            st.error("""
            - **Immediate action required**
            - Start sepsis protocol (blood cultures, antibiotics, fluids)
            - Monitor for organ dysfunction
            - Consider ICU admission
            - Check lactate levels frequently
            """)
        elif risk_percent > 30:
            st.warning("""
            - **High suspicion for sepsis**
            - Repeat assessment in 1 hour
            - Monitor vitals closely
            - Consider early antibiotics
            - Check inflammatory markers
            """)
        else:
            st.success("""
            - **Low risk currently**
            - Continue monitoring
            - Watch for clinical deterioration
            - Reassess if condition changes
            - Maintain infection prevention measures
            """)
    else:  # Diabetes or Readmission
        if risk_percent > 60:
            st.error("""
            - Consult a doctor immediately
            - Consider additional testing
            - Monitor glucose levels regularly
            - Lifestyle changes recommended
            """)
        elif risk_percent > 30:
            st.warning("""
            - Lifestyle changes recommended
            - Regular screening advised
            - Maintain healthy weight
            - Monitor symptoms
            """)
        else:
            st.success("""
            - Continue healthy habits
            - Regular check-ups recommended
            - Preventive measures advised
            """)



if __name__ == "__main__":
    main()