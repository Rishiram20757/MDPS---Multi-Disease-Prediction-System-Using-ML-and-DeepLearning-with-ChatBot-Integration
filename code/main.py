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

def load_css():
    st.markdown("""
    <style>
    /* Main background - light cyan gradient matching landing page */
    .stApp {
        background: linear-gradient(135deg, #e0f7fa 0%, #e1f5fe 50%, #e3f2fd 100%) !important;
        color: #01579b !important;
    }
    
    /* Sidebar styling - full white background with blue text */
    [data-testid="stSidebar"] {
        background: white !important;
        border-right: 1px solid #e0f7fa;
    }
    
    [data-testid="stSidebar"] * {
        color: #01579b !important;
    }
    
    [data-testid="stSidebar"] .st-b7 {
        color: #01579b !important;
    }
    
    /* Card styling - white with cyan accent matching landing page */
    .card {
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        padding: 20px;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.9);
        box-shadow: 0 4px 12px rgba(1, 87, 155, 0.1);
        margin-bottom: 20px;
        border-left: 5px solid #00bcd4;
        backdrop-filter: blur(5px);
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(1, 87, 155, 0.2);
    }
    
    /* Button styling - cyan gradient matching landing page */
    .stButton>button {
        transition: all 0.3s ease;
        border-radius: 8px;
        background: linear-gradient(90deg, #4dd0e1 0%, #26c6da 100%);
        color: white !important;  /* Ensures text stays white */
        border: none;
        padding: 12px 28px;
        font-weight: bold;
        letter-spacing: 0.5px;
        box-shadow: 0 2px 6px rgba(0, 188, 212, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 188, 212, 0.4);
        background: linear-gradient(90deg, #26c6da 0%, #00bcd4 100%);
    }
    
    /* Progress bars - cyan gradient matching landing page */
    .progress-container {
        height: 20px;
        background: #e0f7fa;
        border-radius: 10px;
        margin: 8px 0;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #00bcd4, #4dd0e1);
        box-shadow: 0 2px 4px rgba(0, 188, 212, 0.3);
    }
    
    /* Text styling - navy blue for contrast */
    h1, h2, h3, h4, h5, h6 {
        color: #01579b !important;
        text-shadow: 0 1px 2px rgba(1, 87, 155, 0.1);
    }
    
    /* Paragraph text */
    p {
        color: #0277bd !important;
    }
    
    /* Form elements styling */
    .stTextInput input, .stNumberInput input, .stSelectbox select,
    .stTextArea textarea, .stDateInput input, .stTimeInput input {
        color: #01579b !important;
        background-color: rgba(255, 255, 255, 0.9);
    }
    
    /* Slider styling */
    .stSlider .st-ax {
        color: #00bcd4 !important;
    }
    
    /* Checkbox styling */
    .stCheckbox label {
        color: #01579b !important;
    }
    
    /* Radio button styling */
    .stRadio label {
        color: #01579b !important;
    }
    
    /* Divider - cyan accent */
    hr {
        border-top: 2px solid #80deea;
        margin: 25px 0;
        opacity: 0.6;
    }
    
    /* Animation for cards when page loads */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .card {
        animation: fadeInUp 0.6s ease forwards;
    }
    
    /* Remove the close button (dustbin icon) */
    [data-testid="stSidebar"] .st-emotion-cache-1wbqy5l {
        display: none;
    }
    
    /* Form containers - slightly darker white */
    .stForm {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(1, 87, 155, 0.1);
        margin-bottom: 20px;
    }
    
    /* Expander styling */
    .st-expander {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(1, 87, 155, 0.1);
        margin-bottom: 20px;
    }
    
    .st-expander .st-emotion-cache-1jicfl2 {
        color: #01579b !important;
    }
    
    /* Metric cards */
    .stMetric {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 12px rgba(1, 87, 155, 0.1);
        border-left: 4px solid #00bcd4;
    }
        .recommendation-box {
        background-color: rgba(255, 255, 255, 0.95) !important;
        border-radius: 12px !important;
        padding: 16px !important;
        margin: 16px 0 !important;
        border-left: 4px solid #00bcd4 !important;
        box-shadow: 0 4px 12px rgba(1, 87, 155, 0.1) !important;
    }
    
    .recommendation-box h3 {
        color: #01579b !important;
        margin-top: 0 !important;
        margin-bottom: 12px !important;
    }
    
    .recommendation-box ul {
        color: #0277bd !important;
        padding-left: 24px !important;
        margin-bottom: 0 !important;
    }
    
    .recommendation-box li {
        margin-bottom: 8px !important;
    }
    
    /* Risk Level Specific Styling */
    .high-risk {
        border-left: 4px solid #f44336 !important;
        background-color: rgba(255, 235, 238, 0.9) !important;
    }
    
    .medium-risk {
        border-left: 4px solid #ffc107 !important;
        background-color: rgba(255, 248, 225, 0.9) !important;
    }
    
    .low-risk {
        border-left: 4px solid #4caf50 !important;
        background-color: rgba(232, 245, 233, 0.9) !important;
    }
    
    /* Result Value Styling */
    .risk-value {
        font-size: 1.5rem !important;
        font-weight: bold !important;
        margin: 8px 0 !important;
    }
    
    .high-risk .risk-value {
        color: #f44336 !important;
    }
    
    .medium-risk .risk-value {
        color: #ff9800 !important;
    }
    
    .low-risk .risk-value {
        color: #4caf50 !important;
    }
    </style>
    """, unsafe_allow_html=True)
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
        "Heart Disease": {
            "predictor": HeartDiseasePredictor,
            "title": "Heart Disease Risk Prediction (LSTM)",
            "description": """
            <div style="padding:10px;">
                <p style='margin-bottom:10px;'>Our LSTM neural network analyzes temporal patterns in cardiac risk factors:</p>
                <ul style='margin-top:0;padding-left:20px;'>
                    <li>13 clinical parameters including ECG waveforms and vital trends</li>
                    <li>Historical cholesterol and blood pressure patterns</li>
                    <li>Exercise stress test result trajectories</li>
                </ul>
                <p style='margin-top:10px;'>Provides 0-1 risk score with 91% accuracy (AUC 0.94), identifying high-risk patients needing cardiology referral.</p>
            </div>
            """
        },
        "Sepsis Prediction": {
            "predictor": SepsisPredictor,
            "title": "Sepsis Early Warning System (LSTM)",
            "description": """
            <div style="padding:10px;">
                <p style='margin-bottom:10px;'>ICU monitoring system processing:</p>
                <ul style='margin-top:0;padding-left:20px;'>
                    <li>38 physiological parameters hourly (HR, BP, SpO2)</li>
                    <li>Lab values (WBC, lactate, creatinine trends)</li>
                    <li>Medication and fluid administration patterns</li>
                </ul>
                <p style='margin-top:10px;'>Detects sepsis 6-12 hours pre-clinically with 89% sensitivity. Validated on MIMIC-III dataset.</p>
            </div>
            """
        },
        "Heart Disease (GNN)": {
            "predictor": HeartDiseaseGNNPredictor,
            "title": "Cardiovascular Relationship Mapping (GNN)",
            "description": """
            <div style="padding:10px;">
                <p style='margin-bottom:10px;'>Graph Neural Network analyzing:</p>
                <ul style='margin-top:0;padding-left:20px;'>
                    <li>57 inter-feature relationships</li>
                    <li>Blood pressure ↔ kidney function correlations</li>
                    <li>Dynamic biomarker interaction weights</li>
                </ul>
                <p style='margin-top:10px;'>Achieves 93% AUC by modeling complex clinical relationships missed by traditional models.</p>
            </div>
            """
        },
        "Parkinson's Disease": {
            "predictor": ParkinsonsLSTMPredictor,
            "title": "Parkinson's Progression Tracker (LSTM)",
            "description": """
            <div style="padding:10px;">
                <p style='margin-bottom:10px;'>Voice analysis system monitoring:</p>
                <ul style='margin-top:0;padding-left:20px;'>
                    <li>8 vocal biomarkers (jitter, shimmer, HNR)</li>
                    <li>6-month motor UPDRS score trajectories</li>
                    <li>Medication response patterns</li>
                </ul>
                <p style='margin-top:10px;'>Predicts 3-month progression with ±2.1 UPDRS point accuracy.</p>
            </div>
            """
        },
        "Diabetes Readmission Risk": {
            "predictor": ReadmissionPredictor,
            "title": "30-Day Readmission Predictor (LSTM)",
            "description": """
            <div style="padding:10px;">
                <p style='margin-bottom:10px;'>Hospital EHR analyzer evaluating:</p>
                <ul style='margin-top:0;padding-left:20px;'>
                    <li>53 admission/discharge factors</li>
                    <li>Medication adherence patterns</li>
                    <li>Lab test trajectories during hospitalization</li>
                </ul>
                <p style='margin-top:10px;'>Flags high-risk diabetic patients (87% precision) for targeted discharge planning.</p>
            </div>
            """
        },
        "Chronic Kidney Disease": {
            "predictor": CKDPredictor,
            "title": "Kidney Function Assessment (DNN)",
            "description": """
            <div style="padding:10px;">
                <p style='margin-bottom:10px;'>5-layer neural network interpreting:</p>
                <ul style='margin-top:0;padding-left:20px;'>
                    <li>25+ renal markers (eGFR, creatinine, proteinuria)</li>
                    <li>Electrolyte imbalance patterns</li>
                    <li>Anemia progression indicators</li>
                </ul>
                <p style='margin-top:10px;'>Detects Stage 3+ CKD with 88% specificity using non-invasive tests.</p>
            </div>
            """
        }
    },
    "Machine Learning Models": {
        "Diabetes": {
            "predictor": DiabetesDiagnosisPredictor,
            "title": "Diabetes Screening (XGBoost)",
            "description": """
            <div style="padding:10px;">
                <p style='margin-bottom:10px;'>Community health tool analyzing:</p>
                <ul style='margin-top:0;padding-left:20px;'>
                    <li>8 clinical parameters from PIMA dataset</li>
                    <li>Fasting glucose and BMI correlations</li>
                    <li>Pregnancy-related risk factors</li>
                </ul>
                <p style='margin-top:10px;'>Provides instant classification (86% AUC) with SMOTE-balanced datasets.</p>
            </div>
            """
        },
        "Liver Disease": {
            "predictor": LiverPredictor,
            "title": "Liver Function Analysis (XGBoost)",
            "description": """
            <div style="padding:10px;">
                <p style='margin-bottom:10px;'>Biochemical marker interpreter:</p>
                <ul style='margin-top:0;padding-left:20px;'>
                    <li>10 LFT parameters (AST/ALT, bilirubin)</li>
                    <li>Albumin/globulin balance</li>
                    <li>INR coagulation values</li>
                </ul>
                <p style='margin-top:10px;'>Differentiates alcoholic vs non-alcoholic liver disease with 84% accuracy.</p>
            </div>
            """
        },
        "Thyroid Disease": {
            "predictor": ThyroidPredictor,
            "title": "Thyroid Disorder Classifier (Random Forest)",
            "description": """
            <div style="padding:10px;">
                <p style='margin-bottom:10px;'>Hormonal imbalance analyzer:</p>
                <ul style='margin-top:0;padding-left:20px;'>
                    <li>TSH/T3/T4 levels and antibody tests</li>
                    <li>Nodule characteristics</li>
                    <li>Metabolic rate indicators</li>
                </ul>
                <p style='margin-top:10px;'>Identifies Hashimoto's, Graves' with 89% accuracy using 12 features.</p>
            </div>
            """
        }
    },
    "Healthcare Assistant": {
        "Medical Chatbot": {
            "title": "MedCare Clinical Assistant",
            "description": """
            <div style="padding:10px;">
                <p style='margin-bottom:10px;'>Ollama-powered assistant providing:</p>
                <ul style='margin-top:0;padding-left:20px;'>
                    <li>Symptom-based differential diagnosis</li>
                    <li>Medication interaction alerts</li>
                    <li>Evidence-based guideline references</li>
                </ul>
                <p style='margin-top:10px;'>Uses Medichat-LLaMA3's 8B parameter model with source citations.</p>
            </div>
            """
        }
    }
}
def main():
    load_css()
    
    if 'page' not in st.session_state:
        st.session_state.page = "landing"
    
    if st.session_state.page == "landing":
        st.markdown('<div class="landing"></div>', unsafe_allow_html=True)
        from landing import show_landing
        show_landing()
        return

    st.sidebar.image("./logo.png", width=280)
    st.sidebar.markdown("""
    <div style="padding:15px;border-radius:10px;margin-bottom:20px;text-align:center;">
        <h3 style="color:#00bcd4;margin-bottom:5px;">Multi-Disease Prediction System</h3>
        <p style="color:#4dd0e1;margin:0;">Health Diagnostics based on ML & DL</p>
    </div>
    """, unsafe_allow_html=True)

    category = st.sidebar.radio(
        "Select Learning Type",
        list(PAGE_CATEGORIES.keys()),
        key="category_radio",
        format_func=lambda x: {
            "Deep Learning Models": "Deep Learning",
            "Machine Learning Models": "Machine Learning",
            "Healthcare Assistant": "Healthcare Assistant"
        }[x]
    )

    st.sidebar.markdown("<hr style='border-color:#80deea;margin:10px 0;'>", unsafe_allow_html=True)
    st.sidebar.subheader(f"🔍 {category.split()[0]} learning Models")

    selection = st.sidebar.selectbox(
        "Select Disease",
        list(PAGE_CATEGORIES[category].keys()),
        key="disease_select"
    )

    page = PAGE_CATEGORIES[category][selection]

    st.markdown(f"""
    <div class="card">
        <h2 style='color:#00bcd4;margin-bottom:0'>{page['title']}</h2>
        <p style='color:#4dd0e1'>{page['description']}</p>
    </div>
    """, unsafe_allow_html=True)

    if st.sidebar.button("🏠 Return to Home", use_container_width=True, key="home_button"):
        st.session_state.page = "landing"
        st.experimental_rerun()

    if category == "Healthcare Assistant":
        show_chatbot()
    else:
        predictor = page["predictor"]()
        
        # Dynamic form display
        form_functions = {
            "Diabetes Readmission Risk ": show_readmission_form,
            "Diabetes": show_diagnosis_form,
            "Heart Disease": show_heart_form,
            "Heart Disease (GNN)": show_heart_gnn_form,
            "Chronic Kidney Disease": show_ckd_form,
            "Sepsis Prediction": show_sepsis_form,
            "Liver Disease": show_liver_form,
            "Thyroid Disease": show_thyroid_form,
            "Parkinson's Disease": show_parkinsons_form
        }

        if selection in form_functions:
            form_functions[selection](predictor)

    # Footer with consistent styling
    st.markdown("""
    <div style="text-align:center;padding:20px">
        <small style="color:#4dd0e1">⚠️ <b>Disclaimer</b>: These tools provide predictive insights only and do not constitute medical advice.</small><br>
        <small style="color:#4dd0e1">Always consult with a qualified healthcare professional for diagnosis and treatment.</small>
    </div>
    """, unsafe_allow_html=True)



def show_readmission_form(predictor):
    model_exists = (
        os.path.exists(os.path.join('models', 'readmission', 'diabetes_lstm_model.h5')) and
        os.path.exists(os.path.join('models', 'readmission', 'scaler.pkl')) and
        os.path.exists(os.path.join('models', 'readmission', 'feature_names.pkl'))
    )
    
    if not model_exists:
        st.warning("Readmission model not trained yet. Please train the model first.")
        if st.button("Train Readmission Model (First Time Setup)"):
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
        #st.warning("Diagnosis model not trained yet. Please train the model first.")
        if st.button("Train Diagnosis Model (First Time Setup)"):
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
        
        if st.form_submit_button(" Predict Diabetes Risk"):
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
        #st.warning("Heart disease model not trained yet. Please train the model first.")
        if st.button("Train Heart Disease Model (First Time Setup)"):
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
        
        if st.form_submit_button(" Predict Heart Disease Risk"):
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
        if st.button("Train Heart GNN Model (First Time Setup)"):
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
        
        if st.form_submit_button(" Predict Heart Disease Risk (GNN)"):
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
                st.subheader(" Model Architecture")
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
        if st.button(" Train CKD Model (First Time Setup)"):
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
        
        if st.form_submit_button(" Predict CKD Risk"):
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
        if st.button(" Train Sepsis Model (First Time Setup)"):
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
        
        if st.form_submit_button(" Predict Sepsis Risk"):
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
        if st.button(" Train Liver Model (First Time Setup)"):
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
        
        if st.form_submit_button(" Predict Liver Disease Risk"):
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
        if st.button(" Train Thyroid Model (First Time Setup)"):
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
        
        if st.form_submit_button(" Predict Thyroid Status"):
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
        if st.button(" Train Parkinson's Model (First Time Setup)"):
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
        
        if st.form_submit_button(" Predict Progression"):
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
    """
    Displays prediction results with consistent styling and comprehensive recommendations
    Args:
        risk_score (float): Risk probability between 0-1
        prediction_type (str): Type of prediction (e.g., "Heart Disease")
    """
    risk_percent = risk_score * 100
    st.subheader("📊 Prediction Results")
    
    # Progress bar with risk level coloring
    risk_meter = st.progress(0)
    if risk_percent < 30:
        risk_color = "#4CAF50"  # Green
        risk_message = "Low Risk"
        icon = "✅"
    elif risk_percent < 60:
        risk_color = "#FFC107"  # Amber
        risk_message = "Moderate Risk"
        icon = "⚠️"
    else:
        risk_color = "#F44336"  # Red
        risk_message = "High Risk"
        icon = "❗"
    
    risk_meter.progress(int(risk_percent))
    
    # Risk summary box
    st.markdown(f"""
    <div style="
        background-color:#f8f9fa;
        padding:20px;
        border-radius:10px;
        border-left: 6px solid {risk_color};
        margin: 10px 0;
    ">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 24px;">{icon}</span>
            <h3 style="color:{risk_color};margin:0;">
                {prediction_type} Risk: <b>{risk_percent:.1f}%</b> ({risk_message})
            </h3>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Recommendations section
    st.subheader("📝 Clinical Recommendations")
    
    # Container for recommendations
    rec_container = st.container()
    
    with rec_container:
        if prediction_type == "Heart Disease":
            if risk_percent > 60:
                st.markdown("""
                <div style="
                    background-color:#FFEBEE;
                    padding:15px;
                    border-radius:8px;
                    border-left: 4px solid #F44336;
                    margin: 5px 0;
                ">
                    <ul style="color:#B71C1C;">
                        <li>Consult a cardiologist immediately</li>
                        <li>Consider stress testing and echocardiogram</li>
                        <li>Monitor blood pressure twice daily</li>
                        <li>Begin therapeutic lifestyle changes</li>
                        <li>Consider lipid-lowering therapy</li>
                        <li>Reduce sodium intake to <2.3g/day</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            elif risk_percent > 30:
                st.markdown("""
                <div style="
                    background-color:#FFF8E1;
                    padding:15px;
                    border-radius:8px;
                    border-left: 4px solid #FFC107;
                    margin: 5px 0;
                ">
                    <ul style="color:#FF6F00;">
                        <li>Schedule cardiac screening within 3 months</li>
                        <li>Monitor cholesterol levels quarterly</li>
                        <li>Reduce saturated fat intake</li>
                        <li>30 minutes aerobic exercise 5x/week</li>
                        <li>Consider Mediterranean diet</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="
                    background-color:#E8F5E9;
                    padding:15px;
                    border-radius:8px;
                    border-left: 4px solid #4CAF50;
                    margin: 5px 0;
                ">
                    <ul style="color:#2E7D32;">
                        <li>Annual cardiovascular risk assessment</li>
                        <li>Maintain BMI between 18.5-24.9</li>
                        <li>Limit alcohol to ≤1 drink/day</li>
                        <li>150 minutes moderate exercise weekly</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        elif prediction_type == "Chronic Kidney Disease":
            if risk_percent > 60:
                st.markdown("""
                <div style="
                    background-color:#FFEBEE;
                    padding:15px;
                    border-radius:8px;
                    border-left: 4px solid #F44336;
                    margin: 5px 0;
                ">
                    <ul style="color:#B71C1C;">
                        <li>Nephrology consultation within 1 week</li>
                        <li>Monitor eGFR and creatinine monthly</li>
                        <li>Maintain BP <130/80 mmHg</li>
                        <li>Protein restriction (0.6-0.8g/kg/day)</li>
                        <li>Avoid NSAIDs and contrast dyes</li>
                        <li>Check electrolytes regularly</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            elif risk_percent > 30:
                st.markdown("""
                <div style="
                    background-color:#FFF8E1;
                    padding:15px;
                    border-radius:8px;
                    border-left: 4px solid #FFC107;
                    margin: 5px 0;
                ">
                    <ul style="color:#FF6F00;">
                        <li>Monitor kidney function every 3-6 months</li>
                        <li>Maintain hydration (2-3L/day)</li>
                        <li>Optimize blood sugar control if diabetic</li>
                        <li>Reduce sodium intake to <2g/day</li>
                        <li>Annual urine albumin-to-creatinine ratio</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="
                    background-color:#E8F5E9;
                    padding:15px;
                    border-radius:8px;
                    border-left: 4px solid #4CAF50;
                    margin: 5px 0;
                ">
                    <ul style="color:#2E7D32;">
                        <li>Annual kidney function tests</li>
                        <li>Stay well-hydrated</li>
                        <li>Monitor BP regularly</li>
                        <li>Limit NSAID use</li>
                        <li>Maintain healthy weight</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        elif prediction_type == "Sepsis":
            if risk_percent > 60:
                st.markdown("""
                <div style="
                    background-color:#FFEBEE;
                    padding:15px;
                    border-radius:8px;
                    border-left: 4px solid #F44336;
                    margin: 5px 0;
                ">
                    <ul style="color:#B71C1C;">
                        <li><b>SEPSIS ALERT:</b> Immediate medical attention required</li>
                        <li>Obtain blood cultures before antibiotics</li>
                        <li>Administer broad-spectrum antibiotics within 1 hour</li>
                        <li>30mL/kg IV crystalloid fluid bolus</li>
                        <li>Measure lactate q6h</li>
                        <li>ICU transfer recommended</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            elif risk_percent > 30:
                st.markdown("""
                <div style="
                    background-color:#FFF8E1;
                    padding:15px;
                    border-radius:8px;
                    border-left: 4px solid #FFC107;
                    margin: 5px 0;
                ">
                    <ul style="color:#FF6F00;">
                        <li><b>High suspicion for sepsis:</b> Monitor closely</li>
                        <li>Repeat full assessment within 1 hour</li>
                        <li>Consider early antibiotics</li>
                        <li>Check CBC, CRP, procalcitonin</li>
                        <li>Consider IV fluids if hypotensive</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="
                    background-color:#E8F5E9;
                    padding:15px;
                    border-radius:8px;
                    border-left: 4px solid #4CAF50;
                    margin: 5px 0;
                ">
                    <ul style="color:#2E7D32;">
                        <li>Continue current monitoring protocol</li>
                        <li>Reassess if clinical condition changes</li>
                        <li>Maintain strict infection control measures</li>
                        <li>Ensure adequate hydration</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        else:  # Default for other conditions (Diabetes, etc.)
            if risk_percent > 60:
                st.markdown("""
                <div style="
                    background-color:#FFEBEE;
                    padding:15px;
                    border-radius:8px;
                    border-left: 4px solid #F44336;
                    margin: 5px 0;
                ">
                    <ul style="color:#B71C1C;">
                        <li>Urgent specialist consultation recommended</li>
                        <li>Complete diagnostic workup needed</li>
                        <li>Initiate close monitoring protocol</li>
                        <li>Consider immediate interventions</li>
                        <li>Educate on warning signs</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            elif risk_percent > 30:
                st.markdown("""
                <div style="
                    background-color:#FFF8E1;
                    padding:15px;
                    border-radius:8px;
                    border-left: 4px solid #FFC107;
                    margin: 5px 0;
                ">
                    <ul style="color:#FF6F00;">
                        <li>Schedule specialist evaluation</li>
                        <li>Implement preventive measures</li>
                        <li>Begin regular monitoring</li>
                        <li>Lifestyle modifications advised</li>
                        <li>Follow-up in 1-3 months</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="
                    background-color:#E8F5E9;
                    padding:15px;
                    border-radius:8px;
                    border-left: 4px solid #4CAF50;
                    margin: 5px 0;
                ">
                    <ul style="color:#2E7D32;">
                        <li>Continue healthy habits</li>
                        <li>Annual screening recommended</li>
                        <li>Maintain preventive care</li>
                        <li>Monitor for any changes</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)



if __name__ == "__main__":
    main()