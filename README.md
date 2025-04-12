# 🏥 MDPS-Multi-Disease-Prediction-System-Using-ML-and-DeepLearning-with-ChatBot-Integration

**An end-to-end AI clinical decision support system** with modular disease predictors and chatbot integration.

 *(Adding architecture diagram and demo later)*
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
## 📂 Project Structure
```bash
MDPS/
├── code/ # All prediction modules
│ ├── chatbot.py # Medical Q&A chatbot
│ ├── ckd.py # Chronic Kidney Disease predictor
│ ├── diabetes.py # Diabetes readmission LSTM
│ ├── diabetes_diagnosis.py # XGBoost classifier
│ ├── heart_disease.py # Heart LSTM model
│ ├── heart_gnn.py # Graph Neural Network version
│ ├── liver.py # Liver disease predictor
│ ├── main.py # Streamlit dashboard
│ ├── parkinsons_lstm.py # Time-series analysis
│ ├── sepsis.py # ICU prediction model
│ └── thyroid.py # Thyroid disorder classifier
│
├── data/ #  medical datasets
│
├── models/ # Pretrained models
│ ├── chronic/ # CKD model files
│ ├── diagnosis/ # Diabetes diagnosis
│ ├── heart/ # LSTM weights
│ ├── heart_gnn/ # GNN artifacts
│ ├── liver/ # XGBoost model
│ ├── parkinsons_lstm/ # Time-series model
│ ├── readmission/ # Diabetes LSTM
│ ├── sepsis/ # ICU model
│ └── thyroid/ # Random Forest
│
└── requirements.txt # Dependencies
```
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


## 🚀 How To Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/Rishiram20757/MDPS---Multi-Disease-Prediction-System-Using-ML-and-DeepLearning-with-ChatBot-Integration.git
   cd MDPS
2. **Setup environment**
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   # (Optional) Create virtual environment
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
 3. **Ollama Setup (For Medical Chatbot)**
    ```bash
    # Download Ollama
    curl -fsSL https://ollama.com/install.sh | sh
    # Pull the medical LLM (3.8GB)
    ollama pull monotykamary/medichat-llama3
    # Verify installation
    ollama list
 4. **Launch DashBoard**
    ```bash
    streamlit run code/main.py

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
##  Module Overview

| Module | File | Model Type | Key Features | Dataset | Input Requirements |
|--------|------|------------|--------------|---------|-------------------|
| **Heart Disease (LSTM)** | `heart_disease.py` | LSTM | Temporal pattern analysis | UCI Heart | 13 clinical features |
| **Heart Disease (GNN)** | `heart_gnn.py` | Graph Neural Network | Feature relationship mapping | UCI Heart | Same as LSTM |
| **Diabetes Diagnosis** | `diabetes_diagnosis.py` | XGBoost | PIMA Indians analysis | Kaggle | 8 biomarkers |
| **Diabetes Readmission** | `diabetes.py` | LSTM | 30-day risk prediction | Hospital EHR | 50+ EHR features |
| **Chronic Kidney Disease** | `ckd.py` | Deep Neural Network | 25+ clinical params | Indian CKD | Mixed numerical/categorical |
| **Parkinson's Progression** | `parkinsons_lstm.py` | Time-series LSTM | Voice tremor patterns | NIH Dataset | 8 voice features × 3 timesteps |
| **Sepsis Prediction** | `sepsis.py` | ICU LSTM | Real-time ICU monitoring | MIMIC-III | 38 time-series features |
| **Liver Disease** | `liver.py` | XGBoost | Liver function tests | Indian Liver | 10 blood markers |
| **Thyroid Disorders** | `thyroid.py` | Random Forest | Hormonal imbalance | UCI Thyroid | 5 test results |
| **Medical Chatbot** | `chatbot.py` | Llama3-8B | Symptom analysis | - | Free-text input |

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 💡 Key Novelties

1. **Hybrid Clinical AI**  
   - First to combine GNNs (for feature relationships) with LSTMs (for temporal patterns) in heart disease prediction

2. **Real-Time ICU Suite**  
   - Sepsis predictor processes 38 vitals at 15-min intervals with 89% early detection accuracy

3. **Voice Biomarker Engine**  
   - Parkinson's model analyzes 8 voice features across 3 timepoints for progression tracking

4. **Privacy-Preserving Chat**  
   - Local Ollama deployment ensures no PHI leaves the hospital network

5. **Unified Clinical Interface**  
   - Single dashboard integrates 10+ specialized models with EMR export capabilities

6. **Explainability Built-In**  
   - Every prediction includes:
     - SHAP values for ML models
     - Attention maps for LSTMs
     - Node importance for GNNs
