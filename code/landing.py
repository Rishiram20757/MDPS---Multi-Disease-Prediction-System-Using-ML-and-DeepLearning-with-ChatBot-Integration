# landing.py
import streamlit as st

def show_landing():
    # Custom CSS with enhanced styling
    st.markdown("""
    <style>
    /* Main background - light cyan gradient */
    .stApp {
        background: linear-gradient(135deg, #e0f7fa 0%, #e1f5fe 50%, #e3f2fd 100%);
        color: #01579b;
    }
    
    /* Enhanced Button Styling */
    .stButton>button {
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        border-radius: 12px;
        background: linear-gradient(135deg, #00bcd4 0%, #00838f 100%);
        color: white !important;
        border: none;
        padding: 16px 36px;
        font-weight: 600;
        letter-spacing: 1px;
        box-shadow: 0 4px 8px rgba(0, 188, 212, 0.2);
        font-size: 1.1rem;
        text-transform: uppercase;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(0, 188, 212, 0.3);
        background: linear-gradient(135deg, #00838f 0%, #006064 100%);
    }
    
    /* Card styling */
    .metric-card {
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        padding: 20px;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.9);
        box-shadow: 0 4px 12px rgba(1, 87, 155, 0.1);
        margin-bottom: 20px;
        border-left: 5px solid #00bcd4;
        backdrop-filter: blur(5px);
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(1, 87, 155, 0.2);
    }
    
    /* Progress bars */
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
    }
    
    /* Text styling */
    .metric-title {
        color: #01579b !important;
        margin: 0;
        font-size: 1.1rem;
    }
    .metric-description {
        color: #0288d1 !important;
        font-size: 14px;
        margin-top: 0;
        min-height: 40px;
    }
    .metric-label {
        color: #0288d1 !important;
        font-size: 13px;
        margin-bottom: 5px;
    }
    .metric-value {
        color: #01579b !important;
        font-size: 13px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .metric-card {
        animation: fadeInUp 0.6s ease forwards;
    }
    </style>
    """, unsafe_allow_html=True)

    # Hero Section
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🏥 Multi-Disease Prediction System")
        st.markdown("""
        <div style="
            padding: 20px;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.9);
            box-shadow: 0 4px 12px rgba(1, 87, 155, 0.1);
            margin-bottom: 20px;
            border-left: 5px solid #00bcd4;
        ">
            <p style="font-size:18px; color: #0277bd;">
                MDPS is a cutting-edge AI-powered clinical decision support platform that integrates machine learning and deep learning (LSTM/RNN, GNN & DNN ) models to predict multiple chronic diseases. It features an intelligent medical chatbot powered by Ollama’s Medichat-LLaMA3, enabling real-time health queries and guidance. With modular prediction tabs, users can assess risk for heart, diabetes, thyroid, liver, kidney diseases and more. Designed for innovation, accessibility, and scalability in modern healthcare.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("GET STARTED", key="get_started"):
            st.session_state.page = "main"
            st.experimental_rerun()
    
    with col2:
        st.image("./logo.png", width=400)

    st.markdown("---")
    
    # Key Features Section
    st.header("✨ Key Features")
    features = st.columns(6)
    
    features_data = [
        {"icon": "🤖", "title": "Multi-Model Integration", "desc": "Uses a mix of ML & DL  architectures tailored to each disease for  contextual prediction."},
        {"icon": "⚡", "title": "Real-Time Predictions", "desc": "Processes both static health metrics and dynamic time-series ICU data with <500ms prediction latency using optimized neural architectures"},
        {"icon": "📊", "title": "Clinical Insights", "desc": "Actionable recommendations"},
        {"icon": "📚", "title": "Disease-Specific Modules", "desc": "Separate tabs for Diabetes, Heart, Thyroid, Liver, Kidney, Parkinson's, Sepsis and Readmission Risk prediction—each with optimized models."},
        {"icon": "📦", "title": "Lightweight and Offline", "desc" : "Fully local deployment—no cloud dependency—ideal for quick demos and health camps."},
        {"icon": "🧠", "title": "Diagramatical Explaination", "desc": "Visual model explanations show feature importance and decision pathways for clinician trust (LSTM attention maps, GNN node relationships)"}
    ]
    
    for i, col in enumerate(features):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="text-align:center;">
                <div style="font-size:42px; margin-bottom:10px;">{features_data[i]['icon']}</div>
                <h3 style="color:#01579b; margin-top:0;">{features_data[i]['title']}</h3>
                <p style="color:#0288d1;">{features_data[i]['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Model Metrics Section - Fixed Version
    st.header("📊 Model Performance Metrics")
    
    metrics_data = [
        {"icon": "🫀", "title": "Heart Disease ", "accuracy": 0.81, "auc": 0.79, "desc": "Graph Neural Network analyzing feature relationships"},
        {"icon": "💉", "title": "Diabetes", "accuracy": 0.92, "auc": 0.88, "desc": "Using XGBoost"},
        {"icon": "🦠", "title": "Sepsis Prediction", "accuracy": 0.90, "auc": 0.93, "desc": "LSTM processing ICU vitals"},
        {"icon": "🧠", "title": "Parkinson's ", "accuracy": 0.80, "auc": 0.82, "desc": "LSTM Predicts motor UPDRS scores"},
        {"icon": "", "title": "Liver Disease", "accuracy": 0.94, "auc": 0.96, "desc": "XGBoost classifier"},
        {"icon": "🫀", "title": "Heart Disease ", "accuracy": 0.83, "auc": 0.86, "desc": "Using LSTM"},
        {"icon": "", "title": "Thyroid ", "accuracy": 0.96, "auc": 0.93, "desc": "Random Forest classifier"},
        {"icon": "", "title": "Chronic Kidney Disease ", "accuracy": 0.89, "auc": 0.91, "desc": "Using Deep Neural Network"},
        {"icon": "💉", "title": "Diabetes Readmission Risk ", "accuracy": 0.72, "auc": 0.74, "desc": "LSTM model predicting readmission risk"}
    ]
    
    # Create rows of 3 cards each
    for i in range(0, len(metrics_data), 3):
        cols = st.columns(3)
        row_data = metrics_data[i:i+3]
        
        for j, col in enumerate(cols):
            if j < len(row_data):
                metric = row_data[j]
                with col:
                    # Build the card using Streamlit components
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="display:flex; align-items:center; margin-bottom:12px;">
                            <span style="font-size:32px; margin-right:12px;">{metric['icon']}</span>
                            <h3 class="metric-title">{metric['title']}</h3>
                        </div>
                        <p class="metric-description">{metric['desc']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Accuracy
                    col1, col2 = st.columns([3,1])
                    with col1:
                        st.markdown('<p class="metric-label">Accuracy</p>', unsafe_allow_html=True)
                    with col2:
                        st.markdown(f'<p class="metric-value" style="text-align:right;">{metric["accuracy"]*100:.0f}%</p>', unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {metric['accuracy']*100}%"></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # AUC
                    col1, col2 = st.columns([3,1])
                    with col1:
                        st.markdown('<p class="metric-label" style="margin-top:15px;">AUC Score</p>', unsafe_allow_html=True)
                    with col2:
                        st.markdown(f'<p class="metric-value" style="text-align:right;">{metric["auc"]*100:.0f}%</p>', unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {metric['auc']*100}%"></div>
                    </div>
                    """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding:20px;">
        <small style="color: #0288d1;">
            ⚠️ Predictive insights only - Not medical advice 
        </small>
    </div>
    """, unsafe_allow_html=True)