import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
import os
from typing import Dict

class LiverPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.expected_features = [
            'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
            'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
            'Aspartate_Aminotransferase', 'Total_Protiens',
            'Albumin', 'Albumin_and_Globulin_Ratio'
        ]
        self.model_dir = os.path.join('models', 'liver')
        os.makedirs(self.model_dir, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """Load and preprocess liver dataset"""
        df = pd.read_csv('data/liver.csv')
        
        # Handle missing values and convert gender
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
        df = df.dropna()
        
        # Ensure all expected columns exist
        for col in self.expected_features:
            if col not in df.columns:
                if col == 'Albumin_and_Globulin_Ratio':
                    # Calculate if missing
                    df['Albumin_and_Globulin_Ratio'] = df['Albumin'] / (df['Total_Protiens'] - df['Albumin'])
                elif col not in df.columns:
                    df[col] = 0  # Default value for missing columns
        
        return df

    def train(self) -> Dict:
        """Train and save the model"""
        df = self.load_data()
        X = df[self.expected_features]
        y = df['Dataset'] - 1  # Convert to binary (0=healthy, 1=disease)
        
        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Train XGBoost model
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        self.model.fit(X_train, y_train)
        
        # Save artifacts
        joblib.dump(self.model, os.path.join(self.model_dir, 'liver_model.pkl'))
        joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.pkl'))
        joblib.dump(self.expected_features, os.path.join(self.model_dir, 'feature_names.pkl'))
        
        return {"status": "Model trained successfully"}

    def load(self) -> None:
        """Load trained model"""
        model_path = os.path.join(self.model_dir, 'liver_model.pkl')
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        features_path = os.path.join(self.model_dir, 'feature_names.pkl')
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.expected_features = joblib.load(features_path)

    def predict(self, input_data: Dict) -> float:
        """Make prediction"""
        if not self.model:
            self.load()
            
        # Create input DataFrame with all expected features
        input_df = pd.DataFrame(columns=self.expected_features)
        
        # Fill with provided values, default to 0 for missing
        for feature in self.expected_features:
            input_df[feature] = [input_data.get(feature, 0)]
        
        # Calculate derived features if needed
        if 'Albumin_and_Globulin_Ratio' in self.expected_features:
            if 'Albumin' in input_data and 'Total_Protiens' in input_data:
                alb = input_data['Albumin']
                total_prot = input_data['Total_Protiens']
                if total_prot > alb:  # Prevent division by zero
                    input_df['Albumin_and_Globulin_Ratio'] = alb / (total_prot - alb)
        
        # Scale features
        scaled_input = self.scaler.transform(input_df[self.expected_features])
        
        return float(self.model.predict_proba(scaled_input)[0][1])