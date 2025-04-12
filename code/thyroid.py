import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from typing import Dict
import streamlit as st

class ThyroidPredictor:
    def __init__(self):
        """Initialize with correct settings for comma-separated data"""
        self.model = None
        self.scaler = None
        self.model_dir = os.path.join('models', 'thyroid')
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.expected_features = [
            'T3_resin_uptake',        # Percentage
            'Total_thyroxin',         # μg/dL
            'Total_triiodothyronine',  # ng/dL
            'Basal_TSH',              # μIU/mL
            'Max_TSH_diff'            # After TRH
        ]
        
        self.class_map = {
            '1': 'normal', '1.0': 'normal',
            '2': 'hyper', '2.0': 'hyper',
            '3': 'hypo', '3.0': 'hypo'
        }

    def load_data(self) -> pd.DataFrame:
        """Load comma-separated data with proper validation"""
        try:
            # Try to find the data file
            data_path = self._find_data_file()
            
            # Read CSV with comma separator
            df = pd.read_csv(
                data_path,
                header=None,
                sep=',',  # Changed to comma separator
                names=['diagnosis'] + self.expected_features,
                dtype={'diagnosis': str}  # Force diagnosis to string
            )
            
            # Clean and validate
            df = self._clean_data(df)
            return df
            
        except Exception as e:
            st.error(f"❌ Data loading failed: {str(e)}")
            raise

    def _find_data_file(self) -> str:
        """Locate the data file"""
        possible_paths = [
            'data/thyroid/new-thyroid.data',
            'new-thyroid.data',
            'thyroid/new-thyroid.data',
            os.path.join(os.path.dirname(__file__), 'new-thyroid.data')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
                
        raise FileNotFoundError(f"Could not find thyroid data file in: {possible_paths}")

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the data"""
        # Convert diagnosis
        df['diagnosis'] = (
            df['diagnosis']
            .astype(str)
            .str.strip()
            .map(self.class_map)
        )
        
        # Check for invalid diagnoses
        if df['diagnosis'].isna().any():
            invalid = df[df['diagnosis'].isna()]['diagnosis'].unique()
            raise ValueError(
                f"Invalid diagnosis values found: {invalid}. "
                f"Should be 1 (normal), 2 (hyper), or 3 (hypo)"
            )
        
        # Check features
        for col in self.expected_features:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Non-numeric values in {col}")
            if df[col].isna().any():
                raise ValueError(f"Missing values in {col}")
        
        return df

    def train(self) -> Dict:
        """Train the model"""
        try:
            df = self.load_data()
            
            st.info(f"Data loaded successfully. Shape: {df.shape}")
            st.info(f"Class distribution:\n{df['diagnosis'].value_counts()}")
            
            X = df[self.expected_features]
            y = df['diagnosis']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
            
            self._save_model()
            
            return {
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "accuracy": self.model.score(X_test, y_test),
                "class_distribution": df['diagnosis'].value_counts().to_dict()
            }
            
        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            raise

    def _save_model(self) -> None:
        """Save model artifacts"""
        try:
            joblib.dump(self.model, os.path.join(self.model_dir, 'thyroid_model.pkl'))
            joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.pkl'))
            joblib.dump(self.expected_features, os.path.join(self.model_dir, 'feature_names.pkl'))
            st.success("✅ Model saved successfully")
        except Exception as e:
            st.error(f"❌ Failed to save model: {str(e)}")
            raise

    def load(self) -> None:
        """Load trained model"""
        try:
            self.model = joblib.load(os.path.join(self.model_dir, 'thyroid_model.pkl'))
            self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
            self.expected_features = joblib.load(os.path.join(self.model_dir, 'feature_names.pkl'))
        except Exception as e:
            st.error(f"❌ Failed to load model: {str(e)}")
            raise

    def predict(self, input_data: Dict) -> Dict:
        """Make prediction with input validation"""
        try:
            if not self.model:
                self.load()
            
            # Validate input
            for feature in self.expected_features:
                if feature not in input_data:
                    raise ValueError(f"Missing feature: {feature}")
                if not isinstance(input_data[feature], (int, float)):
                    raise ValueError(f"{feature} must be numeric")
            
            # Create input array
            input_df = pd.DataFrame([input_data])
            scaled_input = self.scaler.transform(input_df[self.expected_features])
            
            # Get probabilities
            proba = self.model.predict_proba(scaled_input)[0]
            return dict(zip(self.model.classes_, proba))
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            raise