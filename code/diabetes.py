import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import os
from typing import Tuple, Dict, Optional

# Configure paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'readmission')  # Updated to subfolder
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(MODEL_DIR, exist_ok=True)  # Creates the readmission folder if needed

class DiabetesPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.expected_features = None
        self.loaded = False
        
    def parse_mapping_file(self, file_path: str) -> Dict[str, Dict[int, str]]:
        """Parse the mapping CSV file with irregular format"""
        with open(file_path, 'r') as f:
            content = f.read()
        
        sections = [s.strip() for s in content.split(',\n') if s.strip()]
        mappings = {}
        current_section = None
        
        for section in sections:
            lines = section.split('\n')
            header = lines[0].strip()
            
            if 'admission_type_id' in header:
                current_section = 'admission_type'
                mappings[current_section] = {}
            elif 'discharge_disposition_id' in header:
                current_section = 'discharge_disposition'
                mappings[current_section] = {}
            elif 'admission_source_id' in header:
                current_section = 'admission_source'
                mappings[current_section] = {}
            elif current_section:
                for line in lines:
                    if line.strip() and ',' in line:
                        try:
                            id_part, desc = line.split(',', 1)
                            id_val = int(id_part.strip())
                            mappings[current_section][id_val] = desc.strip()
                        except (ValueError, IndexError):
                            continue
        
        return mappings

    def load_and_preprocess_data(self) -> pd.DataFrame:
        """Load and preprocess diabetes data"""
        diabetes_df = pd.read_csv(
            os.path.join(DATA_DIR, 'diabetic_data.csv'),
            na_values=['?', 'NA', 'NULL', '', ' ', 'Unknown', 'None'],
            low_memory=False
        )
        
        mappings = self.parse_mapping_file(os.path.join(DATA_DIR, 'IDS_mapping.csv'))
        
        for col, mapping in [
            ('admission_type_id', 'admission_type'),
            ('discharge_disposition_id', 'discharge_disposition'),
            ('admission_source_id', 'admission_source')
        ]:
            if mapping in mappings:
                diabetes_df[mapping] = diabetes_df[col].map(mappings[mapping])

        # Process medications
        medication_columns = [
            'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
            'glimepiride', 'glipizide', 'glyburide', 'tolbutamide',
            'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',
            'troglitazone', 'tolazamide', 'examide', 'citoglipton',
            'insulin', 'glyburide-metformin', 'glipizide-metformin',
            'glimepiride-pioglitazone', 'metformin-rosiglitazone',
            'metformin-pioglitazone'
        ]

        for col in medication_columns:
            diabetes_df[col] = (
                diabetes_df[col]
                .astype(str)
                .str.strip()
                .str.upper()
                .replace(['NO', 'NONE', ''], '0')
                .replace({'STEADY': '1', 'UP': '1', 'DOWN': '1'})
                .apply(lambda x: 1 if x not in ['0', ''] else 0)
            )

        # Process diagnosis codes
        def process_diagnosis_code(code):
            if pd.isna(code) or str(code).strip().upper() in ['?', 'UNKNOWN', '']:
                return 0.0
            
            code = str(code).strip().upper()
            
            if code.startswith('E'):
                try:
                    return 1000 + float(code[1:4])
                except:
                    return 0.0
            elif code.startswith('V'):
                try:
                    return 2000 + float(code[1:3])
                except:
                    return 0.0
            try:
                return float(code)
            except:
                return 0.0

        for diag_col in ['diag_1', 'diag_2', 'diag_3']:
            diabetes_df[diag_col] = diabetes_df[diag_col].apply(process_diagnosis_code)
            diabetes_df[f'{diag_col}_diabetes'] = diabetes_df[diag_col].apply(
                lambda x: 1 if 250 <= x < 251 else 0
            )

        # Process weight
        def convert_weight(weight):
            try:
                if isinstance(weight, str):
                    weight = weight.strip()
                    if weight.startswith('[') and '-' in weight and weight.endswith(')'):
                        nums = weight[1:-1].split('-')
                        if len(nums) == 2:
                            try:
                                return (float(nums[0]) + float(nums[1])) / 2
                            except:
                                pass
                    weight = weight.replace(')', '').strip()
                    return float(weight)
                return float(weight)
            except:
                return 12.5

        diabetes_df['weight'] = diabetes_df['weight'].apply(convert_weight)

        # Process categorical variables
        categorical_cols = [
            'race', 'gender', 'age', 'max_glu_serum', 'A1Cresult',
            'change', 'diabetesMed', 'admission_type',
            'discharge_disposition', 'admission_source'
        ]

        for col in categorical_cols:
            diabetes_df[col] = (
                diabetes_df[col]
                .astype(str)
                .str.strip()
                .str.upper()
                .replace({'?': 'UNKNOWN', '': 'UNKNOWN'})
                .fillna('UNKNOWN')
            )
            
            diabetes_df = pd.get_dummies(
                diabetes_df,
                columns=[col],
                prefix=col,
                drop_first=True,
                dtype=np.float32
            )

        # Process target
        diabetes_df['readmitted'] = (
            diabetes_df['readmitted']
            .astype(str)
            .str.strip()
            .str.upper()
            .map({'NO': 0, '>30': 1, '<30': 1})
            .fillna(0)
        )

        # Cleanup
        drop_cols = [
            'admission_type_id', 'discharge_disposition_id',
            'admission_source_id', 'payer_code', 'medical_specialty'
        ]
        diabetes_df.drop(columns=drop_cols, inplace=True, errors='ignore')

        for col in diabetes_df.columns:
            diabetes_df[col] = pd.to_numeric(diabetes_df[col], errors='coerce').fillna(0)
        
        return diabetes_df.astype(np.float32)

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training"""
        X = df.drop('readmitted', axis=1)
        self.expected_features = X.columns.tolist()
        y = df['readmitted']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        numerical_cols = [
            'time_in_hospital', 'num_lab_procedures', 'num_procedures',
            'num_medications', 'number_outpatient', 'number_emergency',
            'number_inpatient', 'number_diagnoses', 'weight'
        ]

        self.scaler = StandardScaler()
        X_train[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = self.scaler.transform(X_test[numerical_cols])

        with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)

        X_train = np.asarray(X_train, dtype=np.float32)
        X_test = np.asarray(X_test, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        y_test = np.asarray(y_test, dtype=np.float32)

        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

        return X_train, X_test, y_train, y_test

    def build_lstm_model(self, input_shape: tuple) -> Sequential:
        """Build and compile LSTM model"""
        model = Sequential([
            LSTM(64, input_shape=input_shape, return_sequences=True),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(32),
            BatchNormalization(),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )

        return model

    def train(self) -> dict:
        """Train and save the model"""
        diabetes_df = self.load_and_preprocess_data()
        X_train, X_test, y_train, y_test = self.prepare_data(diabetes_df)

        self.model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
        
        model_path = os.path.join(MODEL_DIR, 'diabetes_lstm_model.keras')
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ModelCheckpoint(
                filepath=model_path,
                save_best_only=True,
                monitor='val_loss',
                mode='min'
            )
        ]

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=64,
            callbacks=callbacks,
            verbose=1
        )

        with open(os.path.join(MODEL_DIR, 'feature_names.pkl'), 'wb') as f:
            pickle.dump(self.expected_features, f)

        return history.history

    def load(self) -> None:
        """Load trained model and scaler"""
        model_path = os.path.join(MODEL_DIR, 'diabetes_lstm_model.keras')
        scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
        features_path = os.path.join(MODEL_DIR, 'feature_names.pkl')

        if not all(os.path.exists(p) for p in [model_path, scaler_path, features_path]):
            raise FileNotFoundError("Required model files not found. Train the model first.")

        self.model = load_model(model_path, compile=False)
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
            
        with open(features_path, 'rb') as f:
            self.expected_features = pickle.load(f)
            
        self.loaded = True

    def predict(self, input_data: dict) -> float:
        """Make prediction with input validation"""
        if not self.loaded:
            self.load()
        
        # Create DataFrame with all expected features
        input_df = pd.DataFrame(columns=self.expected_features)
        input_df.loc[0] = 0  # Initialize with zeros
        
        # Update with provided values
        for key, value in input_data.items():
            if key in input_df.columns:
                input_df[key] = value
        
        # Scale numerical features
        numerical_cols = [
            'time_in_hospital', 'num_lab_procedures', 'num_procedures',
            'num_medications', 'number_outpatient', 'number_emergency',
            'number_inpatient', 'number_diagnoses', 'weight'
        ]
        
        input_df[numerical_cols] = self.scaler.transform(input_df[numerical_cols])

        # Convert to numpy array and reshape for LSTM
        input_array = np.asarray(input_df, dtype=np.float32).reshape(1, 1, -1)

        return float(self.model.predict(input_array)[0][0])


if __name__ == "__main__":
    predictor = DiabetesPredictor()
    
    # To train (run once):
    # history = predictor.train()
    
    # To load and predict:
    predictor.load()
    
    sample_input = {
        'time_in_hospital': 5,
        'num_lab_procedures': 30,
        'num_procedures': 2,
        'num_medications': 10,
        'number_outpatient': 0,
        'number_emergency': 1,
        'number_inpatient': 0,
        'number_diagnoses': 5,
        'weight': 75,
        'metformin': 1,
        'insulin': 0
    }
    
    # Fill missing features with defaults
    for feature in predictor.expected_features:
        if feature not in sample_input:
            if 'diabetes' in feature:
                sample_input[feature] = 0
            elif any(x in feature for x in ['race', 'gender', 'age']):
                sample_input[feature] = 0  # Assuming unknown category
            else:
                sample_input[feature] = 0
    
    prediction = predictor.predict(sample_input)
    print(f"Predicted probability of readmission: {prediction:.4f}")