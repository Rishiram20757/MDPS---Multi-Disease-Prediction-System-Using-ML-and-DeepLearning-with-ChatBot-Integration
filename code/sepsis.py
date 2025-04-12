import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import os
from typing import Dict

class SepsisPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.loaded = False
        self.model_dir = os.path.join('models', 'sepsis')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Define all expected features (38 total)
        self.time_series_features = [
            'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 
            'EtCO2', 'BaseExcess', 'HCO3', 'FiO2', 'pH', 
            'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 
            'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
            'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
            'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
            'Fibrinogen', 'Platelets'
        ]
        self.static_features = ['Age', 'Gender', 'HospAdmTime', 'ICULOS']
        self.all_features = self.time_series_features + self.static_features
        self.target_col = 'SepsisLabel'
        self.max_timesteps = 24

    def load_and_preprocess_data(self) -> pd.DataFrame:
        """Load and preprocess sepsis data"""
        df = pd.read_csv('data/sepsis.csv')
        
        # Handle missing values
        for col in self.time_series_features:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # Convert gender to binary
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
        
        return df

    def create_sequences(self, df: pd.DataFrame) -> tuple:
        """Create fixed-length sequences with exactly 38 features"""
        grouped = df.groupby('Patient_ID')
        sequences = []
        labels = []
        
        for _, group in grouped:
            group = group.sort_values('ICULOS').tail(self.max_timesteps)
            
            # Ensure we have exactly 38 features in the correct order
            features = group[self.all_features]
            
            # Pad if needed
            if len(group) < self.max_timesteps:
                padding = pd.DataFrame([group.iloc[-1][self.all_features]] * 
                                     (self.max_timesteps - len(group)))
                features = pd.concat([features, padding])
            
            sequences.append(features.values)
            labels.append(group[self.target_col].iloc[-1])
            
        return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.float32)

    def prepare_data(self) -> tuple:
        """Prepare data ensuring consistent feature dimensions"""
        df = self.load_and_preprocess_data()
        X, y = self.create_sequences(df)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features - reshape to (n_samples * timesteps, n_features)
        self.scaler = StandardScaler()
        X_train_reshaped = X_train.reshape(-1, len(self.all_features))
        X_test_reshaped = X_test.reshape(-1, len(self.all_features))
        
        self.scaler.fit(X_train_reshaped)
        X_train_scaled = self.scaler.transform(X_train_reshaped).reshape(X_train.shape)
        X_test_scaled = self.scaler.transform(X_test_reshaped).reshape(X_test.shape)
        
        # Save scaler and feature order
        joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.pkl'))
        joblib.dump(self.all_features, os.path.join(self.model_dir, 'feature_order.pkl'))
        
        return X_train_scaled, X_test_scaled, y_train, y_test

    def build_lstm_model(self, input_shape: tuple) -> Sequential:
        """Build model expecting 38 features per timestep"""
        model = Sequential([
            Masking(mask_value=0., input_shape=input_shape),
            LSTM(64, return_sequences=True),
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
        """Train model with consistent feature dimensions"""
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.model = self.build_lstm_model((self.max_timesteps, len(self.all_features)))
        
        model_path = os.path.join(self.model_dir, 'sepsis_lstm_model.keras')
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True, monitor='val_auc', mode='max'),
            ModelCheckpoint(model_path, save_best_only=True, monitor='val_auc', mode='max')
        ]

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        return history.history

    def load(self) -> None:
        """Load model and ensure feature dimensions match"""
        model_path = os.path.join(self.model_dir, 'sepsis_lstm_model.keras')
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        features_path = os.path.join(self.model_dir, 'feature_order.pkl')
        
        if not all(os.path.exists(p) for p in [model_path, scaler_path, features_path]):
            raise FileNotFoundError("Model files not found. Train the model first.")

        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.all_features = joblib.load(features_path)
        self.loaded = True

    def predict(self, patient_data: Dict) -> float:
        """Make prediction ensuring 38 features"""
        if not self.loaded:
            self.load()
        
        # Initialize sequence with zeros (1 sample, 24 timesteps, 38 features)
        sequence = np.zeros((1, self.max_timesteps, len(self.all_features)), dtype=np.float32)
        
        # Fill in static features for all timesteps
        for j, feature in enumerate(self.all_features):
            if feature in self.static_features and feature in patient_data:
                sequence[0, :, j] = patient_data[feature]
        
        # Fill in time-series features
        for feature in self.time_series_features:
            if feature in patient_data:
                values = patient_data[feature]
                if not isinstance(values, list):
                    values = [values]  # Convert single value to list
                
                # Fill available time points
                num_points = min(len(values), self.max_timesteps)
                j = self.all_features.index(feature)
                sequence[0, :num_points, j] = values[:num_points]
        
        # Scale features
        sequence_reshaped = sequence.reshape(-1, len(self.all_features))
        sequence_scaled = self.scaler.transform(sequence_reshaped)
        sequence_scaled = sequence_scaled.reshape(sequence.shape)
        
        # Predict
        with tf.device('/cpu:0'):
            return float(self.model.predict(sequence_scaled, verbose=0)[0][0])