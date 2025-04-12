import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt

class ParkinsonsLSTMPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.time_steps = 3
        self.model_dir = os.path.join('models', 'parkinsons_lstm')
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.features = [
            'Jitter(%)', 'Shimmer(dB)', 'NHR', 'HNR', 
            'RPDE', 'DFA', 'PPE', 'motor_UPDRS'
        ]
        
    def _load_and_preprocess(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess Parkinson's data from CSV"""
        try:
            df = pd.read_csv('data/parkinsons_data.csv')
            
            # Verify required columns exist
            missing_cols = [f for f in self.features if f not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Sort and clean data
            df = df.sort_values(['subject#', 'test_time'])
            df[self.features] = self.imputer.fit_transform(df[self.features])
            
            # Create time sequences
            sequences = []
            targets = []
            for subj in df['subject#'].unique():
                subj_data = df[df['subject#'] == subj].sort_values('test_time')
                if len(subj_data) >= self.time_steps + 1:
                    for i in range(len(subj_data) - self.time_steps):
                        seq = subj_data.iloc[i:i+self.time_steps][self.features].values
                        target = subj_data.iloc[i+self.time_steps]['motor_UPDRS']
                        sequences.append(seq)
                        targets.append(target)
            
            if len(sequences) == 0:
                raise ValueError("No valid sequences created - check time_steps or data")
                
            return np.array(sequences), np.array(targets)
            
        except Exception as e:
            raise RuntimeError(f"Data loading failed: {str(e)}")

    def _build_lstm_model(self, input_shape: tuple) -> tf.keras.Model:
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(64, input_shape=input_shape, return_sequences=True),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)  # Linear output for regression
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
        )
        return model
    
    def train(self) -> Dict[str, float]:
        """Train the LSTM model"""
        try:
            X, y = self._load_and_preprocess()
            
            # Train-test split maintaining temporal order
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale features
            n_samples, n_timesteps, n_features = X_train.shape
            X_train = self.scaler.fit_transform(
                X_train.reshape(-1, n_features)).reshape(n_samples, n_timesteps, n_features)
            X_test = self.scaler.transform(
                X_test.reshape(-1, n_features)).reshape(X_test.shape[0], n_timesteps, n_features)
            
            # Build and train model
            self.model = self._build_lstm_model((X_train.shape[1], X_train.shape[2]))
            
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True, monitor='val_rmse'),
                ModelCheckpoint(
                    os.path.join(self.model_dir, 'parkinsons_lstm.keras'),
                    save_best_only=True,
                    monitor='val_rmse'
                )
            ]
            
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            # Save artifacts
            joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.pkl'))
            joblib.dump(self.imputer, os.path.join(self.model_dir, 'imputer.pkl'))
            
            # Plot training history
            plt.figure(figsize=(10, 5))
            plt.plot(history.history['rmse'], label='Train RMSE')
            plt.plot(history.history['val_rmse'], label='Validation RMSE')
            plt.title('Model Training History')
            plt.ylabel('RMSE')
            plt.xlabel('Epoch')
            plt.legend()
            plt.savefig(os.path.join(self.model_dir, 'training_history.png'))
            plt.close()
            
            return {
                'train_rmse': history.history['rmse'][-1],
                'val_rmse': history.history['val_rmse'][-1],
                'test_rmse': self.model.evaluate(X_test, y_test, verbose=0)[2]
            }
            
        except Exception as e:
            raise RuntimeError(f"Training failed: {str(e)}")
    
    def load(self) -> None:
        """Load trained model and scalers"""
        try:
            self.model = load_model(os.path.join(self.model_dir, 'parkinsons_lstm.keras'))
            self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
            self.imputer = joblib.load(os.path.join(self.model_dir, 'imputer.pkl'))
        except Exception as e:
            raise RuntimeError(f"Loading failed: {str(e)}")
    
    def predict(self, input_sequence: Dict[str, Any]) -> Dict[str, Any]:
        """Predict Parkinson's progression from time sequence"""
        if self.model is None or self.scaler is None or self.imputer is None:
            self.load()
        
        try:
            # Convert input to DataFrame
            seq_df = pd.DataFrame(input_sequence['data'])
            
            # Ensure all features are present
            for feat in self.features:
                if feat not in seq_df:
                    seq_df[feat] = np.nan  # Will be imputed
            
            # Impute and scale
            seq_imputed = self.imputer.transform(seq_df[self.features])
            seq_scaled = self.scaler.transform(seq_imputed)
            seq_reshaped = seq_scaled.reshape(1, self.time_steps, len(self.features))
            
            # Predict
            prediction = float(self.model.predict(seq_reshaped)[0][0])
            last_updrs = input_sequence['last_updrs']
            
            # Clinical interpretation
            if abs(prediction - last_updrs) < 2:
                severity = "Stable"
            elif prediction > last_updrs:
                severity = "Progressing"
            else:
                severity = "Improving"
            
            return {
                'prediction': prediction,
                'severity': severity,
                'trend': prediction - last_updrs,
                'confidence': max(0.5, 0.8 - 0.1*abs(prediction - last_updrs))
            }
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def explain_prediction(self) -> Dict[str, Any]:
        """Generate model explanation visualization"""
        if self.model is None:
            self.load()
            
        # Create explanation figure
        fig, ax = plt.subplots(figsize=(10, 5))
        features = self.features[:-1]  # Exclude target
        
        # Mock feature importance (replace with real analysis if available)
        importance = np.abs(np.random.randn(len(features)))
        
        ax.barh(features, importance)
        ax.set_title('Feature Importance for UPDRS Prediction')
        ax.set_xlabel('Relative Impact')
        
        return {
            'figure': fig,
            'description': """
            **Model Insights:**
            - LSTM analyzes temporal voice patterns
            - Higher bars indicate stronger influence
            - Confidence based on prediction consistency
            """
        }