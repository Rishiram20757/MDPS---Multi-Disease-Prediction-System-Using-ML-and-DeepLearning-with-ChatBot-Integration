import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import save_model
import joblib
import os
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class HeartDiseasePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.encoders = {}
        self.loaded = False
        self.expected_features = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        self.model_dir = os.path.join(BASE_DIR, '..', 'models', 'heart')
        os.makedirs(self.model_dir, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """Load and combine all UCI Heart Disease data files"""
        data_files = [
            'processed.cleveland.data',
            'processed.hungarian.data',
            'processed.switzerland.data',
            'processed.va.data'
        ]
        
        dfs = []
        for file in data_files:
            file_path = os.path.join(BASE_DIR, '..', 'data', 'heart', file)
            try:
                df = pd.read_csv(file_path, header=None, na_values='?')
                df.columns = self.expected_features + ['target']
                dfs.append(df)
            except FileNotFoundError:
                print(f"Warning: {file} not found, skipping")
                continue
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
                continue
        
        if not dfs:
            raise Exception("No heart disease data files found")
        
        combined = pd.concat(dfs, axis=0)
        combined['target'] = combined['target'].apply(lambda x: 0 if x == 0 else 1)
        combined['ca'] = combined['ca'].fillna(combined['ca'].median())
        combined['thal'] = combined['thal'].fillna(combined['thal'].median())
        return combined.dropna()

    def preprocess_data(self, df: pd.DataFrame) -> Tuple:
        """Prepare data for LSTM training"""
        X = df[self.expected_features]
        y = df['target']
        
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
        for col in categorical_cols:
            encoder = LabelEncoder()
            X[col] = encoder.fit_transform(X[col])
            self.encoders[col] = encoder
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        
        return X_train, X_test, y_train, y_test

    def build_model(self, input_shape: Tuple) -> Sequential:
        """Create LSTM model architecture"""
        model = Sequential([
            Input(shape=input_shape),
            LSTM(64, return_sequences=True, activation='tanh'),
            Dropout(0.3),
            BatchNormalization(),
            LSTM(32, activation='tanh'),
            Dropout(0.3),
            BatchNormalization(),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        return model

    def train(self) -> Dict:
        """Train the heart disease prediction model"""
        try:
            df = self.load_data()
            X_train, X_test, y_train, y_test = self.preprocess_data(df)
            
            self.model = self.build_model((1, X_train.shape[2]))
            
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            )
            
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=100,
                batch_size=32,
                callbacks=[early_stop],
                verbose=1
            )
            
            save_model(self.model, os.path.join(self.model_dir, 'heart_lstm_model.keras'))
            joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.pkl'))
            joblib.dump(self.encoders, os.path.join(self.model_dir, 'encoders.pkl'))
            
            loss, accuracy, auc = self.model.evaluate(X_test, y_test, verbose=0)
            return {
                'accuracy': accuracy,
                'auc': auc,
                'loss': loss,
                'epochs': len(history.history['loss'])
            }
        except Exception as e:
            raise Exception(f"Training failed: {str(e)}")

    def load(self) -> None:
        """Load trained model and preprocessing objects"""
        try:
            model_path = os.path.join(self.model_dir, 'heart_lstm_model.keras')
            scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
            encoders_path = os.path.join(self.model_dir, 'encoders.pkl')
            
            if not all(os.path.exists(p) for p in [model_path, scaler_path, encoders_path]):
                raise FileNotFoundError("Required model files not found")
                
            self.model = load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            self.encoders = joblib.load(encoders_path)
            self.loaded = True
        except Exception as e:
            raise Exception(f"Loading failed: {str(e)}")

    def predict(self, input_data: Dict) -> float:
        """Make heart disease prediction"""
        if not self.loaded:
            self.load()
        
        try:
            input_df = pd.DataFrame(columns=self.expected_features)
            for feature in self.expected_features:
                input_df[feature] = [input_data.get(feature, 0)]
            
            categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
            for col in categorical_cols:
                if col in input_df.columns:
                    if col in self.encoders:
                        unique_labels = set(self.encoders[col].classes_)
                        if input_df[col].iloc[0] not in unique_labels:
                            input_df[col] = self.encoders[col].transform([self.encoders[col].classes_[0]])[0]
                        else:
                            input_df[col] = self.encoders[col].transform(input_df[col])
            
            scaled_input = self.scaler.transform(input_df)
            scaled_input = scaled_input.reshape((1, 1, scaled_input.shape[1]))
            return float(self.model.predict(scaled_input, verbose=0)[0][0])
            
        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}")

    def explain_prediction(self, input_data: Dict = None, generic: bool = False):
        """Show simplified LSTM model diagram"""
        try:
            plt.figure(figsize=(10, 6))
            
            # Draw simplified LSTM architecture
            plt.title("How Our LSTM Model Works", pad=20, fontsize=16)
            
            # Model components
            components = [
                (0.1, 0.7, "Input Features\n(13 health metrics)", "lightblue"),
                (0.3, 0.7, "LSTM Layer\n(Learns patterns)", "lightgreen"),
                (0.5, 0.7, "LSTM Layer\n(Remembers important\nrelationships)", "lightgreen"),
                (0.7, 0.7, "Dense Layer\n(Combines features)", "gold"),
                (0.9, 0.7, "Risk Prediction\n(0-1 probability)", "salmon")
            ]
            
            for x, y, label, color in components:
                plt.gca().add_patch(plt.Rectangle((x-0.08, y-0.1), 0.16, 0.2, 
                                                facecolor=color, alpha=0.7, edgecolor='black'))
                plt.text(x, y, label, ha='center', va='center', fontsize=10)
                
                if x < 0.8:
                    plt.arrow(x+0.08, y, 0.14, 0, head_width=0.03, head_length=0.02, fc='k')
            
            # Add explanation text
            plt.text(0.5, 0.3, 
                    "LSTMs process data sequentially and learn\n"
                    "patterns in your health metrics over time,\n"
                    "making them good for medical predictions", 
                    ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))
            
            plt.axis('off')
            plt.tight_layout()
            
            return {
                'figure': plt.gcf(),
                'type': 'lstm',
                'description': "LSTM networks process data sequentially and learn temporal patterns"
            }
            
        except Exception as e:
            st.error(f"Visualization failed: {str(e)}")
            return None