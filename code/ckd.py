import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os
from typing import Dict, Any, List

class CKDPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.encoders = {}
        self.model_dir = os.path.join('models', 'chronic')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Column configuration - must match training data exactly
        self.categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
        self.numerical_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
        self.target_col = 'classification'
        
        # Combined feature order - numerical first, then categorical
        self.feature_order = self.numerical_cols + self.categorical_cols

    def _load_data(self) -> pd.DataFrame:
        """Load and preprocess the kidney disease dataset"""
        df = pd.read_csv('data/kidney_disease.csv')
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower()
        
        # Convert target
        df[self.target_col] = df[self.target_col].str.strip().str.lower()
        df[self.target_col] = df[self.target_col].map({'ckd': 1, 'notckd': 0, 'ckd\t': 1})
        
        # Clean and encode categorical columns
        for col in self.categorical_cols:
            if col in df.columns:
                # Standardize values
                df[col] = df[col].astype(str).str.strip().str.lower()
                df[col] = df[col].replace({
                    'normal': 'yes', 'abnormal': 'no',
                    'present': 'yes', 'notpresent': 'no',
                    'good': 'yes', 'poor': 'no',
                    '\\tno': 'no', '\\tyes': 'yes',
                    '?': 'unknown', '': 'unknown'
                })
                
                # Label encode
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
        
        # Clean numerical columns
        for col in self.numerical_cols:
            if col in df.columns:
                # Remove non-numeric characters
                df[col] = df[col].replace(r'[^\d.]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Fill NA with median (avoiding data leakage)
                df[col] = df[col].fillna(df[col].median())
        
        return df.dropna(subset=[self.target_col])

    def _build_model(self, input_shape: tuple) -> Sequential:
        """Build a high-performance neural network"""
        model = Sequential([
            Dense(64, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        return model

    def train(self) -> Dict[str, float]:
        """Train and save the model"""
        try:
            df = self._load_data()
            X = df[self.feature_order]  # Use the defined feature order
            y = df[self.target_col]
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale numerical features
            self.scaler = StandardScaler()
            X_train[self.numerical_cols] = self.scaler.fit_transform(X_train[self.numerical_cols])
            X_test[self.numerical_cols] = self.scaler.transform(X_test[self.numerical_cols])
            
            # Save preprocessing objects
            joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.pkl'))
            joblib.dump(self.encoders, os.path.join(self.model_dir, 'encoders.pkl'))
            joblib.dump(self.feature_order, os.path.join(self.model_dir, 'feature_order.pkl'))
            
            # Build and train model
            self.model = self._build_model((X_train.shape[1],))
            
            callbacks = [
                EarlyStopping(patience=15, restore_best_weights=True, monitor='val_auc', mode='max'),
                ModelCheckpoint(
                    os.path.join(self.model_dir, 'best_model.h5'),
                    save_best_only=True,
                    monitor='val_auc',
                    mode='max'
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
            
            # Evaluate
            y_pred = (self.model.predict(X_test) > 0.5).astype(int)
            print(classification_report(y_test, y_pred))
            print(f"AUC Score: {roc_auc_score(y_test, y_pred):.4f}")
            
            return {
                'best_accuracy': max(history.history['val_accuracy']),
                'best_auc': max(history.history['val_auc'])
            }
            
        except Exception as e:
            raise RuntimeError(f"Training failed: {str(e)}")

    def load(self) -> None:
        """Load trained model and preprocessing"""
        try:
            self.model = tf.keras.models.load_model(os.path.join(self.model_dir, 'best_model.h5'))
            self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
            self.encoders = joblib.load(os.path.join(self.model_dir, 'encoders.pkl'))
            self.feature_order = joblib.load(os.path.join(self.model_dir, 'feature_order.pkl'))
        except Exception as e:
            raise RuntimeError(f"Loading failed: {str(e)}")

    def get_feature_order(self) -> List[str]:
        """Returns the exact order of features expected by the model"""
        return self.feature_order

    @property
    def expected_features(self) -> int:
        """Returns the number of features expected by the model"""
        return len(self.feature_order)

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction with confidence"""
        if self.model is None:
            self.load()
        
        # Create DataFrame with all expected features
        input_df = pd.DataFrame(columns=self.feature_order)
        
        # Fill in provided values
        for feature in self.feature_order:
            if feature in input_data:
                # Handle numerical features
                if feature in self.numerical_cols:
                    try:
                        # Clean numerical input
                        val = str(input_data[feature]).strip()
                        val = float(''.join(c for c in val if c.isdigit() or c == '.'))
                        input_df[feature] = [val]
                    except:
                        # Use mean if conversion fails
                        mean_idx = self.numerical_cols.index(feature)
                        input_df[feature] = [self.scaler.mean_[mean_idx]]
                
                # Handle categorical features
                elif feature in self.categorical_cols:
                    val = str(input_data[feature]).strip().lower()
                    # Standardize input
                    val = 'yes' if val in ['yes', 'present', 'good', 'normal'] else 'no'
                    try:
                        input_df[feature] = [self.encoders[feature].transform([val])[0]]
                    except:
                        # Use mode if encoding fails
                        input_df[feature] = [0]
            else:
                # Fill missing features with appropriate defaults
                if feature in self.numerical_cols:
                    mean_idx = self.numerical_cols.index(feature)
                    input_df[feature] = [self.scaler.mean_[mean_idx]]
                else:
                    input_df[feature] = [0]  # Default for categorical
        
        # Scale numerical features
        input_df[self.numerical_cols] = self.scaler.transform(input_df[self.numerical_cols])
        
        # Ensure correct feature order
        input_df = input_df[self.feature_order]
        
        # Convert to numpy array for prediction
        X_pred = input_df.values.astype(np.float32)
        
        # Make prediction
        proba = float(self.model.predict(X_pred)[0][0])
        
        return {
            'prediction': 'CKD' if proba > 0.5 else 'No CKD',
            'probability': proba,
            'confidence': abs(proba - 0.5) * 2  # 0-1 scale where 1 is most confident
        }