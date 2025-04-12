import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import joblib
import os
from typing import Tuple, Dict
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

class DiabetesDiagnosisPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.expected_features = None
        self.loaded = False
        self.model_dir = os.path.join('models', 'diagnosis')
        os.makedirs(self.model_dir, exist_ok=True)

    def load_and_preprocess_data(self) -> pd.DataFrame:
        """Load and preprocess Pima Indians dataset"""
        df = pd.read_csv('data/pima_indians_diabetes.csv')
        
        # Handle zeros in biological features
        zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        df[zero_features] = df[zero_features].replace(0, np.nan)
        
        # Impute missing values by outcome class
        for col in zero_features:
            df[col] = df.groupby('Outcome')[col].transform(lambda x: x.fillna(x.median()))
        
        return df

    def prepare_data(self, df: pd.DataFrame) -> Tuple:
        """Prepare data for training"""
        X = df.drop('Outcome', axis=1)
        self.expected_features = X.columns.tolist()
        y = df['Outcome']

        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Feature scaling
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Save preprocessing objects
        joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.pkl'))
        joblib.dump(self.expected_features, 
                   os.path.join(self.model_dir, 'feature_names.pkl'))
        
        return X_train, X_test, y_train, y_test

    def evaluate_model(self, model, X_test, y_test) -> Dict:
        """Comprehensive model evaluation"""
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'f1_score': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }

    def train(self) -> dict:
        """Train optimized XGBoost model"""
        diabetes_df = self.load_and_preprocess_data()
        X_train, X_test, y_train, y_test = self.prepare_data(diabetes_df)
        
        # Create SMOTE object for handling class imbalance
        smote = SMOTE(random_state=42)
        
        # Create XGBoost model
        xgb = XGBClassifier(
            random_state=42, 
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        # Create pipeline
        pipeline = ImbPipeline([
            ('smote', smote),
            ('model', xgb)
        ])
        
        # Hyperparameter grid
        param_grid = {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__max_depth': [3, 5, 7],
            'model__subsample': [0.8, 1.0],
            'model__colsample_bytree': [0.8, 1.0]
        }
        
        # 5-fold stratified cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Grid search
        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid.fit(X_train, y_train)
        
        # Get best model
        self.model = grid.best_estimator_.named_steps['model']
        
        # Evaluate on test set
        test_metrics = self.evaluate_model(grid.best_estimator_, X_test, y_test)
        
        # Save the trained model
        joblib.dump(grid.best_estimator_, os.path.join(self.model_dir, 'xgb_model.pkl'))
        
        return {
            'best_params': grid.best_params_,
            'test_metrics': test_metrics
        }

    def load(self) -> None:
        """Load trained model"""
        model_path = os.path.join(self.model_dir, 'xgb_model.pkl')
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        features_path = os.path.join(self.model_dir, 'feature_names.pkl')

        if not all(os.path.exists(p) for p in [model_path, scaler_path, features_path]):
            raise FileNotFoundError("Model files not found. Train the model first.")

        loaded_pipeline = joblib.load(model_path)
        self.model = loaded_pipeline.named_steps['model']
        self.scaler = joblib.load(scaler_path)
        self.expected_features = joblib.load(features_path)
        self.loaded = True

    def predict(self, input_data: dict) -> float:
        """Make prediction - returns probability only (for compatibility)"""
        if not self.loaded:
            self.load()

        # Create input DataFrame with expected features
        input_df = pd.DataFrame(columns=self.expected_features)
        
        # Fill with input data, use 0 for missing features
        for feature in self.expected_features:
            input_df[feature] = [input_data.get(feature, 0)]
        
        # Scale features
        scaled_input = self.scaler.transform(input_df)
        
        # Get probability
        return float(self.model.predict_proba(scaled_input)[0][1])