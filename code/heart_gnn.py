import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
from typing import Dict
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

class HeartDiseaseGNNPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.encoders = {}
        self.expected_features = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        self.model_dir = os.path.join('models', 'heart_gnn')
        os.makedirs(self.model_dir, exist_ok=True)

    def _get_model_path(self, filename):
        """Helper to get full path with validation"""
        path = os.path.join(self.model_dir, filename)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def load_data(self) -> pd.DataFrame:
        """Load data with proper path handling"""
        data_files = [
            'processed.cleveland.data',
            'processed.hungarian.data',
            'processed.switzerland.data',
            'processed.va.data'
        ]
        
        dfs = []
        data_path = os.path.join('data', 'heart')
        
        for file in data_files:
            try:
                file_path = os.path.join(data_path, file)
                if not os.path.exists(file_path):
                    print(f"Warning: {file} not found at {file_path}")
                    continue
                df = pd.read_csv(file_path, header=None, na_values='?')
                df.columns = self.expected_features + ['target']
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
                continue
        
        if not dfs:
            raise FileNotFoundError(f"No heart disease data found in {data_path}")
        
        combined = pd.concat(dfs, axis=0)
        combined['target'] = combined['target'].apply(lambda x: 0 if x == 0 else 1)
        combined['ca'] = combined['ca'].fillna(combined['ca'].median())
        combined['thal'] = combined['thal'].fillna(combined['thal'].median())
        return combined.dropna()

    class GraphNeuralNetwork(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.att = nn.Linear(64, 64)
            self.fc2 = nn.Linear(64, 32)
            self.out = nn.Linear(32, 1)
            
        def forward(self, x, adj=None):
            x = F.relu(self.fc1(x))
            if adj is not None:
                attn = torch.sigmoid(self.att(x))
                x = x * attn
                x = torch.matmul(adj, x)
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu(self.fc2(x))
            return torch.sigmoid(self.out(x))

    def _create_adjacency(self, X):
        """Create adjacency matrix with checks"""
        X = np.asarray(X, dtype=np.float32)
        if len(X.shape) != 2:
            raise ValueError("Input must be 2D array")
            
        distances = np.sqrt(((X[:, None] - X) ** 2).sum(axis=2))
        sigma = np.mean(distances)
        adj = np.exp(-distances / (2 * sigma ** 2))
        np.fill_diagonal(adj, 0)
        return torch.FloatTensor(adj)

    def train(self) -> Dict:
        """Train with proper path handling"""
        try:
            df = self.load_data()
            X = df[self.expected_features]
            y = df['target']
            
            cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
            for col in cat_cols:
                encoder = LabelEncoder()
                X[col] = encoder.fit_transform(X[col])
                self.encoders[col] = encoder
            
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            adj = self._create_adjacency(X_scaled)
            X_tensor = torch.FloatTensor(X_scaled)
            y_tensor = torch.FloatTensor(y.values)
            train_idx, test_idx = train_test_split(range(len(y)), test_size=0.2, random_state=42)
            
            self.model = self.GraphNeuralNetwork(X_scaled.shape[1])
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
            criterion = nn.BCELoss()
            
            best_loss = float('inf')
            for epoch in range(100):
                self.model.train()
                optimizer.zero_grad()
                outputs = self.model(X_tensor, adj).squeeze()
                loss = criterion(outputs[train_idx], y_tensor[train_idx])
                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    val_loss = criterion(outputs[test_idx], y_tensor[test_idx])
                    if val_loss < best_loss:
                        best_loss = val_loss
                        torch.save(self.model.state_dict(), self._get_model_path('best_model.pt'))
            
            joblib.dump(self.scaler, self._get_model_path('scaler.pkl'))
            joblib.dump(self.encoders, self._get_model_path('encoders.pkl'))
            
            return {'best_val_loss': best_loss.item()}
            
        except Exception as e:
            raise RuntimeError(f"Training failed: {str(e)}")

    def load(self):
        """Load with explicit error checking"""
        try:
            model_path = self._get_model_path('best_model.pt')
            scaler_path = self._get_model_path('scaler.pkl')
            encoders_path = self._get_model_path('encoders.pkl')
            
            if not all(os.path.exists(p) for p in [model_path, scaler_path, encoders_path]):
                raise FileNotFoundError("Required model files not found")
                
            self.model = self.GraphNeuralNetwork(len(self.expected_features))
            self.model.load_state_dict(torch.load(model_path))
            self.scaler = joblib.load(scaler_path)
            self.encoders = joblib.load(encoders_path)
            
        except Exception as e:
            raise RuntimeError(f"Loading failed: {str(e)}")

    def predict(self, input_data: Dict) -> float:
        """Predict with error handling"""
        try:
            if not hasattr(self, 'model') or self.model is None:
                self.load()
            
            input_df = pd.DataFrame([input_data], columns=self.expected_features)
            for col, encoder in self.encoders.items():
                input_df[col] = encoder.transform(input_df[col])
            
            X_scaled = self.scaler.transform(input_df)
            X_tensor = torch.FloatTensor(X_scaled)
            
            self.model.eval()
            with torch.no_grad():
                return float(self.model(X_tensor).item())
                
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def explain_prediction(self, input_data: Dict = None, generic: bool = False):
        """Show simplified GNN architecture diagram"""
        try:
            plt.figure(figsize=(10, 6))
            
            # Draw simplified GNN architecture
            plt.title("How Our GNN Model Works", pad=20, fontsize=16)
            
            # Central prediction node
            plt.scatter(0.5, 0.7, s=5000, c='salmon', alpha=0.7)
            plt.text(0.5, 0.7, "Heart Disease\nRisk Prediction", 
                    ha='center', va='center', fontsize=12)
            
            # Feature nodes
            features = [
                (0.2, 0.9, "Age"), 
                (0.8, 0.9, "Cholesterol"),
                (0.1, 0.5, "Blood\nPressure"), 
                (0.9, 0.5, "Heart\nRate"),
                (0.3, 0.3, "Chest\nPain"), 
                (0.7, 0.3, "ST\nDepression")
            ]
            
            for x, y, label in features:
                plt.scatter(x, y, s=3000, c='lightblue', alpha=0.7)
                plt.text(x, y, label, ha='center', va='center', fontsize=10)
                plt.plot([x, 0.5], [y, 0.7], 'k-', alpha=0.3, linewidth=2)
            
            # Explanation text
            plt.text(0.5, 0.1, 
                    "GNNs analyze relationships between features\n"
                    "by passing messages through the graph structure,\n"
                    "capturing how health metrics influence each other", 
                    ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))
            
            plt.axis('off')
            plt.tight_layout()
            
            return {
                'figure': plt.gcf(),
                'type': 'gnn',
                'description': "GNNs analyze feature relationships through graph structures"
            }
            
        except Exception as e:
            st.error(f"Visualization failed: {str(e)}")
            return None