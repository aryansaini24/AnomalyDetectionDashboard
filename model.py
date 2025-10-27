import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    def __init__(self, contamination=0.1):
        """
        Initialize the Anomaly Detector with Isolation Forest
        contamination: Expected proportion of outliers in the dataset
        """
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
    
    def prepare_features(self, data):
        """
        Prepare network/security features for anomaly detection
        """
        if 'timestamp' in data.columns:
            # Convert timestamp to datetime features
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data['hour'] = data['timestamp'].dt.hour
            data['day_of_week'] = data['timestamp'].dt.dayofweek
            data['minute'] = data['timestamp'].dt.minute
        
        # Create numerical features from network data
        feature_data = data.copy()
        
        # Handle categorical variables if present
        if 'protocol' in feature_data.columns:
            feature_data = pd.get_dummies(feature_data, columns=['protocol'], prefix='proto')
        
        if 'src_ip' in feature_data.columns:
            # Extract IP octets as features
            ip_parts = feature_data['src_ip'].str.split('.', expand=True)
            for i in range(4):
                feature_data[f'src_ip_octet_{i}'] = pd.to_numeric(ip_parts[i], errors='coerce')
        
        if 'dst_ip' in feature_data.columns:
            # Extract IP octets as features
            ip_parts = feature_data['dst_ip'].str.split('.', expand=True)
            for i in range(4):
                feature_data[f'dst_ip_octet_{i}'] = pd.to_numeric(ip_parts[i], errors='coerce')
        
        # Select numerical columns for training
        numerical_cols = feature_data.select_dtypes(include=[np.number]).columns
        feature_data = feature_data[numerical_cols]
        
        # Fill NaN values with median
        feature_data = feature_data.fillna(feature_data.median())
        
        return feature_data
    
    def train(self, data):
        """
        Train the anomaly detection model
        """
        try:
            # Prepare features
            feature_data = self.prepare_features(data)
            self.feature_columns = feature_data.columns.tolist()
            
            # Scale the features
            X_scaled = self.scaler.fit_transform(feature_data)
            
            # Train the model
            self.model.fit(X_scaled)
            self.is_trained = True
            
            print(f"Model trained successfully on {len(data)} samples")
            print(f"Features used: {len(self.feature_columns)}")
            
            return True
            
        except Exception as e:
            print(f"Training failed: {str(e)}")
            return False
    
    def predict(self, data):
        """
        Predict anomalies in new data
        Returns: DataFrame with original data + anomaly predictions and scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        try:
            # Prepare features
            feature_data = self.prepare_features(data.copy())
            
            # Ensure same features as training
            for col in self.feature_columns:
                if col not in feature_data.columns:
                    feature_data[col] = 0
            feature_data = feature_data[self.feature_columns]
            
            # Scale the features
            X_scaled = self.scaler.transform(feature_data)
            
            # Predict anomalies
            anomaly_labels = self.model.predict(X_scaled)  # 1 for normal, -1 for anomaly
            anomaly_scores = self.model.score_samples(X_scaled)  # Lower scores = more anomalous
            
            # Create result dataframe
            results = data.copy()
            results['is_anomaly'] = (anomaly_labels == -1)
            results['anomaly_score'] = anomaly_scores
            results['risk_level'] = self._calculate_risk_level(anomaly_scores)
            
            return results
            
        except Exception as e:
            print(f"Prediction failed: {str(e)}")
            return None
    
    def _calculate_risk_level(self, scores):
        """
        Calculate risk levels based on anomaly scores
        """
        # Convert scores to percentiles for risk assessment
        percentiles = np.percentile(scores, [25, 50, 75])
        
        risk_levels = []
        for score in scores:
            if score < percentiles[0]:
                risk_levels.append('Critical')
            elif score < percentiles[1]:
                risk_levels.append('High')
            elif score < percentiles[2]:
                risk_levels.append('Medium')
            else:
                risk_levels.append('Low')
        
        return risk_levels
    
    def save_model(self, filepath):
        """
        Save the trained model and scaler
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a pre-trained model and scaler
        """
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = model_data['is_trained']
            
            print(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"Failed to load model: {str(e)}")
            return False
    
    def get_anomaly_summary(self, predictions):
        """
        Get summary statistics of anomaly detection results
        """
        if predictions is None or predictions.empty:
            return {}
        
        total_records = len(predictions)
        anomalies = predictions[predictions['is_anomaly'] == True]
        
        summary = {
            'total_records': total_records,
            'anomalies_detected': len(anomalies),
            'anomaly_percentage': round((len(anomalies) / total_records) * 100, 2),
            'risk_distribution': predictions['risk_level'].value_counts().to_dict(),
            'avg_anomaly_score': round(predictions['anomaly_score'].mean(), 4),
            'min_anomaly_score': round(predictions['anomaly_score'].min(), 4),
            'max_anomaly_score': round(predictions['anomaly_score'].max(), 4)
        }
        
        return summary

def generate_sample_network_data(num_samples=1000):
    """
    Generate sample network log data for demonstration
    """
    np.random.seed(42)
    
    # Generate timestamps
    base_time = datetime.now()
    timestamps = [base_time - pd.Timedelta(minutes=x) for x in range(num_samples)]
    
    # Generate normal network traffic
    normal_data = {
        'timestamp': timestamps,
        'src_ip': [f"192.168.1.{np.random.randint(1, 255)}" for _ in range(num_samples)],
        'dst_ip': [f"10.0.0.{np.random.randint(1, 255)}" for _ in range(num_samples)],
        'src_port': np.random.randint(1000, 65535, num_samples),
        'dst_port': np.random.choice([80, 443, 22, 21, 25, 53], num_samples),
        'protocol': np.random.choice(['TCP', 'UDP', 'ICMP'], num_samples, p=[0.7, 0.25, 0.05]),
        'bytes_sent': np.random.exponential(1000, num_samples),
        'bytes_received': np.random.exponential(1500, num_samples),
        'duration': np.random.exponential(30, num_samples),
        'packets': np.random.poisson(10, num_samples)
    }
    
    # Add some anomalous patterns
    anomaly_indices = np.random.choice(num_samples, size=int(num_samples * 0.05), replace=False)
    
    for idx in anomaly_indices:
        # Create various types of anomalies
        anomaly_type = np.random.choice(['port_scan', 'data_exfiltration', 'ddos', 'unusual_traffic'])
        
        if anomaly_type == 'port_scan':
            normal_data['dst_port'][idx] = np.random.randint(1, 1024)  # Scanning common ports
            normal_data['bytes_sent'][idx] *= 0.1  # Small packets
            normal_data['packets'][idx] *= 10  # Many packets
        elif anomaly_type == 'data_exfiltration':
            normal_data['bytes_sent'][idx] *= 100  # Large data transfer
            normal_data['duration'][idx] *= 10  # Long connection
        elif anomaly_type == 'ddos':
            normal_data['packets'][idx] *= 50  # High packet count
            normal_data['duration'][idx] *= 0.1  # Short duration
        elif anomaly_type == 'unusual_traffic':
            normal_data['dst_port'][idx] = np.random.randint(8000, 9999)  # Unusual port
            normal_data['protocol'][idx] = 'ICMP'  # Unusual protocol combination
    
    return pd.DataFrame(normal_data)

if __name__ == "__main__":
    # Demo usage
    print("Generating sample network data...")
    data = generate_sample_network_data(1000)
    
    print("Training anomaly detection model...")
    detector = AnomalyDetector(contamination=0.1)
    detector.train(data)
    
    print("\nDetecting anomalies...")
    results = detector.predict(data)
    summary = detector.get_anomaly_summary(results)
    
    print("\n=== Anomaly Detection Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Save sample results for the web app
    results.to_csv('sample_results.csv', index=False)
    with open('sample_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nSample data and results saved!")
