from flask import Flask, render_template, jsonify, request, send_from_directory
import pandas as pd
import json
import os
from datetime import datetime
from model import AnomalyDetector, generate_sample_network_data
import traceback

app = Flask(__name__)

# Global variables
detector = None
latest_results = None
latest_summary = None

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train the anomaly detection model with sample data"""
    global detector, latest_results, latest_summary
    
    try:
        # Generate sample network data
        print("Generating sample network data...")
        data = generate_sample_network_data(1000)
        
        # Initialize and train the model
        print("Training anomaly detection model...")
        detector = AnomalyDetector(contamination=0.1)
        success = detector.train(data)
        
        if success:
            # Generate predictions on the training data
            print("Generating predictions...")
            latest_results = detector.predict(data)
            latest_summary = detector.get_anomaly_summary(latest_results)
            
            # Save results for later use
            latest_results.to_csv('sample_results.csv', index=False)
            with open('sample_summary.json', 'w') as f:
                json.dump(latest_summary, f, indent=2)
            
            return jsonify({
                'success': True,
                'message': 'Model trained successfully!',
                'summary': latest_summary
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Model training failed!'
            })
    
    except Exception as e:
        print(f"Training error: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Training error: {str(e)}'
        })

@app.route('/api/detect', methods=['POST'])
def detect_anomalies():
    """Detect anomalies in new data"""
    global detector, latest_results, latest_summary
    
    try:
        if detector is None or not detector.is_trained:
            return jsonify({
                'success': False,
                'message': 'Model not trained yet. Please train the model first.'
            })
        
        # For demo purposes, generate new sample data
        # In a real application, this would be real network data
        data = generate_sample_network_data(500)
        
        # Detect anomalies
        results = detector.predict(data)
        summary = detector.get_anomaly_summary(results)
        
        # Update global variables
        latest_results = results
        latest_summary = summary
        
        return jsonify({
            'success': True,
            'message': 'Anomaly detection completed!',
            'summary': summary
        })
    
    except Exception as e:
        print(f"Detection error: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Detection error: {str(e)}'
        })

@app.route('/api/results')
def get_results():
    """Get the latest anomaly detection results"""
    global latest_results, latest_summary
    
    try:
        if latest_results is None or latest_summary is None:
            return jsonify({
                'success': False,
                'message': 'No results available. Please train the model first.'
            })
        
        # Get anomalies only
        anomalies = latest_results[latest_results['is_anomaly'] == True]
        
        # Convert to JSON-serializable format
        anomaly_records = []
        for _, row in anomalies.head(20).iterrows():  # Limit to top 20
            record = {
                'timestamp': row['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(row['timestamp']) else 'N/A',
                'src_ip': row['src_ip'],
                'dst_ip': row['dst_ip'],
                'src_port': int(row['src_port']),
                'dst_port': int(row['dst_port']),
                'protocol': row['protocol'],
                'bytes_sent': round(float(row['bytes_sent']), 2),
                'bytes_received': round(float(row['bytes_received']), 2),
                'duration': round(float(row['duration']), 2),
                'packets': int(row['packets']),
                'anomaly_score': round(float(row['anomaly_score']), 4),
                'risk_level': row['risk_level']
            }
            anomaly_records.append(record)
        
        return jsonify({
            'success': True,
            'summary': latest_summary,
            'anomalies': anomaly_records
        })
    
    except Exception as e:
        print(f"Results error: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Results error: {str(e)}'
        })

@app.route('/api/stats')
def get_stats():
    """Get real-time statistics for the dashboard"""
    global latest_summary
    
    try:
        if latest_summary is None:
            # Return default stats if no model is trained
            return jsonify({
                'total_records': 0,
                'anomalies_detected': 0,
                'anomaly_percentage': 0.0,
                'critical_threats': 0,
                'high_threats': 0,
                'medium_threats': 0,
                'low_threats': 0
            })
        
        # Extract risk level counts
        risk_dist = latest_summary.get('risk_distribution', {})
        
        stats = {
            'total_records': latest_summary.get('total_records', 0),
            'anomalies_detected': latest_summary.get('anomalies_detected', 0),
            'anomaly_percentage': latest_summary.get('anomaly_percentage', 0.0),
            'critical_threats': risk_dist.get('Critical', 0),
            'high_threats': risk_dist.get('High', 0),
            'medium_threats': risk_dist.get('Medium', 0),
            'low_threats': risk_dist.get('Low', 0)
        }
        
        return jsonify(stats)
    
    except Exception as e:
        print(f"Stats error: {str(e)}")
        return jsonify({
            'total_records': 0,
            'anomalies_detected': 0,
            'anomaly_percentage': 0.0,
            'critical_threats': 0,
            'high_threats': 0,
            'medium_threats': 0,
            'low_threats': 0
        })

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_trained': detector is not None and detector.is_trained if detector else False,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

# Static file serving for CSS and JS
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    # Create templates and static directories if they don't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("\n" + "="*60)
    print("🛡️  ANOMALY DETECTION DASHBOARD STARTING 🛡️")
    print("="*60)
    print(f"📊 Dashboard URL: http://localhost:5000")
    print(f"🔧 API Health Check: http://localhost:5000/api/health")
    print(f"📈 Real-time Stats: http://localhost:5000/api/stats")
    print("="*60)
    print("\n🚀 Starting Flask server...\n")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
