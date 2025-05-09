from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = joblib.load(r'C:\Users\Shreyansh Singh\Desktop\splunk_hackathon\encodings\xgb_model.pkl')
scaler = joblib.load(r'C:\Users\Shreyansh Singh\Desktop\splunk_hackathon\encodings\esg_scaler.pkl')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model': 'xgboost-esg-risk'
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        data = request.get_json()
        
        # Extract features in correct order
        features = [
            data['total_esg_risk_score'],
            data['environment_risk_score'],
            data['governance_risk_score'],
            data['social_risk_score'],
            data['controversy_level'],
            data['controversy_score'],
            data['esg_risk_percentile']
        ]
        
        # Convert to numpy array and reshape
        features = np.array(features).reshape(1, -1)
        
        # Scale the features
        scaled_features = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(scaled_features)
        prediction_proba = model.predict_proba(scaled_features)
        
        # Get the probability of predicted class
        probability = float(max(prediction_proba[0]))
        
        # Return prediction and probability
        return jsonify({
            'esg_risk_level': int(prediction[0]),
            'probability': probability
        })
        
    except KeyError as e:
        return jsonify({
            'error': f'Missing required feature: {str(e)}'
        }), 400
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)