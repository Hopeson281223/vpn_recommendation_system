import os
import pandas as pd
import numpy as np
from joblib import load
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder

class PreferenceLearner:
    def __init__(self):
        self.model = SGDClassifier(loss='log_loss', warm_start=True)
        self.features = ['price', 'speed', 'logging_policy', 'trial_available']
        # Initialize with dummy data
        dummy_X = [[0, 0, 0, 0]]
        dummy_y = [0]
        self.model.partial_fit(dummy_X, dummy_y, classes=[0, 1])
    
    def update(self, user_feedback):
        """Learn from user feedback"""
        X = pd.DataFrame([user_feedback['features']])
        y = [user_feedback['rating']]
        self.model.partial_fit(X, y)
    
    def get_weights(self):
        """Get feature weights"""
        return np.abs(self.model.coef_[0]) if hasattr(self.model, 'coef_') else np.ones(len(self.features))

def recommend_vpn(inputs):
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = load(os.path.join(base, 'models', 'model.pkl'))
    df = pd.read_csv(os.path.join(base, 'data', 'cleaned_vpn_data.csv'))
    
    # Fill missing encryption data
    encryption_defaults = {
        'encryption': 'AES-256',
        'default_encryption': 'AES-256',
        'strongest_encryption': 'AES-256',
        'handshake_encryption': 'RSA-4096'
    }
    for col, default in encryption_defaults.items():
        if col in df.columns:
            df[col] = df[col].replace(['Unknown', 'nan', '', None], default).fillna(default)

    if df.empty:
        return pd.DataFrame(columns=['vpn_name', 'country', 'score'])
    
    # Initialize PreferenceLearner
    learner = PreferenceLearner()
    
    # Load encoders with fallbacks
    encoder_files = [
        'country', 'logging_policy', 'encryption', 
        'default_encryption', 'strongest_encryption', 
        'handshake_encryption'
    ]
    encoders = {}
    for col in encoder_files:
        try:
            encoders[col] = load(os.path.join(base, 'models', f'enc_{col}.pkl'))
        except:
            encoders[col] = LabelEncoder().fit(['Unknown'])

    # Process user inputs with defaults
    user_features = {
        'speed': float(inputs.get('speed', 5)),
        'price': float(inputs.get('price', 10)),
        'trial_available': 1 if str(inputs.get('trial_available', 'no')).lower() == 'yes' else 0,
        'max_devices': int(inputs.get('max_devices', 1)),
        'encryption': inputs.get('encryption', 'AES-256'),
        'logging_policy': inputs.get('logging_policy', 'no_logs'),
        'country': inputs.get('country', '').strip(),
        'default_encryption': inputs.get('default_encryption', 'AES-256'),
        'strongest_encryption': inputs.get('strongest_encryption', 'AES-256'),
        'handshake_encryption': inputs.get('handshake_encryption', 'RSA-4096')
    }

    # Safe encoding function
    def safe_encode(encoder, value):
        try:
            if value in encoder.classes_:
                return encoder.transform([value])[0]
            return encoder.transform(['Unknown'])[0]
        except:
            return 0

    # Encode all features
    for feature in encoder_files:
        user_features[f'{feature}_encoded'] = safe_encode(encoders[feature], user_features.get(feature, 'Unknown'))

    # Enhanced country handling - don't filter, just score
    if user_features['country']:
        df['country_match'] = df['country'].str.lower().apply(
            lambda x: 1.0 if x == user_features['country'].lower() else 
                     0.7 if user_features['country'].lower() in x else  # partial match
                     0.3  # other countries
        )
    else:
        df['country_match'] = 0.5  # neutral score when no country specified

    # Encode dataframe columns
    for col in encoder_files:
        if col in df.columns:
            df[f'{col}_encoded'] = df[col].apply(lambda x: safe_encode(encoders[col], x))

    # Feature order must match training
    feature_order = [
        'speed', 'price', 'max_devices', 'trial_available',
        'country_encoded', 'logging_policy_encoded',
        'encryption_encoded', 'default_encryption_encoded', 
        'strongest_encryption_encoded', 'handshake_encryption_encoded'
    ]

    def calculate_score(row):
        try:
            X = pd.DataFrame([[
                row['speed'],
                row['price'],
                row['max_devices'],
                user_features['trial_available'],
                user_features['country_encoded'],
                user_features['logging_policy_encoded'],
                row['encryption_encoded'],
                row['default_encryption_encoded'],
                row['strongest_encryption_encoded'],
                row['handshake_encryption_encoded']
            ]], columns=feature_order)
            return model.predict_proba(X)[0][1] * 100
        except Exception as e:
            print(f"Error calculating score: {e}")
            return 0

    df['base_score'] = df.apply(calculate_score, axis=1)
    
    # Apply personalized weights with country consideration
    weights = learner.get_weights()
    df['personalized_score'] = (
        df['base_score'] * 0.6 +  # base model score
        df['country_match'] * 30 +  # country importance (0-30 points)
        (1 - np.abs(df['price'] - user_features['price']) / 20 * weights[0] * 100 * 0.2 +
        (df['speed'] / 10) * weights[1] * 100 * 0.2
    ))

    # Apply strict filters if specified
    if user_features['logging_policy'] == 'no_logs':
        df = df[df['logging_policy'] == 'no_logs']
    
    if 'max_devices' in inputs:
        df = df[df['max_devices'] >= user_features['max_devices']]

    return df.nlargest(5, 'personalized_score')[[
        'name', 'country', 'price', 'speed', 'base_score', 
        'logging_policy', 'encryption', 'max_devices', 
        'trial_available', 'personalized_score'
    ]].rename(columns={
        'name': 'vpn_name',
        'base_score': 'score'
    })