import os
import pandas as pd
from joblib import load
import logging
import numpy as np

# Set up logging
logging.basicConfig(
    filename='vpn_recommendations.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def recommend_vpn(inputs: dict):
    """Recommend VPNs based on user inputs using ML model with proper scoring"""
    try:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base, 'models', 'model.pkl')
        data_path = os.path.join(base, 'data', 'cleaned_vpn_data.csv')

        # Load model and data
        model = load(model_path)
        df = pd.read_csv(data_path)
        
        # Standardize column names
        df.rename(columns={'name': 'vpn_name'}, inplace=True)
        df = df.dropna(subset=['vpn_name'])
        
        if df.empty:
            return create_empty_result(inputs, reason="Dataset is empty")

        logging.info(f"Loaded {len(df)} VPNs for recommendation")
        logging.info(f"User inputs: {inputs}")

        # Load encoders
        enc_encryption = load(os.path.join(base, 'models', 'enc_encryption.pkl'))
        enc_logging = load(os.path.join(base, 'models', 'enc_logging.pkl'))
        enc_handshake = load(os.path.join(base, 'models', 'enc_handshake.pkl'))

        # Validate and prepare user input features
        try:
            user_features = {
                'speed': float(inputs.get('speed', 0)),
                'price': float(inputs.get('price', 100)),
                'trial_available': 1 if str(inputs.get('trial_available', 'no')).lower() in ['yes', '1'] else 0,
                'data_encryption': inputs.get('encryption', 'AES-256'),
                'handshake_encryption': inputs.get('handshake_encryption', 'RSA-4096'),
                'logging_policy': inputs.get('logging_policy', 'no_logs'),
                'max_devices': int(inputs.get('max_devices', 1))
            }
        except ValueError as e:
            logging.error(f"Invalid input format: {str(e)}")
            return create_empty_result(inputs, reason=f"Invalid input format: {str(e)}")

        # Encode categorical features for user inputs
        try:
            user_features['data_encryption'] = enc_encryption.transform([user_features['data_encryption']])[0]
            user_features['handshake_encryption'] = enc_handshake.transform([user_features['handshake_encryption']])[0]
            user_features['logging_policy'] = enc_logging.transform([user_features['logging_policy']])[0]
        except ValueError as e:
            logging.warning(f"Encoding error: {str(e)}. Using defaults")
            user_features['data_encryption'] = enc_encryption.transform(['AES-256'])[0]
            user_features['handshake_encryption'] = enc_handshake.transform(['RSA-4096'])[0]
            user_features['logging_policy'] = enc_logging.transform(['no_logs'])[0]

        # Encode categorical features in the DataFrame
        try:
            df['data_encryption'] = enc_encryption.transform(df['data_encryption'])
            df['handshake_encryption'] = enc_handshake.transform(df['handshake_encryption'])
            df['logging_policy'] = enc_logging.transform(df['logging_policy'])
        except ValueError as e:
            logging.error(f"Data encoding failed: {str(e)}")
            return create_empty_result(inputs, reason="Data encoding failed")

        # Create input DataFrame with proper feature order
        feature_order = [
            'speed', 'price', 'trial_available', 
            'data_encryption', 'handshake_encryption',
            'logging_policy', 'max_devices'
        ]
        
        # Calculate scores for each VPN
        scores = []
        for _, row in df.iterrows():
            comparison_row = pd.DataFrame([{
                'speed': (user_features['speed'] + row['speed']) / 2,
                'price': (user_features['price'] + row['price']) / 2,
                'trial_available': user_features['trial_available'],
                'data_encryption': row['data_encryption'],
                'handshake_encryption': row['handshake_encryption'],
                'logging_policy': row['logging_policy'],
                'max_devices': (user_features['max_devices'] + row['max_devices']) / 2
            }])[feature_order]
            
            try:
                score = model.predict_proba(comparison_row)[0, 1] * 100
                scores.append(score)
            except Exception as e:
                logging.warning(f"Prediction failed for VPN {row['vpn_name']}: {str(e)}")
                scores.append(0)

        df['score'] = scores

        # Calculate feature similarity adjustments
        df['speed_sim'] = 1 - np.abs(df['speed'] - user_features['speed']) / 100
        df['price_sim'] = 1 - np.minimum(np.abs(df['price'] - user_features['price']) / 50, 1)
        df['encryption_sim'] = (df['data_encryption'] == user_features['data_encryption']).astype(int)
        df['logging_sim'] = (df['logging_policy'] == user_features['logging_policy']).astype(int)

        # Combine scores with weights
        df['score'] = (
            df['score'] * 0.6 +
            df['speed_sim'] * 0.15 +
            df['price_sim'] * 0.15 +
            df['encryption_sim'] * 0.05 +
            df['logging_sim'] * 0.05
        ).clip(0, 100).round(2)

        # Filter and sort results
        results = (
            df.sort_values('score', ascending=False)
            .head(10)
            .copy()
        )

        # If no good matches, use hybrid fallback
        if results.empty or results['score'].max() < 30:
            logging.warning("Using hybrid fallback scoring")
            results = (
                df.sort_values(
                    ['score', 'speed', 'price'], 
                    ascending=[False, False, True]
                )
                .head(5)
                .copy()
            )
            results['score'] = results['score'].clip(20, 80)

        # Decode categorical features for output
        results['trial_available'] = results['trial_available'].map({1: 'yes', 0: 'no'})
        results['data_encryption'] = enc_encryption.inverse_transform(results['data_encryption'])
        results['logging_policy'] = enc_logging.inverse_transform(results['logging_policy'])

        final_results = results[[
            'vpn_name', 'score', 'price', 'country',
            'logging_policy', 'trial_available', 'data_encryption', 'max_devices'
        ]].rename(columns={
            'trial_available': 'has_trial',
            'data_encryption': 'encryption'
        })

        logging.info(f"Returning {len(final_results)} recommendations")
        return final_results

    except Exception as e:
        logging.error(f"Recommendation failed: {str(e)}", exc_info=True)
        return create_empty_result(inputs, reason=f"System error: {str(e)}")

def create_empty_result(inputs, reason="No matches found"):
    return pd.DataFrame([{
        "vpn_name": "No recommendations available",
        "score": 0,
        "price": "-",
        "country": inputs.get('country', 'Unknown'),
        "logging_policy": inputs.get('logging_policy', 'N/A'),
        "has_trial": inputs.get('trial_available', 'N/A'),
        "encryption": inputs.get('encryption', 'N/A'),
        "max_devices": "-",
        "note": reason
    }])