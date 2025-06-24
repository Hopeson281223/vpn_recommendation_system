import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from joblib import dump

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base, 'data', 'cleaned_vpn_data.csv')
model_dir = os.path.join(base, 'models')
os.makedirs(model_dir, exist_ok=True)

def load_data():
    df = pd.read_csv(data_path)
    
    # Convert yes/no to 1/0
    df['trial_available'] = df['trial_available'].map({'yes': 1, 'no': 0})
    
    # Encode categorical columns
    categorical_cols = ['country', 'logging_policy', 'encryption', 
                        'default_encryption', 'strongest_encryption', 
                        'handshake_encryption']
    
    encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            encoder = LabelEncoder()
            df[f'{col}_encoded'] = encoder.fit_transform(df[col].astype(str))
            encoders[col] = encoder
            dump(encoder, os.path.join(model_dir, f'enc_{col}.pkl'))
    
    # AI-enhanced label generation
    X = df[['speed', 'price', 'max_devices', 'country_encoded']].values
    y = np.where(
        (df['logging_policy'] == 'no_logs') & 
        (df['price'] <= 15) & 
        (df['speed'] >= 4),
        1,  # Good VPN
        0   # Bad VPN
    )
    
    # Refine labels with semi-supervised learning
    label_prop = LabelPropagation(kernel='rbf').fit(X, y)
    df['label'] = label_prop.transduction_
    return df, encoders

def train():
    df, encoders = load_data()
    
    # Prepare features - include all available columns
    feature_cols = [
        'speed', 'price', 'max_devices', 'trial_available',
        'country_encoded', 'logging_policy_encoded',
        'encryption_encoded', 'default_encryption_encoded',
        'strongest_encryption_encoded', 'handshake_encryption_encoded'
    ]
    
    # Only include columns that exist in the dataframe
    available_features = [col for col in feature_cols if col in df.columns]
    X = df[available_features]
    y = df['label']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100, 
        class_weight='balanced', 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Save model and print accuracy
    dump(model, os.path.join(model_dir, 'model.pkl'))
    print(f"✅ Model trained. Accuracy: {model.score(X_test, y_test):.2f}")
    print(f"Features used: {available_features}")

if __name__ == "__main__":
    train()