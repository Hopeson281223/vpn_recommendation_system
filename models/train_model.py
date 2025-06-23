import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from joblib import dump

# === Paths ===
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Project root
data_path = os.path.join(base, 'data', 'cleaned_vpn_data.csv')
model_dir = os.path.join(base, 'models')
model_path = os.path.join(model_dir, 'model.pkl')

# === Load cleaned data ===
df = pd.read_csv(data_path)

# === Preprocessing ===
df['trial_available'] = df['trial_available'].map({'yes': 1, 'no': 0}).fillna(0).astype(int)

feature_columns = [
    'speed',
    'price',
    'trial_available',
    'data_encryption',
    'handshake_encryption',
    'logging_policy',
    'max_devices'
]

df = df.dropna(subset=feature_columns)

# === Label Logic for Classification ===
def calculate_label(row):
    good_data_enc = row['data_encryption'] in ['AES-256', 'ChaCha20']
    good_handshake = 'RSA-4096' in row['handshake_encryption'] or 'ECDHE' in str(row['handshake_encryption'])
    return int(
        row['logging_policy'] in ['no_logs', 'partial_logs'] and
        good_data_enc and
        good_handshake and
        row['price'] <= 15 and
        row['speed'] >= 4
    )

df['label'] = df.apply(calculate_label, axis=1)

# === Encode Categorical Features ===
enc_encryption = LabelEncoder()
enc_handshake = LabelEncoder()
enc_logging = LabelEncoder()

df['data_encryption'] = enc_encryption.fit_transform(df['data_encryption'])
df['handshake_encryption'] = enc_handshake.fit_transform(df['handshake_encryption'])
df['logging_policy'] = enc_logging.fit_transform(df['logging_policy'])

# === Save Encoders ===
os.makedirs(model_dir, exist_ok=True)
dump(enc_encryption, os.path.join(model_dir, 'enc_encryption.pkl'))
dump(enc_handshake, os.path.join(model_dir, 'enc_handshake.pkl'))  # Optional unless needed in tes.py
dump(enc_logging, os.path.join(model_dir, 'enc_logging.pkl'))

# === Model Training ===
X = df[feature_columns]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)
dump(model, model_path)

# === Report ===
print(f"✅ Model trained and saved to: {model_path}")
print(f"✅ Accuracy: {model.score(X_test, y_test):.2f}")
print(f"✅ Encoders saved in: {model_dir}")
