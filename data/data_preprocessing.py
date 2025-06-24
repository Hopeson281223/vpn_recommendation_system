import os
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from transformers import pipeline
import re

# Initialize AI components
geolocator = Nominatim(user_agent="vpn_cleaner")
logging_classifier = pipeline("text-classification", model="distilbert-base-uncased")

def load_and_clean_data(input_path, output_path):
    print(f"📂 Loading data from {input_path}")
    df = pd.read_csv(input_path)
    
    try:
        df.columns = df.columns.str.strip().str.upper()
        df_cleaned = clean_vpn_data(df)
        df_cleaned.to_csv(output_path, index=False)
        print(f"✅ Saved cleaned data to {output_path} ({len(df_cleaned)} records)")
        return df_cleaned
    except Exception as e:
        print(f"❌ Error cleaning data: {str(e)}")
        raise

def ai_standardize_country(country):
    try:
        location = geolocator.geocode(country)
        if location:
            return location.address.split(",")[-1].strip()
    except:
        pass
    return country  # fallback

def ai_classify_logging(text):
    if pd.isna(text): return 'no_logs'
    result = logging_classifier(text[:512])
    return 'no_logs' if result[0]['label'].upper() == 'NO_LOGS' else 'partial_logs'

def standardize_encryption(text):
    if not isinstance(text, str) or str(text).strip().lower() in ['', 'nan', 'none']:
        return 'Unknown'
    
    text = str(text).strip().upper()
    # Handle multiple encryption types separated by slashes or commas
    if '/' in text or ',' in text:
        encryptions = re.split(r'[/,]', text)
        standardized = []
        for enc in encryptions:
            standardized.append(standardize_single_encryption(enc.strip()))
        return ', '.join(filter(lambda x: x != 'Unknown', standardized))
    
    return standardize_single_encryption(text)

def standardize_single_encryption(text):
    text = re.sub(r'[\s\-_]', '', text.upper())
    
    # Match common encryption algorithms with improved patterns
    patterns = [
        (r'AES(\d+)', 'AES-{}'),
        (r'BLOWFISH(\d+)', 'Blowfish-{}'),
        (r'CAMELLIA(\d+)', 'Camellia-{}'),
        (r'RSA(\d+)', 'RSA-{}'),
        (r'DH(\d+)', 'DH-{}'),
        (r'CHACHA20(\S*)', 'ChaCha20'),
        (r'SHA(\d+)', 'SHA-{}'),
        (r'MPPE(\d*)', 'MPPE'),
        (r'MSCHAP(\S*)', 'MS-CHAP'),
        (r'CA(\d+)', 'CA-{}'),
        (r'3DES', '3DES'),
        (r'RC4', 'RC4')
    ]
    
    for pattern, replacement in patterns:
        match = re.search(pattern, text)
        if match:
            if '{}' in replacement:
                return replacement.format(match.group(1))
            return replacement
            
    return 'Unknown'

def clean_vpn_data(df):
    # First ensure the VPN name column exists
    if 'VPN SERVICE' not in df.columns:
        raise ValueError("VPN name column 'VPN SERVICE' not found in dataset")
    
    # Create the name column first
    df['name'] = df['VPN SERVICE'].str.strip()
    
    column_mapping = {
        'country': 'JURISDICTION BASED IN (COUNTRY)',
        'speed': 'SPEEDS US SERVER AVERAGE (%)',
        'price': 'PRICING $ / MONTH (ANNUAL PRICING)',
        'max_devices': 'AVAILABILITY # OF CONNECTIONS',
        'encryption': 'SECURITY DEFAULT DATA ENCRYPTION',
        'strongest_encryption': 'SECURITY STRONGEST DATA ENCRYPTION',
        'handshake_encryption': 'SECURITY STRONGEST HANDSHAKE ENCRYPTION',
        'trial_available': 'PRICING FREE TRIAL',
        'logging_traffic': 'LOGGING LOGS TRAFFIC',
        'logging_dns': 'LOGGING LOGS DNS REQUESTS'
    }

    # Process basic columns
    df['speed'] = pd.to_numeric(df.get(column_mapping['speed'], 50), errors='coerce').fillna(50) * 0.1
    df['price'] = pd.to_numeric(df.get(column_mapping['price'], 10), errors='coerce').fillna(10.0)
    df['max_devices'] = pd.to_numeric(df.get(column_mapping['max_devices'], 1), errors='coerce').fillna(1).astype(int)

    # Process logging policies
    logging_fields = [
        column_mapping['logging_traffic'],
        column_mapping['logging_dns'],
        'LOGGING LOGS TIMESTAMPS',
        'LOGGING LOGS BANDWIDTH',
        'LOGGING LOGS IP ADDRESS'
    ]

    df['logging_policy'] = df.apply(
        lambda row: 'no_logs' if all(
            str(row.get(field, 'no')).strip().lower() in ['no', 'nan', '']
            for field in logging_fields
        ) else 'partial_logs',
        axis=1
    )

    # Process encryption columns
    for enc_type in ['encryption', 'strongest_encryption', 'handshake_encryption']:
        if column_mapping[enc_type] not in df.columns:
            df[column_mapping[enc_type]] = 'Unknown'

    df['default_encryption'] = df[column_mapping['encryption']].apply(standardize_encryption)
    df['strongest_encryption'] = df[column_mapping['strongest_encryption']].apply(standardize_encryption)
    df['handshake_encryption'] = df[column_mapping['handshake_encryption']].apply(standardize_encryption)

    # Combine encryption info
    df['encryption'] = df.apply(
        lambda row: ', '.join(filter(
            lambda x: x != 'Unknown',
            set([str(row['default_encryption']), str(row['strongest_encryption']), str(row['handshake_encryption'])])
        )) or 'Unknown',
        axis=1
    )

    # Process other columns
    df['trial_available'] = df.get(column_mapping['trial_available'], 'No').map(
        {'Yes': 'yes', 'No': 'no'}
    ).fillna('no')

    df['country'] = df.get(column_mapping['country'], 'Unknown').apply(
        lambda x: ai_standardize_country(x) if pd.notna(x) else 'Unknown'
    )

    # Select final columns
    final_columns = {
        'name': 'name',
        'country': 'country',
        'speed': 'speed',
        'price': 'price',
        'max_devices': 'max_devices',
        'logging_policy': 'logging_policy',
        'encryption': 'encryption',
        'default_encryption': 'default_encryption',
        'strongest_encryption': 'strongest_encryption',
        'handshake_encryption': 'handshake_encryption',
        'trial_available': 'trial_available'
    }

    return df[list(final_columns.keys())].rename(columns=final_columns).dropna()

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(base, 'vpn_data_real.csv')
    cleaned_path = os.path.join(base, 'cleaned_vpn_data.csv')
    os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)
    load_and_clean_data(raw_path, cleaned_path)