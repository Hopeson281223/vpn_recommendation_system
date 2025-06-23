import os
import pandas as pd
import numpy as np

def load_and_clean_data(input_path, output_path):
    print(f"📂 Loading data from {input_path}")
    df = pd.read_csv(input_path)

    try:
        # Standardize column names
        df.columns = df.columns.str.strip().str.upper()

        # Clean and transform data
        df_cleaned = clean_vpn_data(df)

        # Save cleaned data
        df_cleaned.to_csv(output_path, index=False)
        print(f"✅ Saved cleaned data to {output_path} ({len(df_cleaned)} records)")
        return df_cleaned

    except Exception as e:
        print(f"❌ Error cleaning data: {str(e)}")
        raise

def clean_vpn_data(df):
    country_mapping = {
        'USA': 'United States',
        'UK': 'United Kingdom',
        'UAE': 'United Arab Emirates',
        'CZECH REPUBLIC': 'Czechia',
    }
    df['JURISDICTION BASED IN (COUNTRY)'] = df['JURISDICTION BASED IN (COUNTRY)'].astype(str).str.strip().replace(country_mapping)

    logging_fields = [
        'LOGGING LOGS TRAFFIC',
        'LOGGING LOGS DNS REQUESTS',
        'LOGGING LOGS TIMESTAMPS',
        'LOGGING LOGS BANDWIDTH',
        'LOGGING LOGS IP ADDRESS'
    ]
    for field in logging_fields:
        if field not in df.columns:
            df[field] = np.nan

    def determine_logging_policy(row):
        values = [str(row[field]).strip().lower() for field in logging_fields]
        if all(v in ['no', 'nan', ''] for v in values):
            return 'no_logs'
        elif any(v == 'yes' for v in values):
            return 'partial_logs'
        else:
            return 'full_logs'

    df['logging_policy'] = df.apply(determine_logging_policy, axis=1)

    df['encryption'] = df.get('SECURITY DEFAULT DATA ENCRYPTION', 'AES-256')
    df['encryption'] = df['encryption'].fillna('AES-256').apply(lambda x: 'ChaCha20' if 'ChaCha' in str(x) else 'AES-256')

    df['trial_available'] = df['PRICING FREE TRIAL'].map({'Yes': 'yes', 'No': 'no'}).fillna('no')

    df['price'] = pd.to_numeric(df['PRICING $ / MONTH (ANNUAL PRICING)'], errors='coerce')
    df['price'] = df['price'].fillna(df['price'].median())

    df['max_devices'] = pd.to_numeric(df['AVAILABILITY # OF CONNECTIONS'], errors='coerce').fillna(1).astype(int)

    df['speed'] = pd.to_numeric(df['SPEEDS US SERVER AVERAGE (%)'], errors='coerce').fillna(50) * 0.1

    final_columns = {
        'VPN SERVICE': 'name',
        'JURISDICTION BASED IN (COUNTRY)': 'country',
        'speed': 'speed',
        'price': 'price',
        'max_devices': 'max_devices',
        'logging_policy': 'logging_policy',
        'encryption': 'encryption',
        'trial_available': 'trial_available'
    }

    selected = df[list(final_columns.keys())].rename(columns=final_columns)
    return selected.dropna()

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(base, 'vpn_data_real.csv')   # <- Adjusted path
    cleaned_path = os.path.join(base, 'cleaned_vpn_data.csv')        # <- Output path

    os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)
    load_and_clean_data(raw_path, cleaned_path)
