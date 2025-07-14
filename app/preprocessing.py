import pandas as pd
import joblib

from features import generate_features

# Load komponen yang diperlukan (hanya sekali saat import modul ini)
rfe_selector    = joblib.load("models/rfe_selector.pkl")
ord_enc         = joblib.load("models/ordinal_encoder.pkl")
final_columns   = joblib.load("models/final_columns.pkl")

# Kolom kategorikal
total_ordinal_cols = ['tenure_bucket', 'credit_score_bucket']
total_nominal_cols = ['gender', 'country']

def preprocess_input(df_raw):
    """
    Preprocess 1 baris input user menjadi format siap prediksi (X_final)
    """
    df = df_raw.copy()

    # --- Step 1: Feature Engineering ---
    df_fe = generate_features(df)

    # --- Step 2: Encoding ---
    df_fe[total_ordinal_cols] = ord_enc.transform(df_fe[total_ordinal_cols])
    df_fe = pd.get_dummies(df_fe, columns=total_nominal_cols, drop_first=True)

    # --- Step 3: Sinkronisasi kolom hasil OHE ---
    df_fe = df_fe.reindex(columns=final_columns, fill_value=0)

    # --- Step 4: RFE Transform ---
    X_final = rfe_selector.transform(df_fe)

    return X_final
