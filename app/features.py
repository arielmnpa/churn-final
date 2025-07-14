# generate_features.py

import pandas as pd
import numpy as np

def generate_features(df):
    df_fe = df.copy()

    # --- Basic Ratio & Log Transform ---
    df_fe["balance_per_product"]          = df_fe["balance"] / df_fe["products_number"].replace(0, 1)
    df_fe["balance_per_product_log"]      = np.log1p(df_fe["balance_per_product"])

    df_fe["age_tenure_ratio"]             = df_fe["age"] / (df_fe["tenure"] + 1)
    df_fe["age_tenure_ratio_log"]         = np.log1p(df_fe["age_tenure_ratio"])

    df_fe["balance_to_salary_ratio"]      = df_fe["balance"] / df_fe["estimated_salary"].replace(0, 1)
    df_fe["balance_to_salary_ratio_log"]  = np.log1p(df_fe["balance_to_salary_ratio"])

    df_fe["product_per_age"]              = df_fe["products_number"] / df_fe["age"].replace(0, 1)
    df_fe["product_per_age_log"]          = np.log1p(df_fe["product_per_age"])

    df_fe["products_per_tenure"]          = df_fe["products_number"] / (df_fe["tenure"] + 1)
    df_fe["products_per_tenure_log"]      = np.log1p(df_fe["products_per_tenure"])

    df_fe["salary_per_product"]           = df_fe["estimated_salary"] / df_fe["products_number"].replace(0, 1)

    # --- Behavior & Engagement Features ---
    df_fe["has_balance"]                  = (df_fe["balance"] > 0).astype(int)
    df_fe["activity_score"]               = df_fe["credit_card"] + df_fe["active_member"]
    df_fe["engagement_density"]           = df_fe["activity_score"] / (df_fe["tenure"] + 1)
    df_fe["tenure_relative_to_age"]       = df_fe["age"] - df_fe["tenure"]
    df_fe["product_intensity_score"]      = df_fe["products_number"] * df_fe["balance"]
    df_fe["engagement_level"]             = df_fe[["has_balance", "credit_card", "active_member"]].sum(axis=1)

    # --- Interaksi Fitur ---
    df_fe["age_x_product"]                = df_fe["age"] * df_fe["products_number"]
    df_fe["net_value_ratio"]              = (df_fe["estimated_salary"] - df_fe["balance"]) / (df_fe["estimated_salary"] + 1)
    df_fe["engage_x_balprod"]             = df_fe["engagement_density"] * df_fe["balance_per_product_log"]
    df_fe["age_x_salaryprod"]             = df_fe["age"] * df_fe["salary_per_product"]

    # --- Bucketing ---
    df_fe["tenure_bucket"] = pd.cut(df_fe["tenure"], bins=[-1, 2, 5, 10], labels=["Low", "Mid", "High"])

    def bucket_credit(score):
        if score < 580:
            return "Low"
        elif score < 700:
            return "Mid"
        else:
            return "High"
    df_fe["credit_score_bucket"] = df_fe["credit_score"].apply(bucket_credit)

    return df_fe
