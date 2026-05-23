"""
preprocess.py
=============
ChurnZero 26 — Banking Customer Churn Prediction
Feature Engineering & Preprocessing Pipeline

Team:
  - Kesani Sree Sai Charan Teja
  - Nagi Reddy SaiBaba
  - Jagriti Sharma
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings("ignore")

TARGET = "churn"
ID_COL = "customer_id"

CAT_COLS = [
    'gender','marital_status','education_level','occupation_type',
    'income_band','income_category','city_tier','region','customer_segment',
    'onboarding_channel','relationship_type','primary_account_type',
    'card_category','competitor_bank_offer_awareness','customer_feedback_sentiment'
]

# ─────────────────────────────────────────────
def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    print(f"Train : {train.shape}  |  Churn rate: {train[TARGET].mean()*100:.1f}%")
    print(f"Test  : {test.shape}")
    return train, test

# ─────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()

    # Fill missing app_rating_given with median
    df['app_rating_given'] = df['app_rating_given'].fillna(df['app_rating_given'].median())

    # --- Ratio & interaction features ---
    df['balance_income_ratio']        = df['avg_monthly_balance'] / (df['annual_income'] + 1)
    df['txn_per_month']               = df['total_trans_count']   / (df['tenure_months'] + 1)
    df['txn_amt_per_month']           = df['total_trans_amt']     / (df['tenure_months'] + 1)
    df['digital_ratio']               = df['digital_transaction_ratio']
    df['spend_to_limit_ratio']        = df['credit_card_spend']   / (df['credit_card_limit'] + 1)
    df['clv_per_tenure']              = df['customer_lifetime_value'] / (df['tenure_months'] + 1)
    df['complaint_resolution_ratio']  = df['unresolved_complaint_count'] / (df['total_complaints'] + 1)
    df['campaign_response_rate']      = df['campaign_response_count'] / (df['campaign_received_count'] + 1)
    df['emi_stress_ratio']            = df['emi_amount'] / (df['annual_income'] + 1)
    df['balance_decline_flag']        = (df['balance_decline_percentage'] > 20).astype(int)
    df['high_inactivity_flag']        = (df['account_inactive_days'] > 60).astype(int)
    df['high_complaint_flag']         = (df['total_complaints'] > 3).astype(int)
    df['low_satisfaction_flag']       = (df['satisfaction_score'] <= 2).astype(int)
    df['low_nps_flag']                = (df['nps_score'] <= 3).astype(int)
    df['digital_inactive_flag']       = (df['mobile_app_login_count'] == 0).astype(int)
    df['retention_offer_rejected']    = ((df['retention_offer_received'] == 1) &
                                          (df['retention_offer_accepted'] == 0)).astype(int)
    df['multi_loan_flag']             = (df['personal_loan_flag'] + df['home_loan_flag'] +
                                          df['auto_loan_flag'] >= 2).astype(int)
    df['products_count']              = (df['savings_account_flag'] + df['current_account_flag'] +
                                          df['credit_card_flag'] + df['fixed_deposit_flag'] +
                                          df['investment_product_flag'] + df['insurance_product_flag'] +
                                          df['demat_account_flag'])
    df['escalation_rate']             = df['escalation_count'] / (df['total_complaints'] + 1)
    df['login_days_ratio']            = df['last_login_days'] / (df['tenure_months'] * 30 + 1)
    df['competitor_aware_flag']       = (df['competitor_bank_offer_awareness'].isin(['High','Medium'])).astype(int)
    df['negative_sentiment_flag']     = (df['customer_feedback_sentiment'] == 'Negative').astype(int)
    df['log_clv']                     = np.log1p(df['customer_lifetime_value'])
    df['log_balance']                 = np.log1p(df['avg_monthly_balance'])
    df['log_income']                  = np.log1p(df['annual_income'])
    df['age_squared']                 = df['age'] ** 2

    return df

# ─────────────────────────────────────────────
def encode_and_prepare(train, test):
    combined = pd.concat([train, test], axis=0, ignore_index=True)

    # One-hot encode categoricals
    combined = pd.get_dummies(combined, columns=CAT_COLS, drop_first=False)

    n_train = len(train)
    train_enc = combined.iloc[:n_train].copy()
    test_enc  = combined.iloc[n_train:].copy()
    return train_enc, test_enc

# ─────────────────────────────────────────────
def prepare_matrices(train, test):
    drop_cols = [ID_COL, TARGET] if TARGET in train.columns else [ID_COL]
    feature_cols = [c for c in train.columns if c not in drop_cols]

    X_train = train[feature_cols]
    y_train = train[TARGET] if TARGET in train.columns else None
    X_test  = test[[c for c in feature_cols if c in test.columns]]

    return X_train, y_train, X_test, feature_cols

# ─────────────────────────────────────────────
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_tr_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_te_sc = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns,  index=X_test.index)
    return X_tr_sc, X_te_sc, scaler

# ─────────────────────────────────────────────
def full_pipeline(train_path, test_path):
    train, test = load_data(train_path, test_path)
    train = engineer_features(train)
    test  = engineer_features(test)
    train_enc, test_enc = encode_and_prepare(train, test)
    X_train, y_train, X_test, feat_cols = prepare_matrices(train_enc, test_enc)
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)
    return X_train_sc, y_train, X_test_sc, feat_cols, scaler, test

if __name__ == "__main__":
    import sys
    train_df, test_df = load_data(sys.argv[1], sys.argv[2])
    train_df = engineer_features(train_df)
    test_df  = engineer_features(test_df)
    tr, te   = encode_and_prepare(train_df, test_df)
    X_tr, y_tr, X_te, feats = prepare_matrices(tr, te)
    print(f"\n✅ X_train: {X_tr.shape}  X_test: {X_te.shape}")
    print(f"Features : {len(feats)}")
