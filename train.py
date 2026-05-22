"""
train.py
========
ChurnZero 26 — Banking Customer Churn Prediction
Full Training Pipeline: SMOTE + Ensemble + Hyperparameter Tuning

Team:
  - Kesani Sree Sai Charan Teja
  - Nagi Reddy SaiBaba
  - Jagriti Sharma

Usage:
    python src/train.py --train data/train.csv --test data/test.csv
"""

import argparse
import os
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection       import StratifiedKFold, cross_val_score
from sklearn.linear_model          import LogisticRegression
from sklearn.ensemble              import RandomForestClassifier, StackingClassifier
from sklearn.metrics               import (roc_auc_score, f1_score, precision_score,
                                           recall_score, classification_report,
                                           confusion_matrix)
from xgboost                       import XGBClassifier
from lightgbm                      import LGBMClassifier
from catboost                      import CatBoostClassifier
from imblearn.over_sampling        import SMOTE
from imblearn.under_sampling       import RandomUnderSampler
from imblearn.pipeline             import Pipeline as ImbPipeline
import shap

from preprocess import (load_data, engineer_features, prepare_matrices,
                        scale_features)

warnings.filterwarnings("ignore")
os.makedirs("models",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ─────────────────────────────────────────────
# 1. ARGUMENTS
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/train.csv",  help="Path to train CSV")
    p.add_argument("--test",  default="data/test.csv",   help="Path to test CSV")
    p.add_argument("--threshold", type=float, default=0.40,
                   help="Decision threshold (default 0.40 — optimised for recall)")
    p.add_argument("--seed",  type=int, default=42)
    return p.parse_args()


# ─────────────────────────────────────────────
# 2. SMOTE + UNDERSAMPLING
# ─────────────────────────────────────────────
def apply_smote(X_train, y_train, seed=42):
    print("\n[SMOTE] Class distribution before:")
    print(y_train.value_counts())

    smote    = SMOTE(sampling_strategy=0.7, random_state=seed, k_neighbors=5)
    undersam = RandomUnderSampler(sampling_strategy=0.9, random_state=seed)

    pipe = ImbPipeline([("smote", smote), ("under", undersam)])
    X_res, y_res = pipe.fit_resample(X_train, y_train)

    print("[SMOTE] Class distribution after:")
    print(pd.Series(y_res).value_counts())
    return X_res, y_res


# ─────────────────────────────────────────────
# 3. BEST HYPERPARAMETERS
#    (found via Bayesian Optimisation on 5-fold CV)
# ─────────────────────────────────────────────
def get_base_models(seed=42):
    xgb = XGBClassifier(
        n_estimators       = 400,
        max_depth          = 6,
        learning_rate      = 0.05,
        subsample          = 0.85,
        colsample_bytree   = 0.80,
        min_child_weight   = 3,
        gamma              = 0.10,
        reg_alpha          = 0.05,
        reg_lambda         = 1.20,
        scale_pos_weight   = 3.90,
        use_label_encoder  = False,
        eval_metric        = "auc",
        random_state       = seed,
        n_jobs             = -1,
        verbosity          = 0,
    )

    lgbm = LGBMClassifier(
        n_estimators       = 400,
        max_depth          = 6,
        learning_rate      = 0.05,
        subsample          = 0.85,
        colsample_bytree   = 0.80,
        min_child_samples  = 20,
        reg_alpha          = 0.05,
        reg_lambda         = 1.20,
        class_weight       = "balanced",
        random_state       = seed,
        n_jobs             = -1,
        verbose            = -1,
    )

    cat = CatBoostClassifier(
        iterations         = 400,
        depth              = 6,
        learning_rate      = 0.05,
        l2_leaf_reg        = 3,
        auto_class_weights = "Balanced",
        random_seed        = seed,
        verbose            = 0,
    )

    rf = RandomForestClassifier(
        n_estimators       = 300,
        max_depth          = 10,
        min_samples_leaf   = 4,
        class_weight       = "balanced",
        random_state       = seed,
        n_jobs             = -1,
    )

    return [("xgb", xgb), ("lgbm", lgbm), ("cat", cat), ("rf", rf)]


# ─────────────────────────────────────────────
# 4. STACKING ENSEMBLE
# ─────────────────────────────────────────────
def build_stacking(base_models, seed=42):
    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    stack = StackingClassifier(
        estimators    = base_models,
        final_estimator = meta,
        cv            = 5,
        stack_method  = "predict_proba",
        passthrough   = False,
        n_jobs        = -1,
    )
    return stack


# ─────────────────────────────────────────────
# 5. CROSS-VALIDATION
# ─────────────────────────────────────────────
def cross_validate_model(model, X, y, seed=42):
    print("\n[CV] Running 5-fold Stratified Cross-Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    print(f"[CV] Fold AUCs : {np.round(scores, 4)}")
    print(f"[CV] Mean AUC  : {scores.mean():.4f}  ±  {scores.std():.4f}")
    return scores


# ─────────────────────────────────────────────
# 6. EVALUATE ON HOLD-OUT
# ─────────────────────────────────────────────
def evaluate(model, X_val, y_val, threshold=0.40):
    proba = model.predict_proba(X_val)[:, 1]
    preds = (proba >= threshold).astype(int)

    auc  = roc_auc_score(y_val, proba)
    f1   = f1_score(y_val, preds)
    prec = precision_score(y_val, preds)
    rec  = recall_score(y_val, preds)

    print(f"\n{'='*45}")
    print(f"  Threshold : {threshold}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"{'='*45}")
    print("\nClassification Report:")
    print(classification_report(y_val, preds, target_names=["No Churn","Churn"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, preds))

    return {"auc": auc, "f1": f1, "precision": prec, "recall": rec}


# ─────────────────────────────────────────────
# 7. SHAP ANALYSIS
# ─────────────────────────────────────────────
def shap_analysis(model, X_train, feature_names, n_samples=500):
    print("\n[SHAP] Computing global feature importance...")
    try:
        # Use the XGB sub-model for SHAP (fastest)
        xgb_model = model.named_estimators_["xgb"]
        sample    = X_train[:n_samples]
        explainer  = shap.TreeExplainer(xgb_model)
        shap_vals  = explainer.shap_values(sample)

        mean_abs   = pd.Series(
            np.abs(shap_vals).mean(axis=0),
            index=feature_names
        ).sort_values(ascending=False)

        print("\nTop 10 Features by SHAP Importance:")
        print(mean_abs.head(10).round(4).to_string())

        # Save SHAP summary
        mean_abs.to_csv("outputs/shap_importance.csv", header=["mean_abs_shap"])
        print("✅ SHAP saved → outputs/shap_importance.csv")
    except Exception as e:
        print(f"[SHAP] Skipped: {e}")


# ─────────────────────────────────────────────
# 8. RISK SEGMENTATION
# ─────────────────────────────────────────────
def assign_risk_tier(prob: float) -> str:
    if prob >= 0.75:   return "Tier 1 — Critical Risk"
    elif prob >= 0.50: return "Tier 2 — High Risk"
    elif prob >= 0.25: return "Tier 3 — Medium Risk"
    else:              return "Tier 4 — Low Risk"


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    args = parse_args()
    SEED = args.seed

    # ── Load & engineer ──────────────────────
    print("\n[1] Loading data...")
    train_df, test_df = load_data(args.train, args.test)

    print("\n[2] Engineering features...")
    train_df = engineer_features(train_df)
    test_df  = engineer_features(test_df)

    X_train_full, y_train_full, X_test, feature_names = prepare_matrices(train_df, test_df)
    X_train_sc, X_test_sc, scaler = scale_features(X_train_full, X_test)

    # ── Train / validation split ─────────────
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_sc, y_train_full,
        test_size=0.15, stratify=y_train_full, random_state=SEED
    )

    # ── SMOTE ────────────────────────────────
    print("\n[3] Applying SMOTE + Undersampling...")
    X_res, y_res = apply_smote(X_tr, y_tr, seed=SEED)

    # ── Build & train ensemble ───────────────
    print("\n[4] Training Stacking Ensemble...")
    base_models = get_base_models(seed=SEED)
    stack_model = build_stacking(base_models, seed=SEED)
    stack_model.fit(X_res, y_res)
    print("✅ Model trained!")

    # ── Cross-validate ───────────────────────
    print("\n[5] Cross-Validation on full training data...")
    cross_validate_model(stack_model, X_train_sc, y_train_full, seed=SEED)

    # ── Evaluate on hold-out ─────────────────
    print("\n[6] Hold-out Evaluation...")
    metrics = evaluate(stack_model, X_val, y_val, threshold=args.threshold)

    # ── SHAP ─────────────────────────────────
    print("\n[7] SHAP Analysis...")
    shap_analysis(stack_model, X_res, feature_names)

    # ── Save model ───────────────────────────
    print("\n[8] Saving model...")
    with open("models/stacking_model.pkl",  "wb") as f: pickle.dump(stack_model, f)
    with open("models/scaler.pkl",          "wb") as f: pickle.dump(scaler, f)
    with open("models/feature_names.pkl",   "wb") as f: pickle.dump(feature_names, f)
    print("✅ Model saved → models/stacking_model.pkl")

    # ── Predict on test set ──────────────────
    print("\n[9] Generating test set predictions...")
    test_proba = stack_model.predict_proba(X_test_sc)[:, 1]
    test_preds = (test_proba >= args.threshold).astype(int)

    # Build output dataframe
    if "CustomerId" in test_df.columns:
        out_df = pd.DataFrame({
            "CustomerId":    test_df["CustomerId"].values,
            "churn_proba":   np.round(test_proba, 4),
            "Exited":        test_preds,
            "risk_tier":     [assign_risk_tier(p) for p in test_proba],
        })
    else:
        out_df = pd.DataFrame({
            "id":            range(len(test_preds)),
            "churn_proba":   np.round(test_proba, 4),
            "Exited":        test_preds,
            "risk_tier":     [assign_risk_tier(p) for p in test_proba],
        })

    out_df.to_csv("outputs/predictions.csv", index=False)
    print(f"✅ Predictions saved → outputs/predictions.csv  ({len(out_df)} rows)")

    # ── Summary ──────────────────────────────
    print("\n" + "="*45)
    print("  FINAL RESULTS SUMMARY")
    print("="*45)
    print(f"  ROC-AUC   : {metrics['auc']:.4f}")
    print(f"  F1-Score  : {metrics['f1']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  Threshold : {args.threshold}")
    tier_counts = out_df["risk_tier"].value_counts()
    print("\n  Risk Tier Distribution (Test Set):")
    print(tier_counts.to_string())
    print("="*45)


if __name__ == "__main__":
    main()
