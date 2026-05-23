"""
train.py
========
ChurnZero 26 — Banking Customer Churn Prediction
Full Training Pipeline: SMOTE + Stacking Ensemble + SHAP

Team:
  - Kesani Sree Sai Charan Teja
  - Nagi Reddy SaiBaba
  - Jagriti Sharma

Usage:
    python src/train.py --train data/train.csv --test data/test.csv
"""

import argparse, os, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection      import StratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model         import LogisticRegression
from sklearn.ensemble             import RandomForestClassifier, StackingClassifier
from sklearn.metrics              import (roc_auc_score, f1_score, precision_score,
                                          recall_score, classification_report, confusion_matrix)
from xgboost    import XGBClassifier
from lightgbm   import LGBMClassifier
from catboost   import CatBoostClassifier
from imblearn.over_sampling   import SMOTE
from imblearn.under_sampling  import RandomUnderSampler
from imblearn.pipeline        import Pipeline as ImbPipeline
import shap

from preprocess import full_pipeline

warnings.filterwarnings("ignore")
os.makedirs("models",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train",     default="data/train.csv")
    p.add_argument("--test",      default="data/test.csv")
    p.add_argument("--threshold", type=float, default=0.40)
    p.add_argument("--seed",      type=int,   default=42)
    return p.parse_args()

def apply_smote(X, y, seed=42):
    print("\n[SMOTE] Before:", dict(pd.Series(y).value_counts()))
    pipe = ImbPipeline([
        ("smote", SMOTE(sampling_strategy=0.6, random_state=seed, k_neighbors=5)),
        ("under", RandomUnderSampler(sampling_strategy=0.9, random_state=seed))
    ])
    X_res, y_res = pipe.fit_resample(X, y)
    print("[SMOTE] After :", dict(pd.Series(y_res).value_counts()))
    return X_res, y_res

def get_base_models(seed=42):
    xgb = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.75, min_child_weight=3,
        gamma=0.1, reg_alpha=0.05, reg_lambda=1.2,
        scale_pos_weight=5.2, eval_metric="auc",
        random_state=seed, n_jobs=-1, verbosity=0
    )
    lgbm = LGBMClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.75, min_child_samples=20,
        reg_alpha=0.05, reg_lambda=1.2, class_weight="balanced",
        random_state=seed, n_jobs=-1, verbose=-1
    )
    cat = CatBoostClassifier(
        iterations=400, depth=6, learning_rate=0.05,
        l2_leaf_reg=3, auto_class_weights="Balanced",
        random_seed=seed, verbose=0
    )
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=4,
        class_weight="balanced", random_state=seed, n_jobs=-1
    )
    return [("xgb", xgb), ("lgbm", lgbm), ("cat", cat), ("rf", rf)]

def assign_tier(p):
    if p >= 0.75:   return "Tier 1 — Critical Risk"
    elif p >= 0.50: return "Tier 2 — High Risk"
    elif p >= 0.25: return "Tier 3 — Medium Risk"
    else:           return "Tier 4 — Low Risk"

def main():
    args = parse_args()
    SEED = args.seed

    print("\n[1] Loading & preprocessing data...")
    X_train_sc, y_train, X_test_sc, feat_cols, scaler, test_df = \
        full_pipeline(args.train, args.test)

    print(f"     Features  : {X_train_sc.shape[1]}")

    print("\n[2] Train / validation split (85/15)...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_sc, y_train, test_size=0.15, stratify=y_train, random_state=SEED
    )

    print("\n[3] Applying SMOTE + Undersampling...")
    X_res, y_res = apply_smote(X_tr, y_tr, seed=SEED)

    print("\n[4] Building Stacking Ensemble...")
    base_models  = get_base_models(seed=SEED)
    meta_learner = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    stack = StackingClassifier(
        estimators=base_models, final_estimator=meta_learner,
        cv=5, stack_method="predict_proba", passthrough=False, n_jobs=-1
    )

    print("[4] Training... (this takes ~3-5 minutes)")
    stack.fit(X_res, y_res)
    print("✅ Model trained!")

    print("\n[5] 5-Fold Cross-Validation...")
    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(stack, X_train_sc, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)
    print(f"     Fold AUCs : {np.round(scores,4)}")
    print(f"     Mean AUC  : {scores.mean():.4f} ± {scores.std():.4f}")

    print("\n[6] Hold-out Evaluation (threshold={})...".format(args.threshold))
    proba_val = stack.predict_proba(X_val)[:,1]
    preds_val = (proba_val >= args.threshold).astype(int)
    auc  = roc_auc_score(y_val, proba_val)
    f1   = f1_score(y_val, preds_val)
    prec = precision_score(y_val, preds_val)
    rec  = recall_score(y_val, preds_val)
    print(f"\n{'='*45}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"{'='*45}")
    print(classification_report(y_val, preds_val, target_names=["No Churn","Churn"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, preds_val))

    print("\n[7] SHAP Feature Importance...")
    try:
        xgb_sub  = stack.named_estimators_["xgb"]
        sample   = pd.DataFrame(X_res[:500], columns=feat_cols)
        explainer = shap.TreeExplainer(xgb_sub)
        shap_vals = explainer.shap_values(sample)
        shap_imp  = pd.Series(np.abs(shap_vals).mean(axis=0), index=feat_cols).sort_values(ascending=False)
        print("Top 15 features:")
        print(shap_imp.head(15).round(4).to_string())
        shap_imp.to_csv("outputs/shap_importance.csv", header=["mean_abs_shap"])
        print("✅ SHAP saved → outputs/shap_importance.csv")
    except Exception as e:
        print(f"SHAP skipped: {e}")

    print("\n[8] Saving model artefacts...")
    with open("models/stacking_model.pkl", "wb") as f: pickle.dump(stack, f)
    with open("models/scaler.pkl",         "wb") as f: pickle.dump(scaler, f)
    with open("models/feature_names.pkl",  "wb") as f: pickle.dump(feat_cols, f)
    print("✅ Saved → models/")

    print("\n[9] Generating test predictions...")
    proba_test = stack.predict_proba(X_test_sc)[:,1]
    preds_test = (proba_test >= args.threshold).astype(int)

    out = pd.DataFrame({
        "customer_id" : test_df["customer_id"].values,
        "churn_proba" : np.round(proba_test, 4),
        "churn"       : preds_test,
        "risk_tier"   : [assign_tier(p) for p in proba_test],
    })
    out.to_csv("outputs/predictions.csv", index=False)
    print(f"✅ Predictions saved → outputs/predictions.csv ({len(out)} rows)")

    print(f"\n{'='*45}")
    print("  FINAL SUMMARY")
    print(f"{'='*45}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"\n  Risk Tier Distribution:")
    print(out["risk_tier"].value_counts().to_string())
    print(f"{'='*45}")

if __name__ == "__main__":
    main()
