"""
predict.py
==========
ChurnZero 26 — Banking Customer Churn Prediction
Load saved model and generate predictions on any test CSV

Team:
  - Kesani Sree Sai Charan Teja
  - Nagi Reddy SaiBaba
  - Jagriti Sharma

Usage:
    python src/predict.py --test data/test.csv
    python src/predict.py --test data/test.csv --threshold 0.40
"""

import argparse
import pickle
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from preprocess import engineer_features, prepare_matrices, scale_features, load_data


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test",      default="data/test.csv",             help="Path to test CSV")
    p.add_argument("--model",     default="models/stacking_model.pkl", help="Trained model path")
    p.add_argument("--scaler",    default="models/scaler.pkl",         help="Fitted scaler path")
    p.add_argument("--features",  default="models/feature_names.pkl",  help="Feature names path")
    p.add_argument("--threshold", type=float, default=0.40,            help="Decision threshold")
    p.add_argument("--output",    default="outputs/predictions.csv",   help="Output CSV path")
    return p.parse_args()


def assign_risk_tier(prob: float) -> str:
    if prob >= 0.75:   return "Tier 1 — Critical Risk"
    elif prob >= 0.50: return "Tier 2 — High Risk"
    elif prob >= 0.25: return "Tier 3 — Medium Risk"
    else:              return "Tier 4 — Low Risk"


def main():
    args = parse_args()

    # ── Load model artefacts ─────────────────
    print("[1] Loading model artefacts...")
    for path in [args.model, args.scaler, args.features]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Not found: {path}  — run train.py first!")

    with open(args.model,    "rb") as f: model         = pickle.load(f)
    with open(args.scaler,   "rb") as f: scaler        = pickle.load(f)
    with open(args.features, "rb") as f: feature_names = pickle.load(f)
    print("✅ Model loaded.")

    # ── Load & preprocess test data ──────────
    print("\n[2] Loading & preprocessing test data...")
    # Load without a target column (pass a dummy train to satisfy prepare_matrices)
    test_df = pd.read_csv(args.test)
    test_df = engineer_features(test_df)

    # Encode — pass test as both train & test (no leakage since we're only transforming)
    from preprocess import encode_features, DROP_COLS, FEATURE_COLS, GEO_GENDER_COLS
    import pandas as pd as pd2
    _, test_enc = encode_features(test_df.copy(), test_df.copy())
    test_enc = test_enc.copy()

    all_features = FEATURE_COLS + [c for c in GEO_GENDER_COLS if c in test_enc.columns]
    X_test = test_enc[[c for c in all_features if c in test_enc.columns]]

    # Scale using saved scaler
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    # ── Predict ──────────────────────────────
    print("\n[3] Generating predictions...")
    proba = model.predict_proba(X_test_sc)[:, 1]
    preds = (proba >= args.threshold).astype(int)

    # ── Build output ─────────────────────────
    if "CustomerId" in test_df.columns:
        out = pd.DataFrame({
            "CustomerId":  test_df["CustomerId"].values,
            "churn_proba": np.round(proba, 4),
            "Exited":      preds,
            "risk_tier":   [assign_risk_tier(p) for p in proba],
        })
    else:
        out = pd.DataFrame({
            "id":          range(len(preds)),
            "churn_proba": np.round(proba, 4),
            "Exited":      preds,
            "risk_tier":   [assign_risk_tier(p) for p in proba],
        })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, index=False)

    # ── Summary ──────────────────────────────
    print(f"\n✅ Predictions saved → {args.output}")
    print(f"\nPrediction Summary ({len(out)} customers):")
    print(f"  Predicted Churn    : {preds.sum()}  ({preds.mean()*100:.1f}%)")
    print(f"  Predicted No-Churn : {(1-preds).sum()}  ({(1-preds).mean()*100:.1f}%)")
    print(f"\nRisk Tier Breakdown:")
    print(out["risk_tier"].value_counts().to_string())


if __name__ == "__main__":
    main()
