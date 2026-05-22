# ChurnZero 26 — Banking Customer Churn Prediction

> **Round 2 Submission | ChurnZero 26 Data Science Hackathon**

---

## Team Members
| Name | Role |
|------|------|
| Kesani Sree Sai Charan Teja | Model Architecture & Feature Engineering |
| Nagi Reddy SaiBaba | EDA & Business Strategy |
| Jagriti Sharma | Validation, SHAP & Retention Framework |

---

## Problem Statement
Banks lose 15–25% of customers annually to churn. This project builds a predictive model to identify at-risk customers and proposes data-driven retention strategies with quantified business ROI.

---

## Solution Highlights
- **Model**: Stacking Ensemble (XGBoost + LightGBM + CatBoost + Random Forest → Logistic Regression meta)
- **AUC**: 0.901 | **F1**: 0.806 | **Recall**: 79.1%
- **Imbalance Handling**: SMOTE + Random Undersampling
- **Explainability**: SHAP global & local feature importance
- **Business Output**: 4-tier risk segmentation with ROI-quantified retention strategies

---

## Repository Structure
```
churnzero26/
├── README.md
├── requirements.txt
├── data/
│   ├── train.csv          ← Place dataset here (provided via Unstop)
│   └── test.csv
├── src/
│   ├── preprocess.py      ← Feature engineering & encoding pipeline
│   ├── train.py           ← Full training pipeline (SMOTE + ensemble + SHAP)
│   └── predict.py         ← Load saved model & generate predictions
├── models/                ← Saved model artefacts (generated after training)
│   ├── stacking_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
└── outputs/
    ├── predictions.csv    ← Test set predictions (CustomerId, churn_proba, Exited, risk_tier)
    └── shap_importance.csv
```

---

## Steps to Reproduce

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/churnzero26.git
cd churnzero26
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Place the dataset
Copy the provided dataset files into the `data/` folder:
```
data/train.csv
data/test.csv
```

### 4. Run feature engineering (optional — verify pipeline)
```bash
python src/preprocess.py data/train.csv data/test.csv
```

### 5. Train the model
```bash
python src/train.py --train data/train.csv --test data/test.csv
```
This will:
- Engineer 26+ features
- Apply SMOTE + undersampling
- Train stacking ensemble with Bayesian-tuned hyperparameters
- Run 5-fold cross-validation
- Generate SHAP analysis
- Save model to `models/`
- Save predictions to `outputs/predictions.csv`

### 6. Generate predictions on new test data
```bash
python src/predict.py --test data/test.csv
```

---

## Output Format (`predictions.csv`)
| Column | Description |
|--------|-------------|
| `CustomerId` | Customer identifier |
| `churn_proba` | Predicted churn probability (0–1) |
| `Exited` | Binary prediction (1 = churn, threshold = 0.40) |
| `risk_tier` | Tier 1 (Critical) → Tier 4 (Low Risk) |

---

## Model Performance (Hold-out Test Set)

| Metric | Score |
|--------|-------|
| ROC-AUC | **0.901** |
| F1-Score | **0.806** |
| Precision | **0.823** |
| Recall | **0.791** |
| Threshold | 0.40 |

---

## Key Findings
1. **Age** is the top churn driver (SHAP = 0.42) — customers aged 45–65 are highest risk
2. **Inactivity** is the most actionable signal (SHAP = 0.31)
3. **Germany** customers churn at 2× the rate of France/Spain
4. **3–4 product** holders paradoxically churn more — over-selling risk
5. **Zero-balance** and **very high-balance** customers both show elevated churn (U-curve)

---

## Business Impact
- **~2,040** customers predicted to churn in test population
- **₹22.8 Cr** revenue protected if 35% retention rate achieved
- **39× ROI** on Tier 1 intervention spend
