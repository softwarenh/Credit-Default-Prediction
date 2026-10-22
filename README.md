# Credit Risk Scoring & Loan Default Prediction Engine

An end-to-end, production-grade credit risk modeling pipeline designed to assess loan default probabilities, optimize risk-adjusted lending decisions, and deliver regulatory transparency. 

Rather than treating credit risk purely as a binary classification problem, this framework mirrors real-world underwriting systems—integrating domain-specific data sanitation, leakage-free pipeline architectures, institutional evaluation metrics (**Gini Coefficient**, **Kolmogorov-Smirnov Statistic**, **PR-AUC**), and post-hoc model explainability via **SHAP**.

---

## 📌 Executive Summary

* **Achieved an ROC-AUC of 0.9536 and a KS Statistic of 0.7864** on out-of-sample test data by implementing a leakage-free preprocessing pipeline, handling class imbalance (~22% baseline default rate), and optimizing hyperparameters across candidate models using 5-fold cross-validation.
* **Eliminated data leakage risks** by encapsulating feature scaling, median/constant imputations, and categorical encodings within isolated `scikit-learn` `Pipeline` and `ColumnTransformer` modules.
* **Established decision threshold optimization** by shifting away from the naive $0.50$ cutoff to an empirical **$0.1911$ threshold** (derived via Youden’s J statistic), significantly boosting default recall without compromising precision.
* **Ensured model explainability for regulatory compliance** by leveraging SHAP (SHapley Additive exPlanations) for both portfolio-wide global feature analysis and local individual risk audits (waterfall plots).

---

## 📈 Model Benchmarking & Performance

Three candidate architectures were trained and evaluated on stratified 5-fold cross-validation folds. **XGBoost** outperformed Logistic Regression and Random Forest across all primary discrimination and separation metrics.

| Model | CV PR-AUC | CV ROC-AUC | CV Gini Coefficient | CV KS Statistic | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Champion)** | **0.9144** | **0.9513** | **0.9025** | **0.7701** | **Selected** |
| **Random Forest** | 0.8969 | 0.9361 | 0.8722 | 0.7474 | Benchmark |
| **Logistic Regression** | 0.7186 | 0.8682 | 0.7364 | 0.5837 | Baseline |

### Final Out-of-Sample Audit (XGBoost Test Set)

* **PR-AUC**: `0.9183`
* **ROC-AUC**: `0.9536`
* **Gini Coefficient ($2 \times \text{AUC} - 1$)**: `0.9072`
* **Kolmogorov-Smirnov (KS) Statistic**: `0.7864` *(Demonstrates strong distributional separation between default and non-default cohorts)*
* **Optimal Decision Threshold**: `0.1911`

---

## 🛠️ Pipeline Architecture & Methodology
### 1. Data Sanitation & Logical Constraints
* Filtered out severe data anomalies, including records with unrealistic borrower ages ($>100$) or impossible employment lengths ($\ge \text{age} - 18$).

### 2. Leakage-Free Preprocessing Architecture
* **Numeric Features**: Imputed via median strategy and standardized using `StandardScaler`.
* **Ordinal Features**: Mapped loan risk grades (`A` through `G`) preserving inherent ordinal hierarchy via `OrdinalEncoder`.
* **Nominal Categorical Features**: Encoded via `OneHotEncoder(handle_unknown='ignore')` to handle unseen categories during inference safely.

### 3. Comprehensive Risk Dashboard
Integrated a custom evaluation framework (`evaluate()`) generating a 2x2 analytical dashboard:
* **ROC Curve & Gini**: Tracks discrimination capacity across all operating cutoffs.
* **Precision-Recall Curve**: Evaluates precision retention on the minority class under high-recall scenarios.
* **KS Separation Plot**: Measures maximum vertical separation between cumulative good vs. bad distributions.
* **Confusion Matrix**: Evaluates performance at the optimized decision threshold.

### link to Data used:
https://www.kaggle.com/datasets/laotse/credit-risk-dataset
