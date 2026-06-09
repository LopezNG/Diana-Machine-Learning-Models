<!--
DIANA Thesis — Data Analysis Appendices
Machine learning-based predictive screening tool for Type 2 diabetes risk in menopausal women
Cohort size: n = 1,376
Prepared as structured placeholder-ready appendix material for thesis reporting.
-->

# Data Analysis Appendices

## Appendix A — Dataset Overview

### A.1 — Dataset Summary Statistics

This table summarizes the clinical, metabolic, behavioral, and demographic variables used in the DIANA analysis pipeline. Values are realistic placeholders for a menopausal women cohort and should be replaced with final descriptive statistics after dataset lock.

| Feature | Type | Min | Max | Mean | Std Dev | Missing (%) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Age | Continuous | 45.0 | 79.0 | 58.7 | 7.4 | 0.0 |
| BMI | Continuous | 18.4 | 42.8 | 28.6 | 5.1 | 1.2 |
| Fasting Glucose | Continuous | 68.0 | 198.0 | 104.8 | 21.6 | 2.1 |
| HbA1c | Continuous | 4.6 | 9.4 | 5.9 | 0.8 | 1.8 |
| Waist Circumference | Continuous | 68.0 | 128.0 | 91.7 | 11.9 | 1.5 |
| Systolic BP | Continuous | 92.0 | 186.0 | 132.4 | 17.2 | 0.9 |
| Diastolic BP | Continuous | 56.0 | 108.0 | 79.6 | 9.8 | 0.9 |
| Triglycerides | Continuous | 48.0 | 386.0 | 151.3 | 67.5 | 2.7 |
| HDL Cholesterol | Continuous | 28.0 | 94.0 | 52.6 | 13.1 | 2.4 |
| LDL Cholesterol | Continuous | 54.0 | 218.0 | 126.9 | 32.8 | 2.6 |
| Insulin Level | Continuous | 2.8 | 39.5 | 12.7 | 6.4 | 4.3 |
| Menopause Duration | Continuous | 0.5 | 31.0 | 8.9 | 6.7 | 0.7 |
| Physical Activity Score | Ordinal | 0.0 | 10.0 | 4.8 | 2.1 | 3.0 |
| Family History of T2DM | Binary | 0.0 | 1.0 | 0.36 | 0.48 | 0.4 |
| Smoking Status | Categorical | 0.0 | 2.0 | 0.41 | 0.66 | 1.1 |

## Appendix B — Feature Selection Results

### B.1 — Information Gain and Entropy Table

This table ranks candidate predictors by information gain to document the supervised feature selection phase. Features meeting the predefined information gain threshold were retained for downstream model development.

| Rank | Feature | Entropy (H) | Information Gain (IG) | IG Ratio | Selected |
| ---: | :--- | ---: | ---: | ---: | :---: |
| 1 | HbA1c | 0.884 | 0.326 | 0.447 | ✅ |
| 2 | Fasting Glucose | 0.872 | 0.312 | 0.421 | ✅ |
| 3 | BMI | 0.791 | 0.241 | 0.336 | ✅ |
| 4 | Waist Circumference | 0.813 | 0.218 | 0.304 | ✅ |
| 5 | Insulin Level | 0.846 | 0.184 | 0.258 | ✅ |
| 6 | Triglycerides | 0.758 | 0.157 | 0.226 | ✅ |
| 7 | Family History of T2DM | 0.692 | 0.143 | 0.211 | ✅ |
| 8 | HDL Cholesterol | 0.736 | 0.128 | 0.194 | ✅ |
| 9 | Age | 0.804 | 0.116 | 0.167 | ✅ |
| 10 | Systolic BP | 0.779 | 0.104 | 0.151 | ✅ |
| 11 | LDL Cholesterol | 0.721 | 0.092 | 0.138 | ❌ |
| 12 | Menopause Duration | 0.698 | 0.081 | 0.119 | ❌ |
| 13 | Diastolic BP | 0.742 | 0.074 | 0.107 | ❌ |
| 14 | Physical Activity Score | 0.689 | 0.063 | 0.096 | ❌ |
| 15 | Smoking Status | 0.571 | 0.038 | 0.061 | ❌ |

> **Note:** Features were retained when Information Gain (IG) >= 0.10. This criterion retained 10 of 15 candidate predictors for subsequent modeling.

### B.2 — Correlation Matrix Summary (Top 10 Features)

This matrix summarizes pairwise Pearson correlations among the top 10 selected predictors. It was used to inspect collinearity and identify highly redundant variables before model training.

| Feature | HbA1c | Fasting Glucose | BMI | Waist Circumference | Insulin Level | Triglycerides | Family History of T2DM | HDL Cholesterol | Age | Systolic BP |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HbA1c | 1.000 | 0.642 | 0.331 | 0.358 | 0.284 | 0.246 | 0.218 | -0.191 | 0.226 | 0.203 |
| Fasting Glucose | 0.642 | 1.000 | 0.297 | 0.326 | 0.312 | 0.221 | 0.202 | -0.174 | 0.214 | 0.187 |
| BMI | 0.331 | 0.297 | 1.000 | 0.714 | 0.476 | 0.352 | 0.126 | -0.334 | 0.098 | 0.289 |
| Waist Circumference | 0.358 | 0.326 | 0.714 | 1.000 | 0.438 | 0.381 | 0.139 | -0.306 | 0.157 | 0.317 |
| Insulin Level | 0.284 | 0.312 | 0.476 | 0.438 | 1.000 | 0.295 | 0.112 | -0.242 | 0.071 | 0.176 |
| Triglycerides | 0.246 | 0.221 | 0.352 | 0.381 | 0.295 | 1.000 | 0.084 | -0.421 | 0.108 | 0.219 |
| Family History of T2DM | 0.218 | 0.202 | 0.126 | 0.139 | 0.112 | 0.084 | 1.000 | -0.056 | 0.067 | 0.044 |
| HDL Cholesterol | -0.191 | -0.174 | -0.334 | -0.306 | -0.242 | -0.421 | -0.056 | 1.000 | -0.039 | -0.166 |
| Age | 0.226 | 0.214 | 0.098 | 0.157 | 0.071 | 0.108 | 0.067 | -0.039 | 1.000 | 0.274 |
| Systolic BP | 0.203 | 0.187 | 0.289 | 0.317 | 0.176 | 0.219 | 0.044 | -0.166 | 0.274 | 1.000 |

## Appendix C — Model Training and Evaluation

### C.1 — Cross-Validation Results per Fold (5-Fold Stratified CV)

This table reports fold-level performance from stratified cross-validation for the selected predictive model. The summary row provides an estimate of model stability across validation folds.

| Fold | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Fold 1 | 0.704 | 0.672 | 0.641 | 0.656 | 0.728 |
| Fold 2 | 0.716 | 0.689 | 0.658 | 0.673 | 0.739 |
| Fold 3 | 0.711 | 0.681 | 0.666 | 0.673 | 0.737 |
| Fold 4 | 0.722 | 0.695 | 0.671 | 0.683 | 0.746 |
| Fold 5 | 0.709 | 0.678 | 0.649 | 0.663 | 0.733 |
| Mean ± SD | 0.712 ± 0.007 | 0.683 ± 0.009 | 0.657 ± 0.012 | 0.670 ± 0.010 | 0.7366 ± 0.0067 |

### C.2 — Model Comparison Table

This table compares candidate supervised learning models evaluated during DIANA development. The selected model balances discrimination, interpretability, calibration potential, and computational efficiency.

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC | Training Time (s) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.694 | 0.661 | 0.628 | 0.644 | 0.711 | 1.8 |
| Decision Tree | 0.671 | 0.634 | 0.607 | 0.620 | 0.659 | 0.9 |
| Random Forest | 0.708 | 0.676 | 0.651 | 0.663 | 0.729 | 8.6 |
| Gradient Boosting ⭐ | 0.712 | 0.683 | 0.657 | 0.670 | 0.7366 | 10.4 |
| XGBoost | 0.710 | 0.679 | 0.654 | 0.666 | 0.734 | 12.7 |
| SVM (RBF kernel) | 0.699 | 0.667 | 0.633 | 0.650 | 0.718 | 15.9 |

### C.3 — Hyperparameter Tuning Results (Selected Model)

This table documents the hyperparameter search space and optimal values for the selected Gradient Boosting model. These settings should be updated with final tuning outputs from the locked analysis script.

| Parameter | Search Range | Optimal Value |
| :--- | :--- | :--- |
| n_estimators | 50, 100, 150, 200, 300 | 150 |
| learning_rate | 0.01, 0.03, 0.05, 0.10, 0.20 | 0.05 |
| max_depth | 2, 3, 4, 5 | 3 |
| min_samples_split | 2, 5, 10, 20 | 10 |
| min_samples_leaf | 1, 2, 4, 8 | 4 |
| subsample | 0.60, 0.70, 0.80, 0.90, 1.00 | 0.80 |

## Appendix D — Threshold Optimization

### D.1 — Threshold Sensitivity Table

This table evaluates classification performance across candidate probability thresholds for the selected model. Threshold optimization supports the screening objective by balancing sensitivity with specificity and positive predictive value.

| Threshold | Sensitivity (Recall) | Specificity | PPV (Precision) | NPV | F1 Score | Youden's J |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.30 | 0.812 | 0.472 | 0.561 | 0.752 | 0.664 | 0.284 |
| 0.35 | 0.774 | 0.536 | 0.589 | 0.735 | 0.669 | 0.310 |
| 0.40 | 0.728 | 0.604 | 0.626 | 0.712 | 0.673 | 0.332 |
| **0.45 ←** | **0.684** | **0.668** | **0.681** | **0.671** | **0.682** | **0.352** |
| 0.50 | 0.657 | 0.696 | 0.683 | 0.671 | 0.670 | 0.353 |
| 0.55 | 0.611 | 0.741 | 0.704 | 0.655 | 0.654 | 0.352 |
| 0.60 | 0.558 | 0.781 | 0.719 | 0.633 | 0.628 | 0.339 |
| 0.65 | 0.503 | 0.824 | 0.742 | 0.612 | 0.600 | 0.327 |
| 0.70 | 0.446 | 0.861 | 0.761 | 0.592 | 0.562 | 0.307 |

> **Note:** The provisional optimal threshold was 0.45, selected to prioritize screening sensitivity while preserving the highest combined operating performance near the peak Youden's J.

## Appendix E — Risk Stratification and Clustering

### E.1 — Cluster Profile Summary

This table profiles the three patient subgroups identified during unsupervised risk stratification. Cluster summaries support clinical interpretation of DIANA risk categories and downstream decision-support thresholds.

| Cluster | Label | n | % of Sample | Mean Age | Mean BMI | Mean HbA1c | Mean Fasting Glucose | T2DM Risk (%) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cluster 1 | Low Risk | 521 | 37.9 | 55.8 | 24.9 | 5.4 | 91.6 | 12.8 |
| Cluster 2 | Moderate Risk | 604 | 43.9 | 58.9 | 28.7 | 5.9 | 104.3 | 31.6 |
| Cluster 3 | High Risk | 251 | 18.2 | 63.4 | 33.1 | 6.7 | 128.5 | 58.9 |

### E.2 — Cluster Validation Metrics

This table reports internal validation metrics used to evaluate the clustering solution. The selected three-cluster structure was assessed for compactness, separation, and consistency with the elbow method.

| Metric | Value |
| :--- | ---: |
| Silhouette Score | 0.421 |
| Davies-Bouldin Index | 0.873 |
| Calinski-Harabasz Index | 286.4 |
| Optimal k (Elbow Method) | 3 |

> **Note:** The provisional cluster cutoff was based on k = 3 from the elbow method, supported by acceptable separation in the Silhouette Score and Davies-Bouldin Index.
