"""
Model-Based Cross-Dataset Imputation for diana_dataset_final.csv

1. Standardize column names across datasets
2. Train Random Forest on diabetes_dataset00.csv to predict family_history_diabetes
3. KNN Imputer (k=5) for continuous variables (triglycerides, systolic, diastolic)
4. Validate physiological ranges
5. Save diana_dataset_imputed.csv
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load datasets ──────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading datasets")
print("=" * 60)

df_main = pd.read_csv("diana_dataset_final.csv")
df00 = pd.read_csv("diabetes_dataset00.csv")
df_pred = pd.read_csv("diabetes_prediction_dataset.csv")

print(f"  Main dataset:       {df_main.shape}")
print(f"  diabetes_dataset00: {df00.shape}")
print(f"  prediction_dataset: {df_pred.shape}")

# ── 2. Standardize column names ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Standardizing column names")
print("=" * 60)

# diabetes_prediction_dataset.csv mappings
df_pred = df_pred.rename(columns={
    "HbA1c_level": "hba1c",
    "blood_glucose_level": "fbs",
})
print("  diabetes_prediction_dataset: HbA1c_level → hba1c, blood_glucose_level → fbs")

# diabetes_dataset00.csv mappings
df00 = df00.rename(columns={
    "Family History": "family_history_diabetes",
    "Blood Glucose Levels": "fbs",
    "Age": "age",
    "BMI": "bmi",
    "Blood Pressure": "blood_pressure",
    "Cholesterol Levels": "cholesterol_levels",
    "Waist Circumference": "waist_circumference",
})
print("  diabetes_dataset00: Family History → family_history_diabetes, Blood Glucose Levels → fbs, etc.")

# ── 3. Impute family_history_diabetes using Random Forest ─────────────────────
print("\n" + "=" * 60)
print("STEP 3: Imputing family_history_diabetes via Random Forest")
print("=" * 60)

# Convert Family History in df00 to binary (Yes=1, No=0)
df00["family_history_diabetes"] = df00["family_history_diabetes"].map({"Yes": 1, "No": 0})

# Common features between df00 and df_main for training
train_features = ["age", "bmi", "fbs"]

# Prepare training data from diabetes_dataset00.csv (no missing in these columns)
X_train = df00[train_features].copy()
y_train = df00["family_history_diabetes"].copy()

print(f"  Training set size: {len(X_train)}")
print(f"  Class distribution: {dict(y_train.value_counts())}")

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

# Cross-validation score on training data
cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="accuracy")
print(f"  Cross-val accuracy (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Fit on full training set
rf.fit(X_train, y_train)
print(f"  Feature importances: {dict(zip(train_features, rf.feature_importances_.round(4)))}")

# Identify rows with missing family_history_diabetes in main dataset
mask_missing_fh = df_main["family_history_diabetes"].isna()
mask_has_features = df_main[train_features].notna().all(axis=1)

# Predict only for rows that have the required features
predict_mask = mask_missing_fh & mask_has_features
print(f"\n  Missing family_history_diabetes: {mask_missing_fh.sum()}")
print(f"  Rows with features available:   {predict_mask.sum()}")

if predict_mask.sum() > 0:
    X_predict = df_main.loc[predict_mask, train_features].copy()
    predictions = rf.predict(X_predict)
    df_main.loc[predict_mask, "family_history_diabetes"] = predictions
    print(f"  Predicted distribution: 0={int((predictions == 0).sum())}, 1={int((predictions == 1).sum())}")

# For any remaining missing (rows without features), use mode from existing data
remaining_missing = df_main["family_history_diabetes"].isna().sum()
if remaining_missing > 0:
    mode_val = df_main["family_history_diabetes"].mode()[0]
    df_main["family_history_diabetes"].fillna(mode_val, inplace=True)
    print(f"  Filled {remaining_missing} remaining rows with mode ({int(mode_val)})")

print(f"  Final missing family_history_diabetes: {df_main['family_history_diabetes'].isna().sum()}")

# ── 4. Impute continuous variables using KNN Imputer (k=5) ───────────────────
print("\n" + "=" * 60)
print("STEP 4: Imputing continuous variables via KNN Imputer (k=5)")
print("=" * 60)

# Continuous columns to impute (target columns + correlated features)
target_impute_cols = ["triglycerides", "systolic", "diastolic"]
# All numeric columns that help the KNN imputer find similar patients
knn_feature_cols = [
    "age", "hba1c", "fbs", "bmi",
    "total_cholesterol", "ldl", "hdl",
    "triglycerides", "systolic", "diastolic",
    "waist_circumference"
]

# Only keep columns that actually exist in df_main
knn_feature_cols = [c for c in knn_feature_cols if c in df_main.columns]

print(f"  KNN features: {knn_feature_cols}")
for col in target_impute_cols:
    print(f"  Missing {col} before: {df_main[col].isna().sum()}")

# Extract the numeric subset for KNN imputation
knn_data = df_main[knn_feature_cols].copy()

# Apply KNN Imputer
knn_imputer = KNNImputer(n_neighbors=5, weights="distance")
knn_imputed = knn_imputer.fit_transform(knn_data)
knn_imputed_df = pd.DataFrame(knn_imputed, columns=knn_feature_cols, index=df_main.index)

# Write back only the target columns (and any other cols that had missingness)
for col in knn_feature_cols:
    missing_mask = df_main[col].isna()
    if missing_mask.any():
        df_main.loc[missing_mask, col] = knn_imputed_df.loc[missing_mask, col]

for col in target_impute_cols:
    print(f"  Missing {col} after:  {df_main[col].isna().sum()}")

# ── 5. Validate physiological ranges ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Validating physiological ranges")
print("=" * 60)

# Define realistic ranges
physio_ranges = {
    "systolic":       (60, 250),
    "diastolic":      (30, 160),
    "triglycerides":  (20, 1000),
    "bmi":            (10, 80),
    "hba1c":          (3.0, 20.0),
    "fbs":            (30, 600),
    "total_cholesterol": (50, 500),
    "ldl":            (10, 400),
    "hdl":            (10, 150),
    "waist_circumference": (40, 200),
}

clipped_count = 0
for col, (lo, hi) in physio_ranges.items():
    if col in df_main.columns:
        below = (df_main[col] < lo).sum()
        above = (df_main[col] > hi).sum()
        if below + above > 0:
            print(f"  {col}: {below} below {lo}, {above} above {hi} → clipping")
            df_main[col] = df_main[col].clip(lower=lo, upper=hi)
            clipped_count += below + above
        else:
            print(f"  {col}: all values in range [{lo}, {hi}] ✓")

if clipped_count == 0:
    print("  All imputed values within physiological ranges ✓")

# ── 6. Final check & save ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Final summary & saving")
print("=" * 60)

remaining = df_main.isnull().sum()
remaining_any = remaining[remaining > 0]
if len(remaining_any) > 0:
    print("  Remaining missing values:")
    for col, cnt in remaining_any.items():
        print(f"    {col}: {cnt}")
else:
    print("  No missing values in target columns ✓")

# Ensure family_history_diabetes is integer
df_main["family_history_diabetes"] = df_main["family_history_diabetes"].astype(int)

# Round imputed continuous values to reasonable precision
for col in ["triglycerides", "systolic", "diastolic"]:
    df_main[col] = df_main[col].round(1)

df_main.to_csv("diana_dataset_imputed.csv", index=False)
print(f"\n  Saved diana_dataset_imputed.csv ({df_main.shape[0]} rows, {df_main.shape[1]} columns)")
print("=" * 60)
print("Done!")
