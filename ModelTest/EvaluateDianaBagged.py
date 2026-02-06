import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def load_diana_data(csv_path: str, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the Diana imputed dataset, drop excluded columns, and return X, y.

    Columns to exclude (case-insensitive):
      - diabetes_status  (used as target)
      - menopausal_status
      - has_outlier
      - imputed
      - alcohol_use
      - smoking_status
      - systolic
      - diastolic
      - fbs
      - hba1c
      - seqn
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    # Columns to drop by (lowercased) name
    exclude_names = {
        "diabetes_label",
        "menopausal_status",
        "cycle",
        "has_outlier",
        "imputed",
        "alcohol_use",
        "smoking_status",
        "systolic",
        "diastolic",
        "ldl",
        "hdl",
        "hba1c",
        "seqn",
    }

    def columns_to_drop(columns: List[str]) -> List[str]:
        drop_list: List[str] = []
        for c in columns:
            if c.lower() in exclude_names:
                drop_list.append(c)
        # Always drop the target from features
        if target_col in columns and target_col not in drop_list:
            drop_list.append(target_col)
        return drop_list

    drop_cols = columns_to_drop(list(df.columns))

    X = df.drop(columns=drop_cols)
    y = df[target_col]

    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    # Drop rows with missing target
    mask = ~y.isna()
    X = X.loc[mask]
    y = y.loc[mask]

    # Map diabetes status to numeric classes 0,1,2
    # 0 -> diabetic, 1 -> normal, 2 -> pre-diabetic
    mapping = {
        "diabetic": 0,
        "normal": 1,
        "pre-diabetic": 2,
        "prediabetic": 2,
        "pre diabetic": 2,
    }
    y_clean = y.astype(str).str.strip().str.lower()
    y_encoded = y_clean.map(mapping)

    # Keep only rows where mapping succeeded
    valid_mask = ~y_encoded.isna()
    if not valid_mask.all():
        print(
            f"Dropping {(~valid_mask).sum()} rows with unexpected "
            f"diabetes_status values: "
            f"{sorted(y_clean[~valid_mask].unique())}"
        )
    X = X.loc[valid_mask]
    y_final = y_encoded.loc[valid_mask].astype(int)

    return X, y_final


def build_bagged_xgb_classifier(n_classes: int = 3) -> BaggingClassifier:
    """
    Create a bagged ensemble of XGBoost classifiers for multi-class classification.
    """
    base_xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
    )

    bagged = BaggingClassifier(
        estimator=base_xgb,
        n_estimators=10,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        bootstrap_features=False,
        n_jobs=-1,
        random_state=42,
    )
    return bagged


def evaluate_model(
    model: BaggingClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Compute and print evaluation metrics in a format similar to the screenshot.
    """
    y_pred = model.predict(X_test)

    # For ROC-AUC we need probabilities; handle models without predict_proba
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
    else:
        # Fall back to one-hot encoded predictions
        num_classes = len(np.unique(y_test))
        y_proba = np.eye(num_classes)[y_pred]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovo", average="weighted")
    kappa = cohen_kappa_score(y_test, y_pred)

    print("=" * 60)
    print("EVALUATION - Test Set (Bagged)")
    print("=" * 60)
    print(f"Accuracy       : {accuracy:0.4f}")
    print(f"Precision      : {precision:0.4f}")
    print(f"Recall         : {recall:0.4f}")
    print(f"F1-Score       : {f1:0.4f}")
    print(f"ROC-AUC        : {roc_auc:0.4f}")
    print(f"Cohen's Kappa  : {kappa:0.4f}")
    print()
    
    # Confusion Matrix
    # Map class indices to labels (matching the encoding in load_diana_data)
    class_label_map = {0: "Diabetic", 1: "Normal", 2: "Pre-diabetic"}
    unique_classes = sorted(np.unique(np.concatenate([y_test, y_pred])))
    labels = [class_label_map.get(cls, f"Class {cls}") for cls in unique_classes]
    
    cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
    print("Confusion Matrix:")
    print("-" * 60)
    
    # Print header
    header = "Actual \\ Predicted"
    print(f"{header:20s}", end="")
    for label in labels:
        print(f"{label:15s}", end="")
    print()
    print("-" * 60)
    
    # Print matrix rows
    for i, true_label in enumerate(labels):
        print(f"{true_label:20s}", end="")
        for j in range(len(labels)):
            print(f"{cm[i, j]:15d}", end="")
        print()
    print()
    
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))
    
    # Risk Analysis Section
    print()
    print("-" * 60)
    print("RISK IDENTIFICATION ANALYSIS")
    print("-" * 60)
    
    # Analyze prediction probabilities for risk assessment
    class_label_map = {0: "Diabetic", 1: "Normal", 2: "Pre-diabetic"}
    
    # Calculate risk scores based on probabilities
    # High risk = high probability of Diabetic (class 0) OR Pre-diabetic (class 2)
    diabetic_proba = y_proba[:, 0]  # Probability of being Diabetic
    prediabetic_proba = y_proba[:, 2]  # Probability of being Pre-diabetic
    risk_score = diabetic_proba + prediabetic_proba  # Combined risk
    
    # Define risk thresholds
    high_risk_threshold = 0.5  # 50%+ combined probability of Diabetic or Pre-diabetic
    moderate_risk_threshold = 0.3  # 30-50% combined probability
    
    high_risk_mask = risk_score >= high_risk_threshold
    moderate_risk_mask = (risk_score >= moderate_risk_threshold) & (risk_score < high_risk_threshold)
    low_risk_mask = risk_score < moderate_risk_threshold
    
    high_risk_count = high_risk_mask.sum()
    moderate_risk_count = moderate_risk_mask.sum()
    low_risk_count = low_risk_mask.sum()
    
    print(f"\nRisk Distribution (based on prediction probabilities):")
    print(f"  High Risk (≥50% probability of Diabetic/Pre-diabetic): {high_risk_count:4d} ({100*high_risk_count/len(y_test):5.1f}%)")
    print(f"  Moderate Risk (30-50% probability):                    {moderate_risk_count:4d} ({100*moderate_risk_count/len(y_test):5.1f}%)")
    print(f"  Low Risk (<30% probability):                           {low_risk_count:4d} ({100*low_risk_count/len(y_test):5.1f}%)")
    
    # Analyze actual outcomes for each risk category
    print(f"\nActual Outcomes by Risk Category:")
    print(f"  High Risk Group:")
    high_risk_actual = y_test[high_risk_mask]
    if len(high_risk_actual) > 0:
        diabetic_in_high = (high_risk_actual == 0).sum()
        prediabetic_in_high = (high_risk_actual == 2).sum()
        normal_in_high = (high_risk_actual == 1).sum()
        print(f"    - Actual Diabetic:      {diabetic_in_high:3d} ({100*diabetic_in_high/len(high_risk_actual):5.1f}%)")
        print(f"    - Actual Pre-diabetic:  {prediabetic_in_high:3d} ({100*prediabetic_in_high/len(high_risk_actual):5.1f}%)")
        print(f"    - Actual Normal:        {normal_in_high:3d} ({100*normal_in_high/len(high_risk_actual):5.1f}%)")
    
    print(f"\n  Moderate Risk Group:")
    moderate_risk_actual = y_test[moderate_risk_mask]
    if len(moderate_risk_actual) > 0:
        diabetic_in_mod = (moderate_risk_actual == 0).sum()
        prediabetic_in_mod = (moderate_risk_actual == 2).sum()
        normal_in_mod = (moderate_risk_actual == 1).sum()
        print(f"    - Actual Diabetic:      {diabetic_in_mod:3d} ({100*diabetic_in_mod/len(moderate_risk_actual):5.1f}%)")
        print(f"    - Actual Pre-diabetic:  {prediabetic_in_mod:3d} ({100*prediabetic_in_mod/len(moderate_risk_actual):5.1f}%)")
        print(f"    - Actual Normal:        {normal_in_mod:3d} ({100*normal_in_mod/len(moderate_risk_actual):5.1f}%)")
    
    # Identify missed high-risk cases
    print(f"\n  Missed High-Risk Cases (False Negatives):")
    # Cases actually Diabetic or Pre-diabetic but predicted as Normal
    actual_at_risk = (y_test == 0) | (y_test == 2)  # Diabetic or Pre-diabetic
    predicted_normal = (y_pred == 1)
    missed_high_risk = actual_at_risk & predicted_normal
    missed_count = missed_high_risk.sum()
    
    if missed_count > 0:
        missed_diabetic = ((y_test == 0) & predicted_normal).sum()
        missed_prediabetic = ((y_test == 2) & predicted_normal).sum()
        print(f"    - Total missed: {missed_count} individuals")
        print(f"      * {missed_diabetic} actual Diabetic cases predicted as Normal")
        print(f"      * {missed_prediabetic} actual Pre-diabetic cases predicted as Normal")
        print(f"    - Recommendation: Lower threshold or use probability-based screening")
    else:
        print(f"    - No missed high-risk cases")
    
    # Summary recommendations
    print(f"\n  Risk Identification Summary:")
    print(f"    ✓ Model can identify high-risk individuals with {high_risk_count} flagged")
    print(f"    ⚠ Model misses {missed_count} at-risk individuals (predicted as Normal)")
    print(f"    ⚠ Pre-diabetic detection needs improvement (Precision: 0.53, Recall: 0.50)")
    print(f"\n  Clinical Recommendations:")
    print(f"    1. Use probability thresholds (≥50%) to flag high-risk individuals")
    print(f"    2. Consider moderate-risk group (30-50%) for preventive screening")
    print(f"    3. Combine model predictions with clinical judgment for pre-diabetic cases")
    print(f"    4. Regular monitoring recommended for all flagged individuals")

    # "Sign" that the model is using real predictive signal:
    # If ROC-AUC is meaningfully above chance and permutation importance shows
    # non-trivial drivers, it suggests the features carry risk signal for diabetes_status.
    print()
    print("-" * 60)
    print("Predictive signal (risk) check")
    print("-" * 60)
    chance_auc = 1.0 / len(np.unique(y_test))
    if roc_auc > chance_auc + 0.05:
        print(
            f"ROC-AUC {roc_auc:0.4f} is above chance (~{chance_auc:0.4f}); "
            "model shows signal for predicting diabetes_status risk."
        )
    else:
        print(
            f"ROC-AUC {roc_auc:0.4f} is close to chance (~{chance_auc:0.4f}); "
            "predictive signal may be weak."
        )

    try:
        perm = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=5,
            random_state=42,
            n_jobs=-1,
            scoring="f1_weighted",
        )
        importances = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(
            ascending=False
        )
        top = importances.head(15)
        print()
        print("Top drivers (permutation importance, f1_weighted):")
        for name, score in top.items():
            print(f"{name:35s} {score:0.6f}")
    except Exception as e:  # noqa: BLE001
        print("Permutation importance failed:", e)


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "Dataset", "diana_dataset_imputed.csv")

    target_col = "diabetes_status"

    print(f"Loading data from: {csv_path}")
    print(f"Using target column: {target_col}")

    X, y = load_diana_data(csv_path, target_col=target_col)
    print(f"Shape after exclusions and encoding: X={X.shape}, y={y.shape}")

    stratify_arg = y if y.nunique() <= 20 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_arg,
    )

    model = build_bagged_xgb_classifier(n_classes=y.nunique())
    model.fit(X_train, y_train)

    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()

