import os
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def load_data(csv_path: str, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the Diana dataset, drop excluded columns, and return X, y.

    Columns to exclude (case-insensitive):
      - hba1c
      - systolic
      - diastolic
      - menopausal_status
      - imputed
      - seqn
      - smoking_status
      - alcohol_use
      - cycle
      - has_outlier
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
    # Matches the exclusions used in EvaluateDianaBagged, plus "cycle"
    exclude_names = {
        "hba1c",
        "systolic",
        "diastolic",
        "menopausal_status",
        "imputed",
        "seqn",
        "smoking_status",
        "alcohol_use",
        "cycle",
        "has_outlier",
        "diabetes_label"
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
    y_clean = (
        y.astype(str)
        .str.strip()
        .str.lower()
    )
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


def train_xgboost_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    model_output_path: str = "xgboost_model.joblib",
) -> XGBClassifier:
    """
    Train an XGBoost classifier on the given features and labels.
    Saves the trained model to `model_output_path`.
    """
    # Basic train/validation split
    stratify_arg = y if y.nunique() <= 20 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_arg,
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature importances (optional, but useful)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_importances = sorted(
            zip(X.columns, importances),
            key=lambda x: x[1],
            reverse=True,
        )
        print("\nTop 20 feature importances:")
        for name, score in feature_importances[:20]:
            print(f"{name:30s}: {score:.4f}")

    # Save the trained model
    joblib.dump(model, model_output_path)
    print(f"\nModel saved to: {model_output_path}")

    return model


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "Dataset", "diana_dataset_imputed.csv")

    # Default target column based on your existing analysis script
    target_col = "diabetes_status"

    print(f"Loading data from: {csv_path}")
    print(f"Using target column: {target_col}")

    X, y = load_data(csv_path, target_col=target_col)
    print(f"Shape after exclusions and encoding: X={X.shape}, y={y.shape}")

    train_xgboost_classifier(X, y, model_output_path="xgboost_diana.joblib")


if __name__ == "__main__":
    main()

