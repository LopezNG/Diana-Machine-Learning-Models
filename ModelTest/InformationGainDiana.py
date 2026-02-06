import os
from typing import List, Tuple

import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from Entropy import load_dataset, detect_label_column, compute_information_gain


# Columns to exclude from information gain calculations (case-insensitive)
EXCLUDED_COLUMNS = {
    "fbs",
    "hba1c",
    "systolic",
    "diastolic",
    "has_outlier",
    "menopausal_status",
    "imputed",
    "seqn",
    "smokingstatus",
    "cycle",
    "alcohol_use",
}


def compute_manual_information_gain(csv_path: str) -> Tuple[str, List[Tuple[str, float]]]:
    """
    Use the helper functions from Entropy.py to compute
    information gain for all features in diana_dataset_imputed.

    Returns:
        - label column name
        - list of (feature, IG) sorted by IG descending
    """
    col_values, fieldnames = load_dataset(csv_path)
    label_col = detect_label_column(col_values)

    if not label_col:
        raise ValueError(
            "Could not detect label column. "
            "Please ensure one of the standard label names exists in the CSV."
        )

    ig_results: List[Tuple[str, float]] = []

    # Build a case-insensitive map from lowercase name -> actual column name
    lower_to_col = {c.lower(): c for c in fieldnames}

    for feature in fieldnames:
        if feature == label_col:
            continue

        # Skip excluded columns (case-insensitive)
        if feature.lower() in EXCLUDED_COLUMNS:
            continue

        ig = compute_information_gain(col_values, feature, label_col)
        ig_results.append((feature, ig))

    # Sort features by information gain descending
    ig_results.sort(key=lambda x: x[1], reverse=True)
    return label_col, ig_results


def compute_sklearn_mutual_info(csv_path: str, label_col: str) -> pd.DataFrame:
    """
    Compute mutual information (information gain) using scikit-learn.

    Returns:
        - DataFrame with columns ['Feature', 'IG'] sorted descending by IG.
    """
    df = pd.read_csv(csv_path)

    # Match Entropy.py behaviour: drop smoking_status == 'Former' if present
    if "smoking_status" in df.columns:
        df = df[df["smoking_status"] != "Former"]

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset.")

    # Case-insensitive exclusion of columns
    lower_to_col = {c.lower(): c for c in df.columns}

    drop_cols = [label_col]

    # Ensure SEQN is dropped if present, independent of case
    if "seqn" in lower_to_col:
        drop_cols.append(lower_to_col["seqn"])

    # Add user-specified excluded columns (other than the label)
    for col_lower in EXCLUDED_COLUMNS:
        if col_lower in lower_to_col:
            col_name = lower_to_col[col_lower]
            if col_name != label_col and col_name not in drop_cols:
                drop_cols.append(col_name)

    X = df.drop(columns=drop_cols)
    y = df[label_col]

    ig_scores = mutual_info_classif(
        X,
        y,
        discrete_features="auto",
        random_state=42,
    )

    ig_df = (
        pd.DataFrame(
            {
                "Feature": X.columns,
                "IG": ig_scores,
            }
        )
        .sort_values(by="IG", ascending=False)
        .reset_index(drop=True)
    )

    return ig_df


def main() -> None:
    """
    Standalone script to show prediction gain / information gain
    for all features in Dataset/diana_dataset_imputed.csv.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "Dataset", "diana_dataset_imputed.csv")

    print(f"Using dataset: {csv_path}")
    print("=" * 70)

    # --- Manual information gain using Entropy.py helpers ---
    label_col, ig_results = compute_manual_information_gain(csv_path)

    print(f"\nManual information gain with respect to label '{label_col}':")
    print("-" * 70)
    print(f"{'Feature':30s} | {'IG (bits)':>14s}")
    print("-" * 70)
    for feature, ig in ig_results:
        print(f"{feature:30s} | {ig:14.4f}")

    # --- sklearn mutual_info_classif based information gain ---
    print("\nScikit-learn mutual information (information gain):")
    print("-" * 70)
    try:
        ig_df = compute_sklearn_mutual_info(csv_path, label_col)
        print(ig_df)
    except Exception as e:  # noqa: BLE001
        print("Could not compute sklearn mutual_info_classif:", e)


if __name__ == "__main__":
    main()

