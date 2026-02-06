import csv
import math
import os
from collections import Counter
from typing import Dict, List, Tuple


def shannon_entropy_from_counts(counts: Dict[str, int]) -> float:
    """
    Compute Shannon entropy (in bits) from a dictionary of counts.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for c in counts.values():
        if c == 0:
            continue
        p = c / total
        entropy -= p * math.log2(p)
    return entropy


def is_numeric(value: str) -> bool:
    """
    Check if a string value can be interpreted as a float.
    Empty strings and None are treated as non-numeric.
    """
    if value is None or value == "":
        return False
    try:
        float(value)
        return True
    except ValueError:
        return False


def compute_numeric_bins(values: List[float], num_bins: int = 10) -> Dict[str, int]:
    """
    Bin numeric values into equal-width bins and return counts per bin.
    """
    if not values:
        return {}

    v_min = min(values)
    v_max = max(values)

    # If all values are the same, treat as a single bin
    if v_min == v_max:
        return {f"[{v_min}, {v_max}]": len(values)}

    bin_width = (v_max - v_min) / num_bins
    bins = Counter()

    for v in values:
        # Find bin index in [0, num_bins-1]
        idx = int((v - v_min) / bin_width)
        if idx == num_bins:  # include the max value in the last bin
            idx -= 1
        bin_start = v_min + idx * bin_width
        bin_end = bin_start + bin_width
        key = f"[{bin_start:.4f}, {bin_end:.4f})"
        bins[key] += 1

    return dict(bins)


def analyze_entropy(csv_path: str) -> Tuple[Dict[str, float], str]:
    """
    Load the CSV file and compute entropy for each column.

    Returns:
        - A dict mapping column name -> entropy (bits)
        - The name of the target/label column if detected (e.g., 'diabetes_status'),
          otherwise an empty string.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError("CSV file has no header / fieldnames.")

        # Initialize per-column storage
        col_values: Dict[str, List[str]] = {col: [] for col in fieldnames}

        for row in reader:
            for col in fieldnames:
                col_values[col].append(row.get(col, ""))

    entropies: Dict[str, float] = {}

    for col, values in col_values.items():
        # Drop missing values
        values = [v for v in values if v not in ("", None)]
        if not values:
            entropies[col] = 0.0
            continue

        # Determine if this column is numeric
        numeric_flags = [is_numeric(v) for v in values]
        is_num_col = all(numeric_flags)

        if is_num_col:
            num_vals = [float(v) for v in values]
            bin_counts = compute_numeric_bins(num_vals, num_bins=10)
            entropies[col] = shannon_entropy_from_counts(bin_counts)
        else:
            counts = Counter(values)
            entropies[col] = shannon_entropy_from_counts(counts)

    # Try to detect a likely label/target column (heuristic)
    possible_label_cols = [
        "diabetes_status",
        "diabetes_label",
        "label",
        "target",
        "class",
    ]
    detected_label = ""
    for cand in possible_label_cols:
        if cand in col_values:
            detected_label = cand
            break

    return entropies, detected_label


def main() -> None:
    # CSV is expected at: Dataset/diana_dataset_imputed.csv
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "Dataset", "diana_dataset_imputed.csv")

    entropies, label_col = analyze_entropy(csv_path)

    print(f"Entropy analysis for file: {csv_path}")
    print("-" * 60)
    print(f"{'Column':30s} | {'Entropy (bits)':>14s}")
    print("-" * 60)

    # Print label column (if detected) first
    if label_col and label_col in entropies:
        print(f"{label_col:30s} | {entropies[label_col]:14.4f}  <-- likely label")

    # Then print all other columns
    for col, h in entropies.items():
        if col == label_col:
            continue
        print(f"{col:30s} | {h:14.4f}")


if __name__ == "__main__":
    main()

