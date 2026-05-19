import pandas as pd
import numpy as np
from collections import Counter


def calculate_entropy(data, target_column):
    """
    Calculate entropy for the entire dataset based on the target variable.
    """
    print("=" * 80)
    print("ENTROPY CALCULATION FOR DIABETES DATASET")
    print("=" * 80)
    print()
    
    # Get the target variable values
    target_values = data[target_column].values
    total_samples = len(target_values)
    
    print(f"Target Variable: {target_column}")
    print(f"Total Number of Samples: {total_samples}")
    print()
    
    # Count occurrences of each class
    class_counts = Counter(target_values)
    print("Class Distribution:")
    print("-" * 80)
    for class_label, count in sorted(class_counts.items()):
        print(f"  Class '{class_label}': {count} samples")
    print()
    
    # Calculate probabilities for each class
    print("STEP 1: Calculate Probability for Each Class")
    print("-" * 80)
    print("Formula: p_i = (Number of samples in class i) / (Total number of samples)")
    print()
    
    probabilities = {}
    for class_label, count in sorted(class_counts.items()):
        prob = count / total_samples
        probabilities[class_label] = prob
        print(f"  p('{class_label}') = {count}/{total_samples} = {prob:.6f}")
    print()
    
    # Verify probabilities sum to 1
    prob_sum = sum(probabilities.values())
    print(f"Verification: Sum of all probabilities = {prob_sum:.6f}")
    print()
    
    # Calculate entropy
    print("STEP 2: Calculate Entropy Using Shannon's Formula")
    print("-" * 80)
    print("Formula: H(S) = -Sum(p_i * log2(p_i))")
    print()
    print("where:")
    print("  - H(S) is the entropy of the dataset")
    print("  - p_i is the probability of class i")
    print("  - log2 is the logarithm base 2")
    print("  - Sum is the summation over all classes")
    print()
    
    entropy = 0
    print("Calculating each term:")
    print()
    
    for class_label, prob in sorted(probabilities.items()):
        if prob > 0:  # Avoid log(0) which is undefined
            log_prob = np.log2(prob)
            term = -prob * log_prob
            entropy += term
            print(f"  Class '{class_label}':")
            print(f"    p_i = {prob:.6f}")
            print(f"    log2(p_i) = log2({prob:.6f}) = {log_prob:.6f}")
            print(f"    -p_i * log2(p_i) = -{prob:.6f} x {log_prob:.6f} = {term:.6f}")
            print()
    
    print("=" * 80)
    print(f"TOTAL ENTROPY: H(S) = {entropy:.6f} bits")
    print("=" * 80)
    print()
    
    # Interpretation
    print("INTERPRETATION")
    print("-" * 80)
    print()
    
    # Calculate maximum possible entropy for this number of classes
    num_classes = len(class_counts)
    max_entropy = np.log2(num_classes) if num_classes > 1 else 0
    
    print(f"Number of Classes: {num_classes}")
    print(f"Maximum Possible Entropy (log2({num_classes})): {max_entropy:.6f} bits")
    print()
    
    # Calculate normalized entropy (0 to 1 scale)
    if max_entropy > 0:
        normalized_entropy = entropy / max_entropy
        print(f"Normalized Entropy: {entropy:.6f}/{max_entropy:.6f} = {normalized_entropy:.4f} ({normalized_entropy*100:.2f}%)")
    else:
        normalized_entropy = 0
        print(f"Normalized Entropy: 0 (only one class present)")
    print()
    
    # Interpretation based on entropy value
    print("Dataset Purity Analysis:")
    print()
    
    if entropy == 0:
        print("  >> Pure Dataset (Entropy = 0)")
        print("    All samples belong to the same class.")
        print("    No impurity or uncertainty in the data.")
    elif normalized_entropy < 0.3:
        print(f"  >> Relatively Pure Dataset (Normalized Entropy = {normalized_entropy:.4f})")
        print("    One class dominates significantly.")
        print("    Low impurity and uncertainty.")
    elif normalized_entropy < 0.7:
        print(f"  >> Moderately Impure Dataset (Normalized Entropy = {normalized_entropy:.4f})")
        print("    Classes are somewhat balanced but one or more classes dominate.")
        print("    Moderate impurity and uncertainty.")
    else:
        print(f"  >> Highly Impure Dataset (Normalized Entropy = {normalized_entropy:.4f})")
        print("    Classes are relatively balanced.")
        print("    High impurity and uncertainty.")
        print("    Close to maximum entropy (most diverse distribution).")
    
    print()
    print("What does this mean?")
    print()
    print("  - Entropy measures the uncertainty or impurity in the dataset.")
    print("  - Entropy = 0: Perfect purity (all samples same class)")
    print(f"  - Entropy = {max_entropy:.2f}: Maximum impurity (perfectly balanced classes)")
    print("  - Higher entropy => More impurity => More diverse/mixed dataset")
    print("  - Lower entropy => Less impurity => More homogeneous dataset")
    print()
    
    # Additional statistics
    print("Additional Statistics:")
    print("-" * 80)
    majority_class = max(class_counts.items(), key=lambda x: x[1])
    minority_class = min(class_counts.items(), key=lambda x: x[1])
    
    print(f"  Majority Class: '{majority_class[0]}' with {majority_class[1]} samples ({majority_class[1]/total_samples*100:.2f}%)")
    print(f"  Minority Class: '{minority_class[0]}' with {minority_class[1]} samples ({minority_class[1]/total_samples*100:.2f}%)")
    
    if num_classes > 1:
        imbalance_ratio = majority_class[1] / minority_class[1]
        print(f"  Class Imbalance Ratio: {imbalance_ratio:.2f}:1")
    
    print()
    print("=" * 80)
    
    return entropy


def calculate_row_entropy(row_values):
    """
    Calculate entropy for a single row by treating normalized values as a probability distribution.
    Only considers positive numerical values.
    """
    # Filter out non-positive values
    positive_values = row_values[row_values > 0]
    
    if len(positive_values) == 0:
        return 0.0
    
    # Normalize to create a probability distribution
    total = np.sum(positive_values)
    probabilities = positive_values / total
    
    # Calculate entropy using Shannon's formula: H = -sum(p * log2(p))
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * np.log2(p)
    
    return entropy


def calculate_entropy_by_row(data):
    """
    Calculate entropy for each row in the dataset.
    """
    print("=" * 80)
    print("ROW-WISE ENTROPY CALCULATION FOR DIABETES DATASET")
    print("=" * 80)
    print()
    
    # Identify numeric columns (exclude identifiers and categorical columns)
    exclude_columns = ['SEQN', 'smoking_status', 'physical_activity', 'alcohol_use', 
                      'cycle', 'diabetes_status', 'diabetes_label', 'menopausal_status', 
                      'has_outlier', 'imputed']
    
    numeric_columns = [col for col in data.columns if col not in exclude_columns]
    
    print(f"Dataset Information:")
    print(f"  Total Rows: {len(data)}")
    print(f"  Total Columns: {len(data.columns)}")
    print(f"  Numeric Columns Used for Entropy: {len(numeric_columns)}")
    print()
    
    print("Numeric Columns:")
    for i, col in enumerate(numeric_columns, 1):
        print(f"  {i}. {col}")
    print()
    
    print("-" * 80)
    print("METHODOLOGY:")
    print("-" * 80)
    print("For each row:")
    print("  1. Extract all numeric feature values")
    print("  2. Filter out non-positive values")
    print("  3. Normalize values to create a probability distribution")
    print("     (each value divided by sum of all values)")
    print("  4. Apply Shannon's entropy formula: H = -sum(p_i * log2(p_i))")
    print()
    print("Interpretation:")
    print("  - Higher entropy => More evenly distributed feature values")
    print("  - Lower entropy => Some features dominate the row")
    print("=" * 80)
    print()
    
    # Calculate entropy for each row
    print("Calculating entropy for each row...")
    print()
    
    row_entropies = []
    
    for idx, row in data.iterrows():
        # Get numeric values for this row
        numeric_values = row[numeric_columns].values
        
        # Calculate entropy for this row
        entropy = calculate_row_entropy(numeric_values)
        row_entropies.append(entropy)
    
    # Add entropy as a new column to the dataframe
    data_with_entropy = data.copy()
    data_with_entropy['row_entropy'] = row_entropies
    
    # Display statistics
    print("ENTROPY STATISTICS ACROSS ALL ROWS")
    print()
    
    entropy_array = np.array(row_entropies)
    
    print(f"  Mean Entropy:     {np.mean(entropy_array):.6f} bits")
    print(f"  Median Entropy:   {np.median(entropy_array):.6f} bits")
    print(f"  Std Deviation:    {np.std(entropy_array):.6f} bits")
    print(f"  Min Entropy:      {np.min(entropy_array):.6f} bits")
    print(f"  Max Entropy:      {np.max(entropy_array):.6f} bits")
    print()
    
    # Calculate theoretical maximum entropy
    max_possible_entropy = np.log2(len(numeric_columns))
    print(f"  Theoretical Max:  {max_possible_entropy:.6f} bits")
    print(f"    (log2({len(numeric_columns)}) - if all features were equal)")
    print()
    
    # Show percentiles
    print("Entropy Distribution (Percentiles):")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(entropy_array, p)
        print(f"  {p}th percentile: {value:.6f} bits")
    print()
    
    # Show sample rows with different entropy levels
    print("=" * 80)
    print("SAMPLE ROWS WITH DIFFERENT ENTROPY LEVELS")
    print("=" * 80)
    print()
    
    # Find rows with min, max, and median entropy
    min_idx = data_with_entropy['row_entropy'].idxmin()
    max_idx = data_with_entropy['row_entropy'].idxmax()
    median_val = np.median(entropy_array)
    median_idx = (data_with_entropy['row_entropy'] - median_val).abs().idxmin()
    
    print("Row with MINIMUM Entropy (most concentrated):")
    print(f"  Row Index: {min_idx}")
    print(f"  Entropy: {data_with_entropy.loc[min_idx, 'row_entropy']:.6f} bits")
    if 'diabetes_status' in data_with_entropy.columns:
        print(f"  Diabetes Status: {data_with_entropy.loc[min_idx, 'diabetes_status']}")
    print()
    
    print("Row with MEDIAN Entropy:")
    print(f"  Row Index: {median_idx}")
    print(f"  Entropy: {data_with_entropy.loc[median_idx, 'row_entropy']:.6f} bits")
    if 'diabetes_status' in data_with_entropy.columns:
        print(f"  Diabetes Status: {data_with_entropy.loc[median_idx, 'diabetes_status']}")
    print()
    
    print("Row with MAXIMUM Entropy (most evenly distributed):")
    print(f"  Row Index: {max_idx}")
    print(f"  Entropy: {data_with_entropy.loc[max_idx, 'row_entropy']:.6f} bits")
    if 'diabetes_status' in data_with_entropy.columns:
        print(f"  Diabetes Status: {data_with_entropy.loc[max_idx, 'diabetes_status']}")
    print()
    
    # Analyze entropy by diabetes status
    if 'diabetes_status' in data_with_entropy.columns:
        print("=" * 80)
        print("ENTROPY BY DIABETES STATUS")
        print("=" * 80)
        print()
        
        for status in sorted(data_with_entropy['diabetes_status'].unique()):
            status_data = data_with_entropy[data_with_entropy['diabetes_status'] == status]
            status_entropies = status_data['row_entropy'].values
            
            print(f"{status}:")
            print(f"  Count:           {len(status_entropies)}")
            print(f"  Mean Entropy:    {np.mean(status_entropies):.6f} bits")
            print(f"  Median Entropy:  {np.median(status_entropies):.6f} bits")
            print(f"  Std Deviation:   {np.std(status_entropies):.6f} bits")
            print()


# Main execution
if __name__ == "__main__":
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv('diana_dataset_imputed.csv')
    
    print(f"Dataset loaded successfully with {len(df)} samples and {len(df.columns)} features.")
    print()
    
    # Check which target variable to use
    if 'diabetes_label' in df.columns:
        target_column = 'diabetes_label'
    elif 'diabetes_status' in df.columns:
        target_column = 'diabetes_status'
    else:
        print("Error: No suitable target variable found!")
        exit(1)
    
    # ===================================================================
    # PART 1: Calculate entropy for the whole dataset (target variable)
    # ===================================================================
    dataset_entropy = calculate_entropy(df, target_column)
    
    print()
    print()
    
    # ===================================================================
    # PART 2: Calculate entropy for each row
    # ===================================================================
    df_with_entropy = calculate_entropy_by_row(df)
    
    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  - Calculated dataset entropy: {dataset_entropy:.6f} bits")
    print(f"  - Calculated entropy for {len(df)} individual rows")
    print(f"  - Both whole dataset and row-wise entropy analyses completed")
    print()
