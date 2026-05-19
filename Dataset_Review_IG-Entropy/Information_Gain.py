import pandas as pd
import numpy as np
from collections import Counter


def calculate_entropy(labels):
    """
    Calculate entropy for a set of labels.
    H(S) = -sum(p_i * log2(p_i))
    """
    if len(labels) == 0:
        return 0.0
    
    # Count occurrences of each label
    label_counts = Counter(labels)
    total = len(labels)
    
    # Calculate entropy
    entropy = 0.0
    for count in label_counts.values():
        if count > 0:
            probability = count / total
            entropy -= probability * np.log2(probability)
    
    return entropy


def calculate_information_gain_numeric(data, feature, target, num_bins=5):
    """
    Calculate information gain for a numeric feature by discretizing it into bins.
    """
    # Discretize numeric feature into bins
    try:
        data_copy = data.copy()
        data_copy[f'{feature}_binned'] = pd.qcut(data_copy[feature], q=num_bins, 
                                                   duplicates='drop', labels=False)
        binned_feature = f'{feature}_binned'
    except:
        # If qcut fails, use cut instead
        try:
            data_copy = data.copy()
            data_copy[f'{feature}_binned'] = pd.cut(data_copy[feature], bins=num_bins, 
                                                     labels=False)
            binned_feature = f'{feature}_binned'
        except:
            return 0.0
    
    return calculate_information_gain_categorical(data_copy, binned_feature, target)


def calculate_information_gain_categorical(data, feature, target):
    """
    Calculate information gain for a categorical feature.
    IG(S, A) = H(S) - H(S|A)
    """
    # Calculate entropy of the entire dataset
    total_entropy = calculate_entropy(data[target].values)
    
    # Calculate weighted entropy after split
    total_samples = len(data)
    weighted_entropy = 0.0
    
    feature_values = data[feature].unique()
    
    for value in feature_values:
        # Get subset of data with this feature value
        subset = data[data[feature] == value]
        subset_size = len(subset)
        
        # Calculate entropy for this subset
        subset_entropy = calculate_entropy(subset[target].values)
        
        # Add weighted entropy
        weight = subset_size / total_samples
        weighted_entropy += weight * subset_entropy
    
    # Information gain = total entropy - weighted entropy
    information_gain = total_entropy - weighted_entropy
    
    return information_gain


def calculate_all_information_gains(data, target_column, num_bins=5):
    """
    Calculate information gain for all features in the dataset.
    """
    print("=" * 80)
    print("INFORMATION GAIN CALCULATION FOR DIABETES DATASET")
    print("=" * 80)
    print()
    
    print(f"Dataset Information:")
    print(f"  Total Samples: {len(data)}")
    print(f"  Total Features: {len(data.columns) - 1}")
    print(f"  Target Variable: {target_column}")
    print()
    
    # Display target variable distribution
    print("Target Variable Distribution:")
    print("-" * 80)
    target_counts = data[target_column].value_counts().sort_index()
    for label, count in target_counts.items():
        percentage = (count / len(data)) * 100
        print(f"  {label}: {count} samples ({percentage:.2f}%)")
    print()
    
    # Calculate base entropy
    base_entropy = calculate_entropy(data[target_column].values)
    print(f"Base Entropy H(S): {base_entropy:.6f} bits")
    print()
    
    print("=" * 80)
    print("CALCULATING INFORMATION GAIN FOR EACH FEATURE")
    print("=" * 80)
    print()
    
    # Identify feature types
    exclude_columns = [target_column, 'SEQN', 'diabetes_label', 'diabetes_status']
    
    categorical_features = ['smoking_status', 'physical_activity', 'alcohol_use', 
                           'cycle', 'menopausal_status', 'has_outlier', 'imputed']
    
    numeric_features = [col for col in data.columns 
                       if col not in exclude_columns 
                       and col not in categorical_features]
    
    print(f"Feature Types:")
    print(f"  Categorical Features: {len([f for f in categorical_features if f in data.columns])}")
    print(f"  Numeric Features: {len(numeric_features)}")
    print()
    
    print("Methodology:")
    print(f"  - Categorical features: Direct calculation using feature values")
    print(f"  - Numeric features: Discretized into {num_bins} bins using quantiles")
    print()
    
    # Calculate information gain for all features
    information_gains = {}
    
    print("Processing Features")
    print()
    
    # Process numeric features
    for feature in numeric_features:
        if feature in data.columns:
            ig = calculate_information_gain_numeric(data, feature, target_column, num_bins)
            information_gains[feature] = {
                'type': 'numeric',
                'information_gain': ig,
                'gain_ratio': (ig / base_entropy) * 100 if base_entropy > 0 else 0
            }
            print(f"[+] {feature} (numeric): IG = {ig:.6f} bits")
    
    print()
    
    # Process categorical features
    for feature in categorical_features:
        if feature in data.columns:
            ig = calculate_information_gain_categorical(data, feature, target_column)
            information_gains[feature] = {
                'type': 'categorical',
                'information_gain': ig,
                'gain_ratio': (ig / base_entropy) * 100 if base_entropy > 0 else 0
            }
            print(f"[+] {feature} (categorical): IG = {ig:.6f} bits")
    
    print()
    
    return information_gains, base_entropy


def display_detailed_results(information_gains, base_entropy):
    """
    Display detailed information gain results.
    """
    print("INFORMATION GAIN RESULTS - RANKED BY IMPORTANCE")
    print()
    
    # Sort by information gain
    sorted_features = sorted(information_gains.items(), 
                           key=lambda x: x[1]['information_gain'], 
                           reverse=True)
    
    print(f"{'Rank':<6} {'Feature':<25} {'Type':<12} {'Info Gain':<15} {'% of Base':<12}")
    print("-" * 80)
    
    for rank, (feature, stats) in enumerate(sorted_features, 1):
        ig = stats['information_gain']
        gain_ratio = stats['gain_ratio']
        feat_type = stats['type']
        
        print(f"{rank:<6} {feature:<25} {feat_type:<12} {ig:.6f} bits    {gain_ratio:>6.2f}%")
    
    print()
    print("=" * 80)
    
    # Detailed analysis of top features
    print("DETAILED ANALYSIS OF TOP 5 FEATURES")
    print("=" * 80)
    print()
    
    for rank, (feature, stats) in enumerate(sorted_features[:5], 1):
        ig = stats['information_gain']
        gain_ratio = stats['gain_ratio']
        reduction = base_entropy - (base_entropy - ig)
        
        print(f"{rank}. {feature} ({stats['type']})")
        print(f"   Information Gain: {ig:.6f} bits")
        print(f"   Entropy Reduction: {ig:.6f} bits ({gain_ratio:.2f}%)")
        print(f"   Remaining Entropy: {base_entropy - ig:.6f} bits")
        print()
    
    print("=" * 80)
    
    # Feature importance categories
    print("FEATURE IMPORTANCE CATEGORIES")
    print("=" * 80)
    print()
    
    high_importance = [(f, s) for f, s in sorted_features if s['information_gain'] > 0.1]
    medium_importance = [(f, s) for f, s in sorted_features 
                        if 0.01 < s['information_gain'] <= 0.1]
    low_importance = [(f, s) for f, s in sorted_features if s['information_gain'] <= 0.01]
    
    print(f"High Importance (IG > 0.1): {len(high_importance)} features")
    for feature, stats in high_importance:
        print(f"  - {feature}: {stats['information_gain']:.6f} bits")
    print()
    
    print(f"Medium Importance (0.01 < IG <= 0.1): {len(medium_importance)} features")
    for feature, stats in medium_importance:
        print(f"  - {feature}: {stats['information_gain']:.6f} bits")
    print()
    
    print(f"Low Importance (IG <= 0.01): {len(low_importance)} features")
    for feature, stats in low_importance:
        print(f"  - {feature}: {stats['information_gain']:.6f} bits")
    print()
    
    print("=" * 80)
    
    # Interpretation
    print("INTERPRETATION")
    print("=" * 80)
    print()
    
    print("What is Information Gain?")
    print("  Information Gain measures how much information a feature provides")
    print("  about the target variable (diabetes status).")
    print()
    print("Formula: IG(S, A) = H(S) - H(S|A)")
    print("  - H(S): Entropy of the entire dataset")
    print("  - H(S|A): Average entropy after splitting by feature A")
    print("  - IG(S, A): Reduction in entropy (uncertainty)")
    print()
    print("Interpretation:")
    print("  - Higher IG => Feature is more useful for prediction")
    print("  - Lower IG => Feature provides less discriminative information")
    print("  - IG = 0 => Feature provides no information about target")
    print()
    print("Use in Machine Learning:")
    print("  - Feature Selection: Select features with highest IG")
    print("  - Decision Trees: Choose split features based on IG")
    print("  - Understanding Data: Identify which factors matter most")
    print()
    
    # Save results to CSV
    results_df = pd.DataFrame([
        {
            'Feature': feature,
            'Type': stats['type'],
            'Information_Gain': stats['information_gain'],
            'Percentage_of_Base_Entropy': stats['gain_ratio'],
            'Rank': rank
        }
        for rank, (feature, stats) in enumerate(sorted_features, 1)
    ])
    
    output_file = 'information_gain_results.csv'
    results_df.to_csv(output_file, index=False)
    print("=" * 80)
    print(f"Results saved to: {output_file}")
    print("=" * 80)
    print()
    
    return results_df


# Main execution
if __name__ == "__main__":
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv('diana_dataset_imputed.csv')
    
    print(f"Dataset loaded successfully with {len(df)} samples and {len(df.columns)} features.")
    print()
    
    # Determine target variable
    if 'diabetes_status' in df.columns:
        target_column = 'diabetes_status'
    elif 'diabetes_label' in df.columns:
        target_column = 'diabetes_label'
    else:
        print("Error: No suitable target variable found!")
        exit(1)
    
    # Calculate information gain for all features
    information_gains, base_entropy = calculate_all_information_gains(df, target_column, num_bins=5)
    
    # Display detailed results
    results_df = display_detailed_results(information_gains, base_entropy)
    
    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print()
    print(f"Summary:")
    print(f"  - Analyzed {len(information_gains)} features")
    print(f"  - Base entropy: {base_entropy:.6f} bits")
    print(f"  - Top feature: {results_df.iloc[0]['Feature']}")
    print(f"  - Top IG: {results_df.iloc[0]['Information_Gain']:.6f} bits")
    print()
