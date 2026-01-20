import pandas as pd
import numpy as np
from math import log2
import warnings
warnings.filterwarnings('ignore')

def entropy(target):
    if len(target) == 0:
        return 0

    value_counts = target.value_counts()
    probabilities = value_counts / len(target)
    
    ent = 0
    for prob in probabilities:
        if prob > 0:  
            ent -= prob * log2(prob)
    
    return ent

def information_gain(data, feature, target_name):
    total_entropy = entropy(data[target_name])

    feature_values = data[feature].unique()

    weighted_entropy = 0
    total_samples = len(data)
    
    for value in feature_values:
        subset = data[data[feature] == value]
        weight = len(subset) / total_samples

        subset_entropy = entropy(subset[target_name])
        weighted_entropy += weight * subset_entropy

    ig = total_entropy - weighted_entropy
    
    return ig

def discretize_continuous_feature(series, num_bins=5):
    try:
        discretized = pd.qcut(series, q=num_bins, duplicates='drop', labels=False)
    except:
        try:
            discretized = pd.cut(series, bins=num_bins, duplicates='drop', labels=False)
        except:
            discretized = series
    
    return discretized

def main():
    print("="*80)
    print("FEATURE SELECTION USING ENTROPY AND INFORMATION GAIN")
    print("="*80)
    print("\nLoading dataset...")
    
    df = pd.read_csv('Dataset/diana_dataset_imputed.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")

    print(f"\nTarget variable (diabetes_status) distribution:")
    print(df['diabetes_status'].value_counts())
    print(f"\nTarget variable (diabetes_label) distribution:")
    print(df['diabetes_label'].value_counts())
    
    df['diabetes_binary'] = df['diabetes_status'].apply(
        lambda x: 'Yes' if x == 'Diabetic' else 'No'
    )
    
    print(f"\nBinary diabetes classification distribution:")
    print(df['diabetes_binary'].value_counts())
    
    target = 'diabetes_binary'
    
    exclude_cols = ['SEQN', 'diabetes_status', 'diabetes_label', 'diabetes_binary', 
                    'cycle', 'has_outlier', 'imputed']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"\nFeatures to analyze ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i}. {col}")
    
    print("\n" + "="*80)
    print("STEP 1: ENTROPY OF TARGET VARIABLE")
    print("="*80)
    
    target_entropy = entropy(df[target])
    print(f"\nTarget Variable: {target}")
    print(f"Entropy: {target_entropy:.4f}")

    value_counts = df[target].value_counts()
    total = len(df)
    print(f"\nDetailed Calculation:")
    for value, count in value_counts.items():
        prob = count / total
        print(f"  P({value}) = {count}/{total} = {prob:.4f}")
    
    print(f"\nEntropy = -Sum(p_i * log2(p_i))")
    entropy_terms = []
    for value, count in value_counts.items():
        prob = count / total
        term = -prob * log2(prob) if prob > 0 else 0
        entropy_terms.append(term)
        print(f"  - ({prob:.4f} * log2({prob:.4f})) = {term:.4f}")
    
    print(f"\nTotal Entropy = {' + '.join([f'{t:.4f}' for t in entropy_terms])} = {target_entropy:.4f}")
    

    print("\n" + "="*80)
    print("STEP 2: INFORMATION GAIN FOR EACH FEATURE")
    print("="*80)
    
    df_processed = df.copy()
    
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
    
    print(f"\nNumeric features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    
    print(f"\nDiscretizing continuous features into bins for IG calculation...")
    for feature in numeric_features:
        df_processed[feature + '_binned'] = discretize_continuous_feature(df_processed[feature])
    
    ig_results = []
    
    print("\nCalculating Information Gain...\n")
    for feature in feature_cols:
        if feature in numeric_features:
            feature_to_use = feature + '_binned'
        else:
            feature_to_use = feature
        
        try:
            ig = information_gain(df_processed, feature_to_use, target)
            ig_results.append({
                'Feature': feature,
                'Type': 'Numeric' if feature in numeric_features else 'Categorical',
                'Information_Gain': ig,
                'IG_Percentage': (ig / target_entropy * 100) if target_entropy > 0 else 0
            })
            print(f"  {feature:30s} - IG: {ig:.6f} ({ig/target_entropy*100:.2f}%)")
        except Exception as e:
            print(f"  {feature:30s} - Error: {str(e)}")
    
    print("\n" + "="*80)
    print("STEP 3: FEATURE RANKING BY INFORMATION GAIN")
    print("="*80)
    
    ig_df = pd.DataFrame(ig_results)
    ig_df = ig_df.sort_values('Information_Gain', ascending=False)
    ig_df['Rank'] = range(1, len(ig_df) + 1)
    
    ig_df = ig_df[['Rank', 'Feature', 'Type', 'Information_Gain', 'IG_Percentage']]
    
    print("\n" + "-"*90)
    print(f"{'Rank':<6} {'Feature':<30} {'Type':<12} {'Info Gain':<15} {'IG %':<10}")
    print("-"*90)
    
    for idx, row in ig_df.iterrows():
        print(f"{row['Rank']:<6} {row['Feature']:<30} {row['Type']:<12} "
              f"{row['Information_Gain']:<15.6f} {row['IG_Percentage']:<10.2f}")
    
    print("-"*90)
    
    print("\n" + "="*80)
    print("STEP 4: FEATURE SELECTION RECOMMENDATIONS")
    print("="*80)
    
    top_5 = ig_df.head(5)
    top_10 = ig_df.head(10)
    
    significant_features = ig_df[ig_df['IG_Percentage'] > 1.0]
    
    positive_ig = ig_df[ig_df['Information_Gain'] > 0.01]
    
    print("\n[SUMMARY STATISTICS]:")
    print(f"  - Total Features Analyzed: {len(ig_df)}")
    print(f"  - Target Entropy: {target_entropy:.4f}")
    print(f"  - Features with IG > 0.01: {len(positive_ig)}")
    print(f"  - Features with IG > 1% of target entropy: {len(significant_features)}")
    
    print("\n[TOP 5 FEATURES] (Recommended for Model):")
    for idx, row in top_5.iterrows():
        print(f"  {row['Rank']}. {row['Feature']:<30} - IG: {row['Information_Gain']:.6f} ({row['IG_Percentage']:.2f}%)")
    
    print("\n[TOP 10 FEATURES] (Extended Set):")
    for idx, row in top_10.iterrows():
        print(f"  {row['Rank']:2d}. {row['Feature']:<30} - IG: {row['Information_Gain']:.6f} ({row['IG_Percentage']:.2f}%)")
    
    print("\n[RECOMMENDATIONS]:")
    print(f"  1. **Essential Features (Top 5)**: These features have the highest predictive")
    print(f"     power for diabetes prediction:")
    for idx, row in top_5.iterrows():
        print(f"     - {row['Feature']}")
    
    print(f"\n  2. **Extended Feature Set (Top 10)**: For more comprehensive models:")
    for idx, row in top_10.iterrows():
        print(f"     - {row['Feature']}")
    
    print(f"\n  3. **Feature Insights**:")
    
    if len(ig_df) > 0:
        best_feature = ig_df.iloc[0]
        print(f"     - Best single feature: '{best_feature['Feature']}' with IG = {best_feature['Information_Gain']:.6f}")
        print(f"       (explains {best_feature['IG_Percentage']:.2f}% of total uncertainty)")
    
    top_numeric = ig_df[ig_df['Type'] == 'Numeric'].head(3)
    top_categorical = ig_df[ig_df['Type'] == 'Categorical'].head(3)
    
    if len(top_numeric) > 0:
        print(f"\n     - Top Numeric Features:")
        for idx, row in top_numeric.iterrows():
            print(f"       * {row['Feature']} (IG: {row['Information_Gain']:.6f})")
    
    if len(top_categorical) > 0:
        print(f"\n     - Top Categorical Features:")
        for idx, row in top_categorical.iterrows():
            print(f"       * {row['Feature']} (IG: {row['Information_Gain']:.6f})")
    
    print("\n  4. **Model Building Strategy**:")
    print("     - Start with the top 5 features for a baseline model")
    print("     - Add additional features from the top 10 and evaluate performance")
    print("     - Consider feature interactions and correlations")
    print("     - Use cross-validation to prevent overfitting")

if __name__ == "__main__":
    main()
