import pandas as pd
import numpy as np

def main():
    print("="*70)
    print("DIABETES DATASET - DATA QUALITY CHECK")
    print("="*70)
    
    # Load dataset
    df = pd.read_csv('Dataset/diana_dataset_imputed.csv')
    
    print(f"\n[DATASET OVERVIEW]")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    
    # Check for missing values
    print(f"\n[MISSING VALUES]")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("Columns with missing values:")
        for col, count in missing[missing > 0].items():
            print(f"  {col}: {count} ({count/len(df)*100:.2f}%)")
    else:
        print("No missing values found!")

    print(f"\n[TOP 5 FEATURES - DATA QUALITY]")
    top_features = ['hba1c', 'fbs', 'bmi', 'hdl', 'triglycerides']
    
    for feature in top_features:
        if feature in df.columns:
            print(f"\n{feature.upper()}:")
            print(f"  Min: {df[feature].min():.2f}")
            print(f"  Max: {df[feature].max():.2f}")
            print(f"  Mean: {df[feature].mean():.2f}")
            print(f"  Median: {df[feature].median():.2f}")
            print(f"  Std Dev: {df[feature].std():.2f}")
            print(f"  Missing: {df[feature].isnull().sum()}")
    
    print(f"\n[TARGET VARIABLE DISTRIBUTION]")
    print(df['diabetes_status'].value_counts())
    print(f"\nClass Balance:")
    for status, count in df['diabetes_status'].value_counts().items():
        print(f"  {status}: {count} ({count/len(df)*100:.2f}%)")
    
    print(f"\n[OUTLIER CHECK - Top Features]")
    for feature in top_features:
        if feature in df.columns:
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[feature] < (Q1 - 1.5 * IQR)) | 
                       (df[feature] > (Q3 + 1.5 * IQR))).sum()
            print(f"  {feature}: {outliers} outliers ({outliers/len(df)*100:.2f}%)")
    
    print("\n" + "="*70)
    print("DATA CHECK COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()
