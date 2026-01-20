import pandas as pd
import numpy as np
from scipy import stats

def analyze_data_leakage():
    print("="*80)
    print("DATA LEAKAGE ANALYSIS")
    print("="*80)
    
    # Load dataset
    df = pd.read_csv('Dataset/diana_dataset_imputed.csv')
    
    df['diabetes_binary'] = df['diabetes_status'].apply(
        lambda x: 1 if x == 'Diabetic' else 0
    )
    
    print(f"\n[1] Dataset Overview")
    print(f"    Total samples: {len(df)}")
    print(f"    Diabetic: {df['diabetes_binary'].sum()} ({df['diabetes_binary'].sum()/len(df)*100:.2f}%)")
    print(f"    All rows imputed: {df['imputed'].all()}")
    
    print(f"\n[2] HbA1c Analysis (Clinical Diagnostic Criterion)")
    print(f"    Clinical threshold: HbA1c >= 6.5% indicates diabetes")
    
    diabetic = df[df['diabetes_binary'] == 1]['hba1c']
    non_diabetic = df[df['diabetes_binary'] == 0]['hba1c']
    
    print(f"\n    Diabetic patients:")
    print(f"      Mean HbA1c: {diabetic.mean():.2f}")
    print(f"      Min HbA1c:  {diabetic.min():.2f}")
    print(f"      Max HbA1c:  {diabetic.max():.2f}")
    print(f"      % with HbA1c >= 6.5: {(diabetic >= 6.5).sum() / len(diabetic) * 100:.2f}%")
    
    print(f"\n    Non-diabetic patients:")
    print(f"      Mean HbA1c: {non_diabetic.mean():.2f}")
    print(f"      Min HbA1c:  {non_diabetic.min():.2f}")
    print(f"      Max HbA1c:  {non_diabetic.max():.2f}")
    print(f"      % with HbA1c >= 6.5: {(non_diabetic >= 6.5).sum() / len(non_diabetic) * 100:.2f}%")

    print(f"\n[3] CRITICAL DATA LEAKAGE DETECTION")

    threshold = 6.5
    predicted_by_hba1c = (df['hba1c'] >= threshold).astype(int)
    accuracy_hba1c = (predicted_by_hba1c == df['diabetes_binary']).mean()
    
    print(f"\n    *** ALERT: HbA1c >= 6.5 rule predicts diabetes with {accuracy_hba1c*100:.2f}% accuracy! ***")
    
    if accuracy_hba1c >= 0.95:
        print(f"\n    WARNING: The target variable appears to be DIRECTLY DERIVED from HbA1c!")
        print(f"    This is DATA LEAKAGE - HbA1c is the diagnostic criterion itself!")
        print(f"    Models are essentially just learning the clinical threshold.")
    
    fbs_threshold = 126
    predicted_by_fbs = (df['fbs'] >= fbs_threshold).astype(int)
    accuracy_fbs = (predicted_by_fbs == df['diabetes_binary']).mean()
    
    print(f"\n    FBS >= 126 rule predicts diabetes with {accuracy_fbs*100:.2f}% accuracy")
    
    print(f"\n[4] Class Overlap Analysis")
    
    features = ['hba1c', 'fbs', 'bmi', 'hdl', 'triglycerides']
    
    for feat in features:
        diabetic_vals = df[df['diabetes_binary'] == 1][feat]
        non_diabetic_vals = df[df['diabetes_binary'] == 0][feat]
        
        overlap_min = max(diabetic_vals.min(), non_diabetic_vals.min())
        overlap_max = min(diabetic_vals.max(), non_diabetic_vals.max())
        
        if overlap_max > overlap_min:
            overlap_pct = len(df[(df[feat] >= overlap_min) & (df[feat] <= overlap_max)]) / len(df) * 100
        else:
            overlap_pct = 0
        
        print(f"\n    {feat}:")
        print(f"      Diabetic range:     [{diabetic_vals.min():.2f}, {diabetic_vals.max():.2f}]")
        print(f"      Non-diabetic range: [{non_diabetic_vals.min():.2f}, {non_diabetic_vals.max():.2f}]")
        print(f"      Overlap: {overlap_pct:.2f}% of samples")
        
        t_stat, p_value = stats.ttest_ind(diabetic_vals, non_diabetic_vals)
        print(f"      T-test p-value: {p_value:.2e} {'(HIGHLY SEPARATED)' if p_value < 0.001 else ''}")
    
    print(f"\n[5] Data Quality Issues")
    
    print(f"\n    Imputation analysis:")
    print(f"      All rows marked as imputed: {df['imputed'].all()}")
    print(f"      This suggests the dataset is heavily synthetic/imputed")
    
    print(f"\n    Numeric precision check (suspicious if too many decimals):")
    for col in ['bmi', 'total_cholesterol', 'hdl']:
        sample_val = df[col].iloc[0]
        decimal_places = len(str(sample_val).split('.')[-1]) if '.' in str(sample_val) else 0
        print(f"      {col}: {decimal_places} decimal places (e.g., {sample_val})")
    
    print(f"\n      NOTE: Real clinical measurements typically have 1-2 decimal places")
    print(f"      Many decimal places suggest synthetic/imputed data")
    
    print(f"\n" + "="*80)
    print("SUMMARY OF FINDINGS")
    print("="*80)
    
    issues = []
    
    if accuracy_hba1c >= 0.95:
        issues.append("HbA1c perfectly predicts diabetes (target leakage)")
    
    if df['imputed'].all():
        issues.append("All data is imputed/synthetic")
    
    if len(issues) == 0:
        print("\n  No critical data leakage issues detected.")
    else:
        print("\n  CRITICAL ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")
    
    print(f"\n" + "="*80)
    
    return df

if __name__ == "__main__":
    df = analyze_data_leakage()
