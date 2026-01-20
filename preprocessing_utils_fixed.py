import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def load_and_prepare_data_fixed(dataset_path='Dataset/diana_dataset_imputed.csv', 
                                test_size=0.3, 
                                random_state=42,
                                exclude_diagnostic_features=True):

    print("="*80)
    print("FIXED DATA PREPROCESSING (NO DATA LEAKAGE)")
    print("="*80)
    
    # Load dataset
    print(f"\n[1] Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"    Dataset shape: {df.shape}")
    
    # Create binary target variable
    print("\n[2] Creating binary target variable")
    df['diabetes_binary'] = df['diabetes_status'].apply(
        lambda x: 1 if x == 'Diabetic' else 0
    )
    
    target_counts = df['diabetes_binary'].value_counts()
    print(f"    Class 0 (Non-Diabetic): {target_counts[0]} ({target_counts[0]/len(df)*100:.2f}%)")
    print(f"    Class 1 (Diabetic): {target_counts[1]} ({target_counts[1]/len(df)*100:.2f}%)")
    
    # Define features
    if exclude_diagnostic_features:
        print(f"\n[3] EXCLUDING HbA1c and FBS (diagnostic criteria - cause data leakage)")
        feature_names = [
            'age', 'bmi', 'hdl', 'triglycerides', 'total_cholesterol', 
            'systolic', 'diastolic', 'ldl',
            'smoking_status', 'physical_activity', 'alcohol_use'
        ]
        print(f"    Using {len(feature_names)} features (no diagnostic markers)")
    else:
        print(f"\n[3] WARNING: Including HbA1c and FBS (will give unrealistic accuracy)")
        feature_names = [
            'age', 'hba1c', 'fbs', 'bmi', 'hdl', 'triglycerides',
            'total_cholesterol', 'systolic', 'diastolic', 'ldl',
            'smoking_status', 'physical_activity', 'alcohol_use'
        ]
        print(f"    Using {len(feature_names)} features (with diagnostic markers)")
    
    for i, feat in enumerate(feature_names, 1):
        print(f"    {i:2d}. {feat}")
    
    # Separate features and target
    X = df[feature_names].copy()
    y = df['diabetes_binary'].copy()
    
    # CRITICAL: Split BEFORE any preprocessing
    print(f"\n[4] Splitting data BEFORE preprocessing (prevents leakage)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"    Training set: {X_train.shape[0]} samples")
    print(f"    Testing set: {X_test.shape[0]} samples")
    print(f"    Training - Diabetic: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.2f}%)")
    print(f"    Testing - Diabetic: {y_test.sum()} ({y_test.sum()/len(y_test)*100:.2f}%)")
    
    print("\n[5] Returning RAW data (preprocessing will be done in Pipeline)")
    print("    - This ensures scaler and encoders are fit ONLY on training data")
    print("    - Prevents test set information from leaking into training")
    
    print("\n" + "="*80)
    
    return X_train, X_test, y_train, y_test, feature_names


def create_preprocessing_pipeline(X_train, feature_names):

    # Identify categorical and numerical features
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"\n[PIPELINE SETUP]")
    print(f"  Categorical features ({len(categorical_features)}): {categorical_features}")
    print(f"  Numerical features ({len(numerical_features)}): {numerical_features}")
    
    # Create preprocessing for numerical features
    numerical_transformer = StandardScaler()
    
    # Create preprocessing for categorical features  
    # Use OneHotEncoder instead of LabelEncoder for better ML practice
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    
    # Combine preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    return preprocessor


def evaluate_with_cross_validation(model, X, y, cv=5):

    from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    print(f"\n[CROSS-VALIDATION] Using {cv}-Fold Stratified CV")
    
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0),
        'f1': make_scorer(f1_score, zero_division=0),
        'roc_auc': make_scorer(roc_auc_score, needs_proba=True)
    }
    
    results = {}
    
    for metric_name, scorer in scoring.items():
        try:
            if metric_name == 'roc_auc':
                # ROC-AUC needs probability predictions
                scores = cross_val_score(model, X, y, cv=cv_strategy, scoring='roc_auc')
            else:
                scores = cross_val_score(model, X, y, cv=cv_strategy, scoring=scorer)
            
            results[metric_name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
            
            print(f"  {metric_name.capitalize():12s}: {scores.mean():.4f} (+/- {scores.std():.4f})")
        except Exception as e:
            print(f"  {metric_name.capitalize():12s}: Could not compute ({str(e)[:30]})")
    
    return results


def print_evaluation_metrics_fixed(y_true, y_pred, y_pred_proba, model_name):
    """
    Print evaluation metrics with warnings about unrealistic performance.
    """
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                 f1_score, roc_auc_score, confusion_matrix,
                                 classification_report)
    
    print("\n" + "="*80)
    print(f"EVALUATION RESULTS - {model_name}")
    print("="*80)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    cm = confusion_matrix(y_true, y_pred)
    
    # Print metrics
    print("\n[PERFORMANCE METRICS]")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    # Warning for unrealistic performance
    if accuracy >= 0.99:
        print("\n  *** WARNING: Accuracy >= 99% is unrealistic for medical data! ***")
        print("  This suggests data leakage or overfitting.")
    
    # Print confusion matrix
    print("\n[CONFUSION MATRIX]")
    print(f"                Predicted")
    print(f"              Neg    Pos")
    print(f"  Actual Neg  {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"         Pos  {cm[1,0]:4d}   {cm[1,1]:4d}")
    
    # Additional metrics
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    print("\n[ADDITIONAL METRICS]")
    print(f"  True Positives:  {tp}")
    print(f"  True Negatives:  {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  Specificity:     {specificity:.4f}")
    print(f"  NPV:             {npv:.4f}")
    
    # Classification report
    print("\n[CLASSIFICATION REPORT]")
    print(classification_report(y_true, y_pred, 
                               target_names=['Non-Diabetic', 'Diabetic'],
                               zero_division=0))
    
    print("="*80)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }
