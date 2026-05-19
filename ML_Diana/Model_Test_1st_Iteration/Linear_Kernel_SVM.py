import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, cohen_kappa_score,
    classification_report, confusion_matrix
)
from scipy.stats import loguniform
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================
# 1. LOAD DATA & EXCLUDE COLUMNS
# ============================================================
print("=" * 65)
print("1. LOADING DATA & EXCLUDING COLUMNS")
print("=" * 65)

df = pd.read_csv('diana_dataset_imputed.csv')
print(f"Original dataset shape: {df.shape}")

columns_to_exclude = ['menopausal_status', 'imputed', 'diastolic', 'systolic']
df.drop(columns=columns_to_exclude, inplace=True)
print(f"After excluding {columns_to_exclude}: {df.shape}")

df.drop(columns=['SEQN', 'diabetes_status'], inplace=True)
print(f"After dropping SEQN & diabetes_status: {df.shape}")
print(f"\nRemaining columns: {list(df.columns)}")
print(f"\nTarget distribution:\n{df['diabetes_label'].value_counts().sort_index()}")

# ============================================================
# 2. FEATURE SPLITTING  (Cycle Column -> two numeric columns)
# ============================================================
print("\n" + "=" * 65)
print("2. FEATURE SPLITTING - Cycle Column")
print("=" * 65)

df[['cycle_start', 'cycle_end']] = (
    df['cycle'].str.split('-', expand=True).astype(int)
)
df.drop(columns=['cycle'], inplace=True)
print("Cycle column split into cycle_start and cycle_end")
print(f"  cycle_start unique: {sorted(df['cycle_start'].unique())}")
print(f"  cycle_end   unique: {sorted(df['cycle_end'].unique())}")

# ============================================================
# 3. ROUNDING / PRECISION REDUCTION
# ============================================================
print("\n" + "=" * 65)
print("3. ROUNDING / PRECISION REDUCTION")
print("=" * 65)

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numerical_cols = [c for c in numerical_cols if c not in ['diabetes_label']]
print(f"Numerical columns to round: {numerical_cols}")

for col in numerical_cols:
    df[col] = df[col].round(2)

print("All numerical features rounded to 2 decimal places.")
print(f"\nSample after rounding:\n{df[numerical_cols].head(3).to_string()}")

# ============================================================
# 4. ENCODE CATEGORICAL VARIABLES (Ordinal Encoding)
# ============================================================
print("\n" + "=" * 65)
print("4. ENCODING CATEGORICAL VARIABLES")
print("=" * 65)

smoking_map    = {'Never': 0, 'Former': 1, 'Current': 2, 'Unknown': 3}
activity_map   = {'Sedentary': 0, 'Moderate': 1, 'Active': 2, 'Unknown': 3}
alcohol_map    = {'Light': 0, 'Moderate': 1, 'Heavy': 2}

categorical_mappings = {
    'smoking_status':    smoking_map,
    'physical_activity': activity_map,
    'alcohol_use':       alcohol_map,
}

for col, mapping in categorical_mappings.items():
    df[col] = df[col].map(mapping)
    print(f"  {col}: {mapping}")

# ============================================================
# 5. SEPARATE FEATURES, TARGET & OUTLIER FLAG
# ============================================================
print("\n" + "=" * 65)
print("5. SEPARATING FEATURES, TARGET & OUTLIER FLAG")
print("=" * 65)

has_outlier = df['has_outlier'].copy()
df.drop(columns=['has_outlier'], inplace=True)

X = df.drop(columns=['diabetes_label'])
y = df['diabetes_label']

print(f"Features shape : {X.shape}")
print(f"Target shape   : {y.shape}")
print(f"Feature list   : {list(X.columns)}")

# ============================================================
# 6. STRATIFIED 60 / 20 / 20 SPLIT
# ============================================================
print("\n" + "=" * 65)
print("6. STRATIFIED 60-20-20 SPLIT")
print("=" * 65)

X_train, X_temp, y_train, y_temp, out_train, out_temp = train_test_split(
    X, y, has_outlier,
    test_size=0.40, random_state=RANDOM_STATE, stratify=y
)

X_val, X_test, y_val, y_test, out_val, out_test = train_test_split(
    X_temp, y_temp, out_temp,
    test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
)

for name, xs, ys in [('Training', X_train, y_train),
                      ('Validation', X_val, y_val),
                      ('Test', X_test, y_test)]:
    pct = xs.shape[0] / len(X) * 100
    print(f"  {name:12s}: {xs.shape[0]:>5} samples ({pct:5.1f}%)")
    print(f"{'':16s}Class dist -> {dict(ys.value_counts().sort_index())}")

# ============================================================
# 7. CLAMP OUTLIERS (identified by has_outlier flag)
# ============================================================
print("\n" + "=" * 65)
print("7. CLAMPING OUTLIERS (has_outlier flag + IQR bounds)")
print("=" * 65)

num_feats = X_train.select_dtypes(include=[np.number]).columns.tolist()
clean_train = X_train[~out_train]

lower_bounds = {}
upper_bounds = {}
for col in num_feats:
    Q1  = clean_train[col].quantile(0.25)
    Q3  = clean_train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bounds[col] = Q1 - 1.5 * IQR
    upper_bounds[col] = Q3 + 1.5 * IQR

print(f"IQR bounds computed from {len(clean_train)} non-outlier training rows")
print(f"Outlier-flagged rows -> train: {out_train.sum()}, "
      f"val: {out_val.sum()}, test: {out_test.sum()}")


def clamp_outliers(X_set, outlier_flags, lb, ub, cols):
    X_c = X_set.copy()
    mask = outlier_flags.values
    for col in cols:
        X_c.loc[mask, col] = X_c.loc[mask, col].clip(lower=lb[col], upper=ub[col])
    return X_c


X_train = clamp_outliers(X_train, out_train, lower_bounds, upper_bounds, num_feats)
X_val   = clamp_outliers(X_val,   out_val,   lower_bounds, upper_bounds, num_feats)
X_test  = clamp_outliers(X_test,  out_test,  lower_bounds, upper_bounds, num_feats)
print("Outlier clamping applied to all sets.")

# ============================================================
# 8. NORMALIZATION - StandardScaler (Standardization)
# ============================================================
print("\n" + "=" * 65)
print("8. NORMALIZATION - StandardScaler")
print("=" * 65)

scaler = StandardScaler()
X_train_sc = pd.DataFrame(scaler.fit_transform(X_train),
                           columns=X_train.columns, index=X_train.index)
X_val_sc   = pd.DataFrame(scaler.transform(X_val),
                           columns=X_val.columns,   index=X_val.index)
X_test_sc  = pd.DataFrame(scaler.transform(X_test),
                           columns=X_test.columns,  index=X_test.index)

print(f"Scaler fitted on training set ({X_train_sc.shape[0]} samples)")
print(f"  Training mean ~ {X_train_sc.mean().mean():.6f}  (expect ~0)")
print(f"  Training std  ~ {X_train_sc.std().mean():.6f}  (expect ~1)")

# ============================================================
# 9. HYPERPARAMETER TUNING - RandomizedSearchCV
#    Stratified 10-Fold CV  |  Linear Kernel SVM
# ============================================================
print("\n" + "=" * 65)
print("9. HYPERPARAMETER TUNING - RandomizedSearchCV (Linear Kernel SVM)")
print("=" * 65)

param_dist = {
    'C':             loguniform(1e-3, 1e3),
    'kernel':        ['linear'],
    'class_weight':  ['balanced', None],
    'tol':           loguniform(1e-5, 1e-2),
}

svc_base = SVC(kernel='linear', probability=True, random_state=RANDOM_STATE)

cv_strat = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

search = RandomizedSearchCV(
    estimator=svc_base,
    param_distributions=param_dist,
    n_iter=100,
    cv=cv_strat,
    scoring='f1_weighted',
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
    return_train_score=True,
)

print("Starting search  (100 iterations x 10 folds = 1,000 fits) ...")
search.fit(X_train_sc, y_train)

best_svc = search.best_estimator_
print(f"\n  Best CV F1 (weighted): {search.best_score_:.4f}")
print(f"\nBest hyper-parameters:")
for k, v in search.best_params_.items():
    print(f"  {k:22s}: {v}")

# ============================================================
# EVALUATION HELPER
# ============================================================
def evaluate(model, X_data, y_data, set_name):
    print(f"\n{'=' * 65}")
    print(f"EVALUATION - {set_name}")
    print(f"{'=' * 65}")

    y_pred  = model.predict(X_data)
    y_proba = model.predict_proba(X_data)

    acc   = accuracy_score(y_data, y_pred)
    prec  = precision_score(y_data, y_pred, average='weighted', zero_division=0)
    rec   = recall_score(y_data, y_pred, average='weighted', zero_division=0)
    f1    = f1_score(y_data, y_pred, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(y_data, y_pred)
    auc   = roc_auc_score(y_data, y_proba, multi_class='ovr', average='weighted')

    print(f"\n  Accuracy      : {acc:.4f}")
    print(f"  Precision     : {prec:.4f}")
    print(f"  Recall        : {rec:.4f}")
    print(f"  F1-Score      : {f1:.4f}")
    print(f"  ROC-AUC       : {auc:.4f}")
    print(f"  Cohen's Kappa : {kappa:.4f}")

    print(f"\n  Classification Report:\n")
    print(classification_report(y_data, y_pred, zero_division=0))

    print(f"  Confusion Matrix:")
    print(confusion_matrix(y_data, y_pred))

    return {'Accuracy': acc, 'Precision': prec, 'Recall': rec,
            'F1-Score': f1, 'ROC-AUC': auc, "Cohen's Kappa": kappa}

# ============================================================
# 10. VALIDATION SET EVALUATION
# ============================================================
val_metrics = evaluate(best_svc, X_val_sc, y_val, "Validation Set")

# ============================================================
# 11. TEST SET EVALUATION
# ============================================================
print("\n" + "=" * 65)
print("11. TEST SET (Final Model Performance)")
print("=" * 65)

n_sv = best_svc.n_support_
print(f"\nLinear Kernel SVM trained with C = {best_svc.C:.6f}")
print(f"Number of support vectors per class: {n_sv}")
print(f"Total support vectors: {n_sv.sum()}  "
      f"({n_sv.sum() / X_train_sc.shape[0] * 100:.1f}% of training samples)")

# Display the learned weight matrix (coef_) dimensions
print(f"Weight matrix shape (coef_): {best_svc.coef_.shape}")
print(f"  -> {best_svc.coef_.shape[0]} decision functions x "
      f"{best_svc.coef_.shape[1]} features\n")

test_metrics = evaluate(best_svc, X_test_sc, y_test, "Test Set")

# ============================================================
# 12. FINAL SUMMARY TABLE
# ============================================================
print("\n" + "=" * 65)
print("12. FINAL SUMMARY")
print("=" * 65)

summary_df = pd.DataFrame({'Validation': val_metrics,
                            'Test': test_metrics}).T
print(f"\n{summary_df.round(4).to_string()}")

print(f"\nBest Hyperparameters:")
for k, v in search.best_params_.items():
    print(f"  {k:22s}: {v}")
print(f"\n  Support Vectors: {best_svc.n_support_.sum()}")
print(f"  Random State   : {RANDOM_STATE}")
print("\n" + "=" * 65)
print("DONE")
print("=" * 65)
