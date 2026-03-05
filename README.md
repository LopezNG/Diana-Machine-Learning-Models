# Diana ML Models - Neoron Research Module

This directory contains the original research and experimentation code for the DIANA diabetes prediction models. It includes data analysis, feature selection, model training notebooks, and utility scripts used during the research phase.

**Purpose**: Research & experimentation workspace for ML model development  
**Status**: Archive/Reference - Production code is in `Ian_ML/` directory  
**Last Updated**: January 2026

---

## Directory Structure

```
Neoron_ML/
├── README.md                          # This documentation
├── Datacheck.py                       # Data quality validation script
├── data_leakage_analysis.py           # Data leakage detection tool
├── feature_selection_entropy.py       # Feature selection using Information Gain
├── preprocessing_utils_fixed.py       # Data preprocessing utilities
├── logistic_regression.ipynb          # Logistic Regression model notebook
├── random_forest.ipynb                # Random Forest model notebook
├── xgboost_model.ipynb                # XGBoost model notebook
└── Dataset/                           # Research datasets
    ├── clustered_data.csv             # Clustered patient data
    ├── clustered_data_enhanced.csv    # Enhanced clustered data
    ├── diana_clustered_final.csv      # Final clustered dataset
    ├── diana_dataset_final.csv        # Final clean dataset
    ├── diana_dataset_imputed.csv      # Imputed dataset (primary source)
    ├── diana_training_data.csv        # Training subset
    └── diana_training_data_multi.csv  # Multi-class training data
```

---

## Python Scripts

### 1. Datacheck.py
**Purpose**: Data quality assessment and validation

**Functionality**:
- Loads `Dataset/diana_dataset_imputed.csv`
- Checks for missing values across all columns
- Generates statistical summaries for top 5 features (hba1c, fbs, bmi, hdl, triglycerides)
- Analyzes target variable distribution (diabetes_status)
- Detects outliers using IQR method

**Usage**:
```bash
python Datacheck.py
```

**Output**:
```
======================================================================
DIABETES DATASET - DATA QUALITY CHECK
======================================================================

[DATASET OVERVIEW]
Shape: 1376 rows x 25 columns

[MISSING VALUES]
No missing values found!

[TOP 5 FEATURES - DATA QUALITY]
HBA1C:
  Min: 3.90
  Max: 14.80
  Mean: 5.86
  ...
```

---

### 2. data_leakage_analysis.py
**Purpose**: Detect data leakage issues in the dataset

**Functionality**:
- Analyzes HbA1c as a diagnostic criterion (threshold: ≥6.5%)
- Checks if target variable is directly derived from HbA1c/FBS
- Performs class overlap analysis across features
- Detects synthetic/imputed data patterns

**Key Checks**:
- HbA1c ≥ 6.5 rule accuracy (warns if >95%)
- FBS ≥ 126 rule accuracy
- Feature overlap between diabetic/non-diabetic classes
- Statistical significance tests (t-tests)

**Usage**:
```bash
python data_leakage_analysis.py
```

**Critical Findings**:
```
*** ALERT: HbA1c >= 6.5 rule predicts diabetes with 98.45% accuracy! ***

WARNING: The target variable appears to be DIRECTLY DERIVED from HbA1c!
This is DATA LEAKAGE - HbA1c is the diagnostic criterion itself!
```

---

### 3. feature_selection_entropy.py
**Purpose**: Feature selection using Entropy and Information Gain

**Functionality**:
- Calculates entropy of the target variable
- Computes Information Gain (IG) for each feature
- Discretizes continuous features for IG calculation
- Ranks features by predictive power
- Provides feature selection recommendations

**Algorithm**:
1. Calculate target entropy: `H(Y) = -Σ p(y) log₂ p(y)`
2. For each feature, calculate weighted subset entropy
3. Information Gain: `IG(X,Y) = H(Y) - H(Y|X)`
4. Rank features by IG percentage

**Usage**:
```bash
python feature_selection_entropy.py
```

**Output**:
```
================================================================================
FEATURE SELECTION USING ENTROPY AND INFORMATION GAIN
================================================================================

STEP 1: ENTROPY OF TARGET VARIABLE
Target Variable: diabetes_binary
Entropy: 0.4521

STEP 2: INFORMATION GAIN FOR EACH FEATURE
  hba1c                          - IG: 0.412345 (91.23%)
  bmi                            - IG: 0.089234 (19.74%)
  ...

STEP 3: FEATURE RANKING BY INFORMATION GAIN
Rank   Feature                        Type         Info Gain       IG %
--------------------------------------------------------------------------------
1      hba1c                          Numeric      0.412345        91.23
2      bmi                            Numeric      0.089234        19.74
...
```

**Recommendations**:
- **Top 5 features**: Essential for baseline models
- **Top 10 features**: Extended feature set
- **IG > 1%**: Statistically significant features

---

### 4. preprocessing_utils_fixed.py
**Purpose**: Data preprocessing utilities with leakage prevention

**Functions**:

#### `load_and_prepare_data_fixed()`
Loads and prepares data with proper train/test splitting.

**Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_path` | str | `'Dataset/diana_dataset_imputed.csv'` | Path to dataset |
| `test_size` | float | `0.3` | Proportion for test set |
| `random_state` | int | `42` | Random seed |
| `exclude_diagnostic_features` | bool | `True` | Exclude HbA1c/FBS to prevent leakage |

**Returns**: `(X_train, X_test, y_train, y_test, feature_names)`

**Features Used** (when excluding diagnostic markers):
- Numeric: `age`, `bmi`, `hdl`, `triglycerides`, `total_cholesterol`, `systolic`, `diastolic`, `ldl`
- Categorical: `smoking_status`, `physical_activity`, `alcohol_use`

**Key Design**:
- Splits data BEFORE any preprocessing (prevents leakage)
- Returns raw data; preprocessing done in Pipeline
- Ensures scaler fit only on training data

#### `create_preprocessing_pipeline()`
Creates sklearn preprocessing pipeline.

```python
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])
```

#### `evaluate_with_cross_validation()`
Performs stratified k-fold cross-validation.

**Metrics computed**:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

#### `print_evaluation_metrics_fixed()`
Prints comprehensive evaluation metrics with warnings.

**Features**:
- Standard metrics (accuracy, precision, recall, F1, ROC-AUC)
- Confusion matrix
- Specificity and NPV
- Warning if accuracy ≥ 99% (indicates leakage/overfitting)

**Usage Example**:
```python
from preprocessing_utils_fixed import (
    load_and_prepare_data_fixed,
    create_preprocessing_pipeline,
    evaluate_with_cross_validation,
    print_evaluation_metrics_fixed
)

# Load data (no diagnostic features to prevent leakage)
X_train, X_test, y_train, y_test, features = load_and_prepare_data_fixed(
    exclude_diagnostic_features=True
)

# Create pipeline
preprocessor = create_preprocessing_pipeline(X_train, features)

# Use in model pipeline
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

model.fit(X_train, y_train)
```

---

## Jupyter Notebooks

### 1. logistic_regression.ipynb
**Purpose**: Logistic Regression model training and evaluation

**Contents**:
- Data loading with `preprocessing_utils_fixed`
- Logistic Regression with L1/L2 regularization
- Hyperparameter tuning (C parameter)
- Cross-validation evaluation
- ROC curve and precision-recall analysis
- Feature coefficient analysis

**Key Outputs**:
- Model accuracy, precision, recall, F1
- ROC-AUC score
- Feature importance (coefficients)
- Confusion matrix

### 2. random_forest.ipynb
**Purpose**: Random Forest model training and evaluation

**Contents**:
- Random Forest classifier with various estimators
- Hyperparameter tuning (n_estimators, max_depth, min_samples_split)
- Feature importance ranking
- Out-of-bag error analysis
- Cross-validation comparison

**Key Outputs**:
- Accuracy metrics
- Feature importance plot
- OOB error rate
- Tree visualization samples

### 3. xgboost_model.ipynb
**Purpose**: XGBoost model training and evaluation

**Contents**:
- XGBoost classifier implementation
- Hyperparameter tuning grid:
  - `n_estimators`: [100, 200, 300]
  - `max_depth`: [3, 4, 5]
  - `learning_rate`: [0.01, 0.05, 0.1]
  - `subsample`: [0.6, 0.8, 1.0]
  - `colsample_bytree`: [0.6, 0.8, 1.0]
- Early stopping
- Feature importance (gain, cover, weight)
- SHAP value analysis (if available)

**Key Outputs**:
- Best hyperparameters
- Cross-validation scores
- Feature importance plots
- Learning curves

---

## Datasets

### Primary Dataset: `diana_dataset_imputed.csv`
The main dataset used for model training.

**Characteristics**:
- **Samples**: ~1,376 postmenopausal women
- **Features**: 25+ clinical and lifestyle variables
- **Target**: diabetes_status (Normal/Pre-diabetic/Diabetic)
- **Source**: NHANES 2009-2023 (imputed)

**Key Features**:
| Feature | Type | Description |
|---------|------|-------------|
| `hba1c` | Numeric | Glycated hemoglobin (%) |
| `fbs` | Numeric | Fasting blood glucose (mg/dL) |
| `bmi` | Numeric | Body mass index (kg/m²) |
| `hdl` | Numeric | HDL cholesterol (mg/dL) |
| `ldl` | Numeric | LDL cholesterol (mg/dL) |
| `triglycerides` | Numeric | Triglycerides (mg/dL) |
| `age` | Numeric | Age in years |
| `diabetes_status` | Categorical | Target: Normal/Pre-diabetic/Diabetic |

### Dataset Variants

| File | Description |
|------|-------------|
| `clustered_data.csv` | K-means clustered patient segments |
| `clustered_data_enhanced.csv` | Clusters with additional metadata |
| `diana_clustered_final.csv` | Final clustered output |
| `diana_dataset_final.csv` | Cleaned dataset (before imputation) |
| `diana_training_data.csv` | Small training subset |
| `diana_training_data_multi.csv` | Multi-class training data |

---

## Usage Workflow

### Step 1: Data Quality Check
```bash
python Datacheck.py
```

### Step 2: Check for Data Leakage
```bash
python data_leakage_analysis.py
```

### Step 3: Feature Selection
```bash
python feature_selection_entropy.py
```

### Step 4: Model Training
Open Jupyter notebooks and run cells:
```bash
jupyter notebook logistic_regression.ipynb
jupyter notebook random_forest.ipynb
jupyter notebook xgboost_model.ipynb
```

---

## Important Notes

### Data Leakage Warning
⚠️ **Critical**: The dataset contains HbA1c and FBS which are diagnostic criteria. Using them gives ~99% accuracy but is clinically unrealistic for screening. For realistic screening models, use `exclude_diagnostic_features=True`.

### Production Code
This directory is for **research and experimentation**. The production-ready ML code is in:
- `Ian_ML/predict.py` - Prediction API
- `Ian_ML/train.py` - Model training pipeline
- `Ian_ML/server.py` - ML inference server

### Reproducibility
- Random state: 42
- All scripts use fixed random seeds where applicable
- Cross-validation uses StratifiedKFold

---

## Dependencies

```
pandas
numpy
scikit-learn
scipy
jupyter
matplotlib (for notebooks)
seaborn (for notebooks)
```

Install:
```bash
pip install pandas numpy scikit-learn scipy jupyter matplotlib seaborn
```

---

## Research Context

This module was developed as part of the DIANA (Diabetes Intelligent Assessment & Novel Analytics) system for diabetes risk prediction in postmenopausal women, based on the Ahlqvist et al. (2018) diabetes subtype classification.

**Key Research Questions**:
1. Can we predict diabetes risk without diagnostic markers (HbA1c/FBS)?
2. What are the most informative features for screening?
3. How do different ML models compare for this task?

**Baseline Performance** (without HbA1c/FBS):
- XGBoost: AUC ~0.67
- Random Forest: AUC ~0.65
- Logistic Regression: AUC ~0.63

---

## Contact & Attribution

- **Project**: DIANA Diabetes Prediction System
- **Research Team**: Neoron ML Research
- **Related**: See `Ian_ML/` for production implementation
