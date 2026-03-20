# Loan Payback Analysis

## Project Overview
This is a machine learning classification project that predicts whether loans will be paid back. The project uses ensemble learning techniques with multiple base models (XGBoost, CatBoost, Random Forest) combined using stacking to maximize prediction accuracy.

## Dataset
- **Source:** Kaggle Playground Series Season 5, Episode 11
- **Training samples:** Various loan records with features and target variable
- **Test samples:** Unlabeled loans for prediction
- **Target:** `loan_paid_back` (binary classification: whether the loan was paid back)

## Key Data Columns

### Numerical Features
- `annual_income` - Annual income of the borrower
- `debt_to_income_ratio` - Ratio of debt to income
- `credit_score` - Credit score
- `loan_amount` - Amount borrowed
- `interest_rate` - Interest rate on the loan

### Categorical Features
- `gender` - Borrower's gender
- `marital_status` - Marital status
- `education_level` - Education level
- `employment_status` - Employment status
- `loan_purpose` - Purpose of the loan
- `grade_subgrade` - Grade and subgrade classification

## Feature Engineering
The project creates advanced features to improve model performance:

- **`income_to_loan_ratio`** - Ratio of income to loan amount
- **`debt_burden`** - Actual debt amount (income × debt-to-income ratio)
- **`affordability_ratio`** - Monthly affordability based on loan payment
- **`credit_income_ratio`** - Credit score normalized by income
- **`risk_score`** - Composite risk score combining multiple factors (30% debt-to-income, 30% credit score deviation, 20% interest rate, 20% loan-to-income ratio)
- **`employment_stability`** - Numerical encoding of employment security (Unemployed: 0, Student: 1, Self-employed: 2, Employed: 3, Retired: 2)
- **`education_num`** - Numerical encoding of education level (High School: 1, Other: 2, Bachelor's: 3, Master's: 4, PhD: 5)

## Models Used

### Base Models
1. **XGBoost** - Gradient boosting model with hyperparameter optimization
2. **CatBoost** - Gradient boosting optimized for categorical features
3. **Random Forest** - Ensemble of decision trees

### Ensemble Approach
**Stacking Ensemble** - Logistic Regression meta-model that combines predictions from all three base models

## Model Training Flow

### 1. Data Preprocessing
- Feature engineering to create derived features
- Label encoding for categorical variables
- Standard scaling for numerical features
- Train-validation split (80-20)

### 2. Hyperparameter Tuning
- Grid search with 3-fold cross-validation for each base model
- ROC-AUC scoring metric

**XGBoost Grid Search Parameters:**
- n_estimators: [50, 100, 200]
- max_depth: [4, 6, 10]
- learning_rate: [0.01, 0.001, 0.1]
- subsample: [0.5, 0.8, 1.0]
- colsample_bytree: [0.8, 1.0]

**CatBoost Grid Search Parameters:**
- iterations: [50, 100, 200]
- depth: [4, 6, 8]
- learning_rate: [0.01, 0.1]

**Random Forest Grid Search Parameters:**
- n_estimators: [100]
- max_depth: [8]
- min_samples_split: [5]
- min_samples_leaf: [2]

### 3. Ensemble Creation
- Base model predictions combined as meta-features
- Meta-model (Logistic Regression) trained on validation set
- Final retraining on full training data with best parameters

### 4. Test Set Predictions
- Test set predictions generated using all best-parameter models
- Final ensemble predictions output as probabilities (0-1)

## Output

**submission.csv** - Contains:
- `id` - Loan ID from test set
- `loan_paid_back` - Predicted probability of loan payback (0-1)

**Summary Statistics:**
- Prediction probability range
- Mean probability value

## Performance Metrics
- Validation AUC scores for each base model and ensemble
- ROC-AUC used as primary evaluation metric for model selection

## Dependencies

### Data Processing
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computations

### Machine Learning
- `scikit-learn` - ML models, preprocessing, metrics
- `xgboost` - XGBoost classifier
- `catboost` - CatBoost classifier

### Visualization
- `matplotlib` - Plotting library
- `seaborn` - Statistical visualization

## Usage

1. Load the training and test CSV files from Kaggle
2. Run the notebook cells in sequence to:
   - Load and explore data
   - Perform feature engineering
   - Preprocess data
   - Perform hyperparameter tuning with grid search
   - Train base models on full training data
   - Generate predictions on test set
3. Submit `submission.csv` for evaluation

## Notes
- Data is preprocessed consistently between train and test sets
- Label encoding ensures same mappings for categorical variables
- Scaling is fitted on training data only to avoid data leakage
- All models use consistent random states for reproducibility
- Meta-features are created by stacking base model probability predictions
