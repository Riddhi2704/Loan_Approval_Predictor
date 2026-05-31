import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'loan_approval_dataset.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'backend', 'model.joblib')
SCALER_PATH = os.path.join(BASE_DIR, 'backend', 'scaler.joblib')

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please download it from Kaggle and place it there.")
    df = pd.read_csv(DATA_PATH)
    # Strip any leading/trailing whitespaces in column names
    df.columns = df.columns.str.strip()
    return df

def clean_and_preprocess(df):
    # 1. Drop unnecessary columns
    if 'loan_id' in df.columns:
        df = df.drop(columns=['loan_id'])

    # 2. Handle missing values
    # For numerical columns, fill with median
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # For categorical columns, fill with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # 3. Handle Outliers for continuous variables using IQR method
    continuous_cols = ['income_annum', 'loan_amount', 'residential_assets_value', 
                       'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']
    
    for col in continuous_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            # Capping the outliers
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

    # 4. Feature Engineering
    # Strip spaces from categorical features
    for col in cat_cols:
        df[col] = df[col].str.strip()
        
    # Convert categorical to numerical using LabelEncoder or Mapping
    # Assuming standard columns like 'education' (Graduate/Not Graduate), 'self_employed' (Yes/No), 'loan_status' (Approved/Rejected)
    label_encoders = {}
    for col in ['education', 'self_employed', 'loan_status']:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
            
    return df, label_encoders

def main():
    print("Loading dataset...")
    try:
        df = load_data()
    except FileNotFoundError as e:
        print(e)
        return

    print("Cleaning and preprocessing data...")
    df, label_encoders = clean_and_preprocess(df)

    print("Splitting data into features and target...")
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']

    print("Performing train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training XGBoost model with hyperparameter tuning...")
    xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    
    # Grid search for hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
    
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_scaled, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best parameters found: {grid_search.best_params_}")
    
    print("Evaluating model...")
    y_pred = best_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Saving model and scaler...")
    # ensure backend dir exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("Model and scaler saved successfully to backend/ directory.")

if __name__ == "__main__":
    main()
