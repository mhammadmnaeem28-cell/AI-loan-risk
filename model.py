import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# 1. Loading the real dataset
# Dataset load kar rahe hain
df = pd.read_csv('loan_data.csv.csv')

# 2. Data Cleaning & Balancing
# Ghalat age aur employment data ko remove kar rahe hain
df = df[
    (df['person_age'] <= 100) & 
    (
        (df['person_emp_length'] <= (df['person_age'] - 18)) | 
        df['person_emp_length'].isna()
    )
]

# Features aur Target (loan_status) ko alag kar rahe hain
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# 3. Preprocessing Pipeline (Easy and Professional)
# Numbers aur Text ke liye alag alag processors
num_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
cat_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)])

# 4. Training Random Forest with Class Balancing
# class_weight='balanced' use kiya hai taake data balance ho jaye
model_pipeline = Pipeline([
    ('pre', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

# Saving the model and accuracy
# Model aur accuracy ko pickle file mein save kar rahe hain
with open('loan_model.pkl', 'wb') as f:
    pickle.dump({'model': model_pipeline, 'acc': model_pipeline.score(X_test, y_test)}, f)

print("Model training complete. File 'loan_model.pkl' generated.")