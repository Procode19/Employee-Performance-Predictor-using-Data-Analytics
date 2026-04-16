# train_employee_perf.py
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load data
df = pd.read_csv("data/employee_features.csv")
y = df['perf_band_next']
X = df.drop(columns=['perf_band_next', 'employee_id'])  # Drop ID to avoid leakage

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Preprocessing
cat_cols = X.select_dtypes(include='object').columns
num_cols = X.select_dtypes(include='number').columns

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

pre = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
])

# Model pipeline
pipe = Pipeline([
    ('pre', pre),
    ('clf', RandomForestClassifier(class_weight='balanced', random_state=42))
])

# Hyperparameter tuning
param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [10, 20, None],
    'clf__min_samples_leaf': [1, 2, 4]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gs = GridSearchCV(pipe, param_grid, scoring='f1_macro', cv=cv, n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)

print("Best CV F1_macro:", gs.best_score_)
print("Best params:", gs.best_params_)

# Evaluate on test
y_pred = gs.best_estimator_.predict(X_test)
print("Test Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(gs.best_estimator_, "models/employee_perf_model.pkl")
print("Model saved to models/employee_perf_model.pkl")