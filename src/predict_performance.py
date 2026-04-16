# predict_performance.py
import pandas as pd
import joblib

# Load model
model = joblib.load("models/employee_perf_model.pkl")

# Example new employee data (replace with actual input)
new_employee = pd.DataFrame({
    'age': [30],
    'gender': ['Male'],
    'education': ['Master'],
    'experience_years': [5],
    'department': ['Engineering'],
    'job_level': ['Mid'],
    'manager_tenure': [3],
    'projects_count': [10],
    'avg_task_delay_days': [2.0],
    'on_time_delivery_rate': [0.85],
    'bug_count': [5],
    'code_review_score': [4.0],
    'qa_defect_density': [0.1],
    'story_points_completed': [200],
    'billable_hours_ratio': [0.9],
    'training_hours': [20.0],
    'certifications_count': [2],
    'internal_hackathons_participated': [1],
    'sick_days': [2],
    'unplanned_absences': [1],
    'avg_login_hours': [8.5],
    'peer_feedback_score': [4.2],
    'manager_score': [4.5],
    'kudos_count': [5],
    'promotions_in_2y': [1],
    'salary_percentile_band': ['Medium']
})

# Predict
prediction = model.predict(new_employee)
print("Predicted Performance Band:", prediction[0])

# For probabilities
probabilities = model.predict_proba(new_employee)
print("Probabilities:", probabilities)