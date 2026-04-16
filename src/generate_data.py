import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)

# Number of employees
n = 1000

# Generate synthetic data
data = {
    'employee_id': range(1, n+1),
    'age': np.random.randint(22, 60, n),
    'gender': np.random.choice(['Male', 'Female'], n),
    'education': np.random.choice(['Bachelor', 'Master', 'PhD'], n, p=[0.5, 0.4, 0.1]),
    'experience_years': np.random.randint(0, 40, n),
    'department': np.random.choice(['Engineering', 'Sales', 'HR', 'Finance', 'Marketing'], n),
    'job_level': np.random.choice(['Junior', 'Mid', 'Senior'], n),
    'manager_tenure': np.random.randint(1, 20, n),
    'projects_count': np.random.randint(1, 20, n),
    'avg_task_delay_days': np.random.uniform(-5, 30, n),
    'on_time_delivery_rate': np.random.uniform(0.5, 1.0, n),
    'bug_count': np.random.randint(0, 50, n),
    'code_review_score': np.random.uniform(1, 5, n),
    'qa_defect_density': np.random.uniform(0, 0.5, n),
    'story_points_completed': np.random.randint(50, 500, n),
    'billable_hours_ratio': np.random.uniform(0.6, 1.0, n),
    'training_hours': np.random.uniform(0, 100, n),
    'certifications_count': np.random.randint(0, 10, n),
    'internal_hackathons_participated': np.random.randint(0, 5, n),
    'sick_days': np.random.randint(0, 20, n),
    'unplanned_absences': np.random.randint(0, 10, n),
    'avg_login_hours': np.random.uniform(6, 12, n),
    'peer_feedback_score': np.random.uniform(1, 5, n),
    'manager_score': np.random.uniform(1, 5, n),
    'kudos_count': np.random.randint(0, 20, n),
    'promotions_in_2y': np.random.randint(0, 3, n),
    'salary_percentile_band': np.random.choice(['Low', 'Medium', 'High'], n)
}

df = pd.DataFrame(data)

# Simulate performance band based on features (simplified logic)
def calculate_performance(row):
    score = 0
    score += row['on_time_delivery_rate'] * 20
    score += row['code_review_score'] * 10
    score += row['peer_feedback_score'] * 10
    score += row['manager_score'] * 15
    score += row['training_hours'] / 10
    score += row['certifications_count'] * 5
    score -= row['avg_task_delay_days'] / 2
    score -= row['bug_count'] / 5
    score -= row['sick_days']
    score -= row['unplanned_absences'] * 2
    if row['experience_years'] > 10:
        score += 10
    if row['job_level'] == 'Senior':
        score += 5
    # Add some noise
    score += np.random.normal(0, 10)
    if score > 70:
        return 'High'
    elif score > 40:
        return 'Medium'
    else:
        return 'Low'

df['perf_band_next'] = df.apply(calculate_performance, axis=1)

# Save to CSV
df.to_csv('data/employee_features.csv', index=False)

print("Synthetic dataset created and saved to data/employee_features.csv")
print(df.head())