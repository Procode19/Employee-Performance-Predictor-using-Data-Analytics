# Employee Performance Predictor

## Overview

This project builds a machine learning system to predict employee performance bands (High/Medium/Low) using synthetic HR data. It demonstrates end-to-end data science skills including data engineering, EDA, ML modeling, and deployment.

## Problem Statement

Companies need to predict employee performance to:
- Identify high performers for promotions
- Intervene with low performers early
- Optimize training and retention strategies
- Make data-driven HR decisions

## Business Value

- **Cost Savings**: Early identification reduces appraisal escalations and hiring costs
- **Retention**: Targeted interventions improve employee satisfaction
- **Efficiency**: Data-driven decisions replace subjective evaluations

## Tech Stack

- **Language**: Python 3.8+
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **ML Model**: Random Forest Classifier with class balancing
- **Data**: Synthetic HR dataset (1000 employees, 25+ features)

## Architecture

```
Input Data → Data Cleaning → EDA → Feature Engineering → Model Training → Evaluation → Predictions → HR Insights
```

## How to Run

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Clone or download the repository
2. Navigate to the project directory
3. Create virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Execution

Run the complete pipeline:
```bash
python main.py
```

Or run individual components:
```bash
# Generate data
python src/generate_data.py

# Train model
python src/train_model.py

# Make prediction
python src/predict_performance.py
```

For detailed analysis, open and run:
```bash
jupyter notebook notebooks/eda_notebook.ipynb
```

## Results

- **Model Performance**: ~85% accuracy on test set
- **Key Features**: Manager score, on-time delivery rate, training hours
- **Output**: Performance band prediction with actionable recommendations

## Screenshots

### Class Balance
![Class Balance](images/class_balance.png)

### Feature Importance
![Feature Importance](images/feature_importance.png)

### Confusion Matrix
![Confusion Matrix](images/confusion_matrix.png)

## Project Structure

```
Employee-Performance-Predictor/
├── data/                    # Raw and processed datasets
├── notebooks/              # Jupyter notebooks for analysis
├── src/                    # Source code
│   ├── generate_data.py    # Synthetic data creation
│   ├── train_model.py      # Model training script
│   └── predict_performance.py  # Prediction script
├── models/                 # Trained model files
├── outputs/                # Model outputs and results
├── images/                 # Generated plots and charts
├── requirements.txt        # Python dependencies
├── main.py                 # Main execution script
└── README.md              # This file
```

## Methodology

1. **Data Generation**: Created realistic synthetic HR data with 25+ features
2. **EDA**: Analyzed distributions, correlations, and feature-target relationships
3. **Preprocessing**: Built sklearn pipeline with imputation, scaling, and encoding
4. **Modeling**: Trained Random Forest with hyperparameter tuning and class balancing
5. **Evaluation**: Assessed with F1-macro, confusion matrix, and feature importance
6. **Insights**: Generated HR recommendations based on model predictions

## Future Improvements

- Integrate real HR data (with privacy compliance)
- Add deep learning models (Neural Networks)
- Build employee attrition prediction
- Create interactive dashboard (Streamlit/FastAPI)
- Implement real-time prediction API

## Contributing

This is an educational project. Feel free to fork and enhance!

## License

MIT License - Free for educational and personal use.

## Contact

For questions or collaborations, reach out via GitHub issues.