# 📊 Employee Performance Predictor

## 🚀 Overview

This project is an end-to-end Machine Learning system that predicts employee performance (High / Medium / Low) using synthetic HR data. It demonstrates a complete data science pipeline including data generation, preprocessing, feature engineering, model training, evaluation, and prediction.

---

## 🎯 Problem Statement

Organizations struggle to objectively evaluate employee performance. This system helps to:

* Identify high-performing employees for promotion
* Detect low performers for early intervention
* Improve HR decision-making using data-driven insights
* Optimize training and workforce productivity

---

## 💼 Business Impact

* 📉 Cost Reduction: Reduces hiring and training inefficiencies
* 📈 Performance Optimization: Helps identify skill gaps
* 🔍 Better HR Decisions: Removes bias from evaluations
* 🧠 Data-Driven Insights: Enables predictive workforce analytics

---

## 🛠 Tech Stack

* Language: Python 3.8+
* Libraries: Pandas, NumPy, Scikit-learn
* Visualization: Matplotlib, Seaborn
* Model: Random Forest Classifier
* Tools: Jupyter Notebook, Streamlit (optional dashboard)

---

## 🧠 Machine Learning Pipeline

Data Generation → Data Cleaning → EDA → Feature Engineering → Model Training → Evaluation → Prediction → Insights

---

## 📁 Project Structure

Employee-Performance-Predictor/
│
├── data/                    # Raw & processed datasets
├── notebooks/              # EDA & analysis notebooks
├── src/                    # Source code
│   ├── generate_data.py
│   ├── train_model.py
│   ├── predict_performance.py
│
├── models/                 # Saved ML models
├── outputs/                # Results & logs
├── images/                 # Visualizations
├── main.py                 # Full pipeline runner
├── requirements.txt        # Dependencies
└── README.md               # Project documentation

---

## 🚀 How to Run This Project

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Run Full Pipeline

```bash
python main.py
```

---

### 3️⃣ Run Individual Modules

```bash
# Generate dataset
python src/generate_data.py

# Train model
python src/train_model.py

# Make predictions
python src/predict_performance.py
```

---

## 📊 Model Performance

* Achieves strong classification accuracy on synthetic HR dataset
* Handles multi-class classification (High / Medium / Low)
* Uses Random Forest for robustness and feature importance analysis

---

## 📌 Key Features Used

* Age
* Experience
* Training Hours
* Manager Score
* Work Efficiency Metrics
* Attendance Patterns

---

## 📈 Output

The model predicts:

* 🟢 High Performer
* 🟡 Medium Performer
* 🔴 Low Performer

with probability scores for each class.

---

## 🔮 Future Improvements

* Integrate real HR datasets
* Add employee attrition prediction
* Deploy as Streamlit web app
* Build FastAPI backend for real-time prediction
* Add explainable AI (SHAP/LIME)

---

## 👨‍💻 Author

Om Navgire
Guided by: Umesh Yadav Sir (EDC IIT Delhi)

---

## 📜 License

This project is intended for educational and portfolio purposes.

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and share it with others!
