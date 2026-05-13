
# 📞 Telecom Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.10-yellow)
![Machine Learning](https://img.shields.io/badge/ML-Classification-blue)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Used-orange)

---

# 📌 Project Overview

This project focuses on predicting **customer churn** for a telecom company using machine learning.
The goal is to identify customers who are likely to leave the service so that proactive retention strategies can be applied.

The dataset contains customer demographic information, subscription details, and service usage behavior.

Built using **Python, Pandas, Scikit-learn, and Machine Learning classification models**.

---

# 🎯 Objective

To build a predictive model that classifies whether a customer will:

* ✅ Stay (`No Churn`)
* ❌ Leave (`Churn`)

---

# 📂 Dataset

The dataset used is the **Telco Customer Churn dataset**, containing 7043 records and 21 features.

### Key Features:

* Customer demographics (gender, senior citizen, partner, dependents)
* Services subscribed (internet, phone, streaming, tech support, etc.)
* Account information (contract type, payment method)
* Billing details (monthly charges, total charges)
* Target variable: **Churn**

---

# 🧹 Data Preprocessing

### Steps performed:

* Converted `TotalCharges` to numeric
* Handled missing values using imputation
* Feature engineering on categorical variables
* Encoded categorical features using **OneHotEncoder**
* Scaled numerical features using **StandardScaler**
* Train-test split (80/20)

---

# 📊 Exploratory Data Analysis (EDA)

Performed analysis on:

* Distribution of numerical features (tenure, charges)
* Class imbalance in churn target
* Categorical feature distributions
* Correlation between numerical variables

Key insight:

* Customers with **short tenure + high monthly charges** are more likely to churn.

---

# 🧠 Machine Learning Models

## 1️⃣ Logistic Regression (Baseline Model)

Pipeline included:

* Preprocessing (scaling + encoding)
* Logistic Regression classifier

### 📈 Performance:

* Accuracy: **~82%**
* Good balance between precision and recall

---

## 2️⃣ Decision Tree Classifier

* Trained with same preprocessing pipeline

### 📉 Performance:

* Accuracy: **~72%**
* Lower generalization compared to Logistic Regression

---

## 3️⃣ Hyperparameter Tuning (GridSearchCV)

Optimized Decision Tree using cross-validation:

### Best Parameters:

```python
criterion = entropy
max_depth = 5
min_samples_leaf = 7
min_samples_split = 2
```

### Best CV Score:

* **~79.3% accuracy**

---

# ⚙️ Model Pipeline

All models were built using **Scikit-learn Pipelines**:

* ColumnTransformer (Preprocessing)

  * Numerical → Imputer + StandardScaler
  * Categorical → Imputer + OneHotEncoder
* Classifier (Logistic Regression / Decision Tree)

---

# 📈 Evaluation Metrics

Used:

* Accuracy
* Precision
* Recall
* F1-score
* Classification Report

---

# 🚀 Model Deployment

The final trained model was saved using Joblib:

```python
joblib.dump(log_reg_pipeline, 'telco.joblib')
```

---

# 🧠 Key Insights

* Tenure is a strong predictor of churn
* Month-to-month contracts have higher churn rates
* Electronic check payment users churn more frequently
* Longer contracts reduce churn probability

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib & Seaborn
* Scikit-learn
* GridSearchCV
* Joblib

---

# 📊 Future Improvements

* Use advanced models (XGBoost, LightGBM)
* Handle class imbalance (SMOTE / class weights)
* Feature selection techniques
* Deploy using Streamlit or Flask
* Add SHAP for model explainability

---

# 👩‍💻 Author

**Sarah Osama El Khouly**
Mechanical Engineering Graduate | Data Science & AI Enthusiast

📧 Email:(sarahelkhouly2@gmail.com)
🔗 LinkedIn: (https://www.linkedin.com/in/sarah-el-khouly-423307247/)


