# Credit Card Fraud Detection using Machine Learning

This project focuses on detecting fraudulent credit card transactions using various machine learning classifiers. The dataset is highly imbalanced and contains transactions made by European cardholders in September 2013.

## 📂 Dataset

The dataset used is `creditcard.csv` which includes:

- 284,807 transactions
- 492 fraudulent transactions
- Features: PCA-transformed V1–V28, `Time`, `Amount`, and target class `Class` (0 = Non-fraud, 1 = Fraud)

## 🔍 Objective

To build a robust fraud detection model that minimizes false negatives while handling class imbalance effectively.

## ⚙️ Technologies and Libraries Used

- Python, Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Machine Learning Algorithms:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - Support Vector Machine (SVM)

## 🧪 Methodology

1. **Exploratory Data Analysis (EDA)**: Analyzed class distribution, correlations, and feature distributions.
2. **Data Preprocessing**:
   - Standardized features
   - Handled class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
3. **Model Training**:
   - Split data into training and test sets
   - Trained various classifiers using cross-validation
   - Performed GridSearchCV for hyperparameter tuning
4. **Model Evaluation**:
   - Evaluated models using ROC-AUC, Precision-Recall, F1-score, Confusion Matrix

## 📊 Results

| Model                 | ROC-AUC Score | Precision | Recall | F1 Score |
|----------------------|---------------|-----------|--------|----------|
| Logistic Regression  | 0.97+         | High      | High   | Good     |
| Random Forest        | 0.98+         | High      | High   | Excellent|
| Gradient Boosting    | 0.98+         | High      | High   | Excellent|
| SVM                  | 0.95+         | Medium    | Medium | Fair     |

> Random Forest and Gradient Boosting outperformed others in terms of accuracy and recall, which is crucial in fraud detection to reduce false negatives.

## 📌 Key Takeaways

- Addressing class imbalance is crucial; SMOTE significantly improved recall.
- Ensemble methods like Random Forest and Gradient Boosting provided the best results.
- ROC-AUC and Precision-Recall metrics are preferred over accuracy in imbalanced classification problems.

