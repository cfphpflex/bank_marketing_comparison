# Bank Marketing Classifier Comparison

## 📌 Overview
This project compares four supervised machine learning classifiers — Logistic Regression, K-Nearest Neighbors (KNN), Decision Trees, and Support Vector Machines (SVM)—on a UCI bank marketing dataset to predict if a customer will subscribe to a term deposit. 
The analysis helps optimize telemarketing efforts, reduce costs, and increase subscription conversion rates.

## 📁 Dataset
- Source: [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- Filename: `data/bank.csv`
- Features include client demographic info, contact duration, past campaign outcomes, and more.
- Target: `y` (binary) — whether the client subscribed to a term deposit (`yes`/`no`).

## 📊 Models Compared

| Model                | Cross-Validation Score | Accuracy | Precision (1) | Recall (1) | F1-Score (1) |
|----------------------|------------------------|----------|----------------|------------|--------------|
| Logistic Regression  | 0.9134                 | 0.8887   | 0.5041         | 0.4079     | 0.4509       |
| K-Nearest Neighbors  | 0.9186                 | 0.8688   | 0.3587         | 0.2171     | 0.2705       |
| Decision Tree        | 0.9007                 | 0.8548   | 0.3770         | 0.4539     | 0.4119       |
| Support Vector Machine | **0.9191**           | **0.8924** | **0.5283**     | 0.3684     | **0.4341**   |

## 📈 Visualizations
- Count plots for class balance and categorical variables
- Heatmap of correlations
- ROC Curve and Precision-Recall Curve for best models
- Confusion matrices for all models

## 📋 Key Findings

### 🔍 Model Performance Comparison

1. **SVM**
   - ✅ Highest cross-validation score (91.91%)
   - ✅ Best precision and F1-score for positive class
   - Excellent overall balance with lowest false negatives among models
   - Confusion Matrix: 1155 TN, 56 TP

2. **Logistic Regression**
   - Strong general performance with 91.34% CV score
   - Highest accuracy for non-subscribers (class 0)
   - Confusion Matrix: 1144 TN, 62 TP

3. **Decision Tree**
   - Most balanced recall for class 1
   - Captures more true positives (69) than KNN or LR
   - Confusion Matrix: 1091 TN, 69 TP

4. **KNN**
   - Lowest recall for class 1 (only 33 TPs)
   - May underperform with high-dimensional or imbalanced data
   - Confusion Matrix: 1146 TN, 33 TP

### 🔑 Observations

- All models are highly accurate in identifying **non-subscribers**.
- **Subscribers (class 1)** remain harder to predict, reflecting dataset imbalance.
- **SVM** and **Logistic Regression** offer the best tradeoff between precision and recall.
- **Decision Tree** can be considered when interpretability is essential.

## 📘 How to Run
Open the notebook here:  
📎 [`bank_marketing_comparison.ipynb`](./bank_marketing_comparison.ipynb)

---

## ✅ Recommendations

### 1. Model Selection
- **Primary model**: SVM (highest F1 and precision on minority class)
- **Secondary model**: Logistic Regression (more interpretable, slightly lower performance)
- **Fallback**: Decision Tree (for explainability or business rule generation)

### 2. Feature Engineering
- Explore additional interactions (e.g., `job x education`)
- Use SHAP or feature importances to reduce dimensionality
- Engineer time-based or frequency features if available

### 3. Handling Imbalance
- SMOTE was helpful — keep using it
- Explore hybrid over/under sampling or ensemble boosting (e.g., AdaBoost, Balanced Random Forest)

### 4. Deployment Strategy
- Use models to score prospects before campaign outreach
- Apply A/B testing on model-guided vs. unguided calls
- Monitor false positives to control marketing costs

---

## 🔜 Next Steps
- Apply **ensemble models**: XGBoost, LightGBM, RandomForest
- Try **class weighting** as an alternative to SMOTE
- Evaluate **model drift** with new campaign cycles
- Improve **recall** further through deeper tuning

---