# Bank Marketing Classifier Comparison

## 📌 Overview
This project compares four supervised machine learning classifiers — Logistic Regression, K-Nearest Neighbors (KNN), Decision Trees, and Support Vector Machines (SVM)—on a Kagel bank marketing dataset to predict if a customer will subscribe to a term deposit. 
The analysis helps optimize telemarketing efforts, reduce costs, and increase subscription conversion rates.

## 📁 Dataset
- Source: [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- Filename: `data/bank.csv`
- Features include client demographic info, contact duration, past campaign outcomes, and more.
- Target: `y` (binary) — whether the client subscribed to a term deposit (`yes`/`no`).

## 📊 Models Compared
| Model                | Cross-Validation Score | Accuracy | Precision (1) | Recall (1) | F1-Score (1) |
|----------------------|------------------------|----------|----------------|------------|--------------|
| Logistic Regression  | 0.9049                 | 0.8983   | 0.5795         | 0.3355     | 0.4250       |
| K-Nearest Neighbors  | 0.8903                 | 0.8858   | 0.4746         | 0.1842     | 0.2654       |
| Decision Tree        | 0.8584                 | 0.8666   | 0.4014         | 0.3882     | 0.3946       |
| Support Vector Machine | 0.8897               | 0.8924   | 0.5500         | 0.2171     | 0.3113       |

## 📈 Visualizations
- Count plots for class balance and categorical variables
- Heatmap of correlations
- ROC Curve for Logistic Regression showing strong model performance
- Confusion matrices for all models

## 📋 Key Findings

### Model Performance Comparison

1. **Logistic Regression**
   - Best overall performance with 90.49% cross-validation score
   - Highest precision (92.04%) for negative class
   - Good balance between precision and recall
   - Confusion Matrix: 1168 true negatives, 51 true positives

2. **SVM**
   - Second-best performance with 88.97% cross-validation score
   - Highest precision (90.83%) for negative class
   - Good at identifying non-subscribers
   - Confusion Matrix: 1178 true negatives, 33 true positives

3. **KNN**
   - Third-best performance with 89.03% cross-validation score
   - Good precision (90.45%) for negative class
   - Lower recall for positive class
   - Confusion Matrix: 1174 true negatives, 28 true positives

4. **Decision Tree**
   - Base model performance: 85.84% cross-validation score
   - Optimized model (depth=2):
     * Training Accuracy: 89.89%
     * Test Accuracy: 89.61%
     * Good precision (92.31%) for negative class
     * Balanced performance between classes
   - Confusion Matrix: 1117 true negatives, 59 true positives
   - Shows good generalization with minimal overfitting (train-test gap < 0.3%)

### Key Observations
1. All models show strong performance in identifying non-subscribers (class 0)
2. Models struggle with identifying subscribers (class 1), showing lower recall
3. Class imbalance is evident in the dataset (1205 non-subscribers vs 152 subscribers)
4. Logistic Regression provides the best balance between precision and recall
5. Decision Tree with depth=2 shows promising results with minimal overfitting

## 📌 Next Steps
- Apply **class weights** or **SMOTE** to improve recall on the minority class.
- Experiment with **ensemble models** (e.g., Random Forest, XGBoost).
- Explore **real-time deployment** and **feature engineering** for campaign metadata.

## 📘 How to Run
Open the Jupyter Notebook:  
📎 [`bank_marketing_comparison.ipynb`](./bank_marketing_comparison.ipynb)

 
## Recommendations

1. **Model Selection**
   - Use Logistic Regression as the primary model due to its balanced performance
   - Consider the optimized Decision Tree (depth=2) as a secondary model
   - Implement ensemble methods to improve prediction of positive class

2. **Feature Engineering**
   - Investigate feature importance to identify key predictors
   - Consider creating interaction terms between important features
   - Explore feature selection techniques to reduce dimensionality

3. **Data Collection**
   - Gather more data for the positive class to address imbalance
   - Consider collecting additional customer behavior metrics
   - Implement tracking of campaign timing and frequency

4. **Implementation Strategy**
   - Use model predictions to prioritize high-probability customers
   - Implement A/B testing for different marketing approaches
   - Monitor model performance regularly and retrain as needed
   - Consider using the Decision Tree model for its interpretability

## Notebook
The complete analysis can be found in [bank_marketing_comparison.ipynb](bank_marketing_comparison.ipynb)

## Next Steps
- Implement ensemble methods
- Gather additional campaign data
- Explore feature engineering opportunities
- Consider real-time prediction capabilities
- Further optimize Decision Tree hyperparameters
