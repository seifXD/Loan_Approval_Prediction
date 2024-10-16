# Loan Approval Prediction Model
This repository contains a machine learning project focused on predicting loan approval statuses based on applicant features such as age, income, home ownership, employment length, and other credit-related factors. The goal is to predict whether a loan will be approved (loan_status = 1) or rejected (loan_status = 0).

## Project Overview
This project leverages the Playground Series S4E10 dataset from Kaggle, with features such as:

- **Person_age:** Age of the applicant.
- **Person_income:** Applicant's income.
- **Person_home_ownership:** Home ownership status (e.g., rent, own).
- **Person_emp_length:** Duration of employment.
- **Loan_intent:** Purpose of the loan (e.g., education, medical).
- **Loan_amnt:** Loan amount.
- And others related to credit history and loan interest rates.
## Approach
Two different machine learning models were trained and evaluated:

- XGBoost - An advanced boosting algorithm known for its speed and accuracy.
- LightGBM - A gradient boosting framework that is optimized for faster training and low memory usage.
Both models were evaluated using the Area Under the ROC Curve (AUC) to determine their effectiveness in predicting loan approvals.

## Key Steps:
**Data Preprocessing:** Missing values were handled, and categorical features were label-encoded to transform them into numerical format.
Feature Scaling: A StandardScaler was used to normalize the data.
Model Evaluation: The models were trained, and predictions were generated based on their respective parameters.
## Results
**Best Model:** The best performing model was Version 1 (regression model), achieving a score of 0.95293 in the AUC evaluation.
Version 2 (classification model) achieved a lower score of 0.87702, showing that regression worked better for this dataset.
## Future Improvements
Tuning model parameters further to enhance performance.
Avoiding data leakage during scaling by using pipelines to separate the training and test data preprocessing steps.
