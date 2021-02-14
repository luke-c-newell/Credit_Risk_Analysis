# Credit_Risk_Analysis
## Overview
This repository contains code for machine learning models that I have used to assess credit risk across a portfolio of personal loans. My client, LendingClub, wants to understand how the performance of different machine learning models differs so they can assess whether to use them for assessing credit risk. As credit risk is an unbalanced classification dataset, due to the fact that good loans vastly outnumber bad loans, there are a number of different models that can be employed. 

Using a Jupyter Notebook, the scikit-learn and imbalanced-learn libraries, I have used oversampling, undersampling and combinatorial techniques to analyze a credit card dataset (LoanStats_2019Q1.csv). To compare the models, I have calculated the balanced accuracy score, as well as the precision and recall scores to understand which model performs best. 

### Terms used to assess Machine Learning models
Precision: Number of true positives / Sum of all positives

Recall: Number of true positives / Number of relevant elements

Balanced Accuracy Score: Average of Recall obtained on each class (High Risk and Low Risk)

## Results
Below, I have described the accuracy, precision and recall scores for each machine learning algorithm used in the analysis. I have rounded the balanced accuracy scores to 2dp, for consistency across the results, full values can be seen in the credit_risk_resampling and credit_risk_ensemble files. 

### Oversampling algorithms
#### RandomOverSampler
- Balanced accuracy score: 0.65
- High Risk Precision score: 0.01
- Low Risk Precision score: 1.00
- Overall Precision score: 0.99
- High Risk Recall score: 0.74
- Low Risk Recall score: 0.55
- Overall Recall score: 0.55

#### SMOTE
- Balanced accuracy score: 0.66
- High Risk Precision score: 0.01
- Low Risk Precision score: 1.00
- Overall Precision score: 0.99
- High Risk Recall score: 0.63
- Low Risk Recall score: 0.69
- Overall Recall score: 0.69

### Undersampling algorithm
#### ClusterCentroids
- Balanced accuracy score: 0.54
- High Risk Precision score: 0.01
- Low Risk Precision score: 1.00
- Overall Precision score: 0.99
- High Risk Recall score: 0.67
- Low Risk Recall score: 0.42
- Overall Recall score: 0.42

### Combinatorial algorithm
#### SMOTEENN
- Balanced accuracy score: 0.66
- High Risk Precision score: 0.01
- Low Risk Precision score: 1.00
- Overall Precision score: 0.99
- High Risk Recall score: 0.71
- Low Risk Recall score: 0.60
- Overall Recall score: 0.61

### Machine Learning models that reduce bias
#### BalancedRandomForestClassifier
- Balanced accuracy score: 0.78
- High Risk Precision score: 0.03
- Low Risk Precision score: 1.00
- Overall Precision score: 0.99
- High Risk Recall score: 0.70
- Low Risk Recall score: 0.87
- Overall Recall score: 0.87

#### EasyEnsembleClassifier
- Balanced accuracy score: 0.93
- High Risk Precision score: 0.09
- Low Risk Precision score: 1.00
- Overall Precision score: 0.99
- High Risk Recall score: 0.92
- Low Risk Recall score: 0.94
- Overall Recall score: 0.94

## Summary
### Balanced Accuracy Scores
|	|RandomOverSampler|SMOTE|ClusterCentroids|SMOTEENN|BalancedRandomForestClassifier|EasyEnsembleClassifier|
|------|-----|------|-------|-------|-------|-------|
|Score|0.65|0.66|0.54|0.66|0.78|0.93|

### Precision Scores
|	|RandomOverSampler|SMOTE|ClusterCentroids|SMOTEENN|BalancedRandomForestClassifier|EasyEnsembleClassifier|
|------|-----|------|-------|-------|-------|-------|
|high_risk|0.01|0.01|0.01|0.01|0.03|0.09|
|low_risk|1.00|1.00|1.00|1.00|1.00|1.00|
|overall|0.99|0.99|0.99|0.99|0.99|0.99|

### Precision Scores
|	|RandomOverSampler|SMOTE|ClusterCentroids|SMOTEENN|BalancedRandomForestClassifier|EasyEnsembleClassifier|
|------|-----|------|-------|-------|-------|-------|
|high_risk|0.74|0.63|0.67|0.71|0.70|0.92|
|low_risk|0.55|0.69|0.42|0.60|0.87|0.94|
|overall|0.55|0.69|0.42|0.61|0.87|0.94|
