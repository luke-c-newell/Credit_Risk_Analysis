# Credit_Risk_Analysis
## Overview
This repository contains code for machine learning models that I have used to assess credit risk across a portfolio of personal loans. My client, LendingClub, wants to understand how the performance of different machine learning models differs so they can assess whether to use them for assessing credit risk. As credit risk is an unbalanced classification dataset, due to the fact that good loans vastly outnumber bad loans, there are a number of different models that can be employed. 

Using a Jupyter Notebook, the scikit-learn and imbalanced-learn libraries, I have used oversampling, undersampling and combinatorial techniques to analyze a credit card dataset (LoanStats_2019Q1.csv). To compare the models, I have calculated the balanced accuracy score, as well as the precision and recall scores to understand which model performs best. 

### Terms used to assess Machine Learning models
Precision: Number of true positives / Sum of all positives

Recall: Number of true positives / Number of relevant elements

Balanced Accuracy Score: Average of Recall obtained on each class (High Risk and Low Risk)

## Results
Below is a comparison of the classification reports and balanced accuarcy scores of each ML model. I have also described the accuracy, precision and recall scores for each machine learning algorithm used in the analysis. I have rounded the balanced accuracy scores to 2 decimal places for consistency across the results, full values can be found in the credit_risk_resampling.ipynb and credit_risk_ensemble.ipynb files. 

### Oversampling algorithms
#### RandomOverSampler
![RandomOverSampler](https://github.com/luke-c-newell/Credit_Risk_Analysis/blob/main/images/RandomOverSampler.png "RandomOverSampler.png")

- Balanced accuracy score: 0.65
- High Risk Precision score: 0.01
- Low Risk Precision score: 1.00
- Overall Precision score: 0.99
- High Risk Recall score: 0.74
- Low Risk Recall score: 0.55
- Overall Recall score: 0.55

#### SMOTE
![SMOTE](https://github.com/luke-c-newell/Credit_Risk_Analysis/blob/main/images/SMOTE.png "SMOTE.png")

- Balanced accuracy score: 0.66
- High Risk Precision score: 0.01
- Low Risk Precision score: 1.00
- Overall Precision score: 0.99
- High Risk Recall score: 0.63
- Low Risk Recall score: 0.69
- Overall Recall score: 0.69

### Undersampling algorithms
#### ClusterCentroids
![ClusterCentroids](https://github.com/luke-c-newell/Credit_Risk_Analysis/blob/main/images/ClusterCentroids.png "ClusterCentroids.png")

- Balanced accuracy score: 0.54
- High Risk Precision score: 0.01
- Low Risk Precision score: 1.00
- Overall Precision score: 0.99
- High Risk Recall score: 0.67
- Low Risk Recall score: 0.42
- Overall Recall score: 0.42

### Combinatorial algorithms
#### SMOTEENN
![SMOTEENN](https://github.com/luke-c-newell/Credit_Risk_Analysis/blob/main/images/SMOTEENN.png "SMOTEENN.png")

- Balanced accuracy score: 0.66
- High Risk Precision score: 0.01
- Low Risk Precision score: 1.00
- Overall Precision score: 0.99
- High Risk Recall score: 0.71
- Low Risk Recall score: 0.60
- Overall Recall score: 0.61

### Machine Learning models that reduce bias
#### BalancedRandomForestClassifier
![BalancedRandomForestClassifier](https://github.com/luke-c-newell/Credit_Risk_Analysis/blob/main/images/BalancedRandomForestClassifier.png "BalancedRandomForestClassifier.png")

- Balanced accuracy score: 0.78
- High Risk Precision score: 0.03
- Low Risk Precision score: 1.00
- Overall Precision score: 0.99
- High Risk Recall score: 0.70
- Low Risk Recall score: 0.87
- Overall Recall score: 0.87

#### EasyEnsembleClassifier
![EasyEnsembleClassifier](https://github.com/luke-c-newell/Credit_Risk_Analysis/blob/main/images/EasyEnsembleClassifier.png "EasyEnsembleClassifier.png")

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

Overall, the EasyEnsembleClassifier model had the highest balanced accuracy score (0.93), with the ClusterCentroids model having the lowest score (0.54). The BalancedRandomForestClassifier had the second highest score (0.78) which indicates that the algorithms designed to reduce bias were the most accurate of the models used in this analysis.

### Precision Scores
|	|RandomOverSampler|SMOTE|ClusterCentroids|SMOTEENN|BalancedRandomForestClassifier|EasyEnsembleClassifier|
|------|-----|------|-------|-------|-------|-------|
|high_risk|0.01|0.01|0.01|0.01|0.03|0.09|
|low_risk|1.00|1.00|1.00|1.00|1.00|1.00|
|overall|0.99|0.99|0.99|0.99|0.99|0.99|

The precision scores for the RandomOverSampler, SMOTE, ClusterCentroids and SMOTEENN algorithms were all identical with 0.01 for the high_risk and 1.00 for the low risk. The BalancedRandomForestClassifier and EasyEnsembleClassifier performed slightly better with 0.03 and 0.09 respectively for the high_risk datapoints. This means that for the EasyEnsembleClassifier model, 9% of loans identified as high risk were correctly identified and 91% were actually low risk. 

### Recall Scores
|	|RandomOverSampler|SMOTE|ClusterCentroids|SMOTEENN|BalancedRandomForestClassifier|EasyEnsembleClassifier|
|------|-----|------|-------|-------|-------|-------|
|high_risk|0.74|0.63|0.67|0.71|0.70|0.92|
|low_risk|0.55|0.69|0.42|0.60|0.87|0.94|
|overall|0.55|0.69|0.42|0.61|0.87|0.94|

The recall scores indicate that the EasyEnsembleClassifier was the most effective with 92% of high risk loans being correctly identified and an overall recall score of 94%. The ClusterCentroids algorithm performed the worst in recall, with an overall score of 0.42.

### Recommendations
Overall, the best performing machine learning model for the loan data was the EasyEnsembleClassifier algorithm. This model had the highest balanced accuracy score, the highest precision for high risk loans and the highest recall scores for both high and low risk loans. Despite being the best model of the algorithms tested, the EasyEnsembleClassifier still can only predict high risk loans at a 9% precision level. This is a drawback of using machine learning models to predict the results of highly unbalanced datasets, as even the best models still struggle to accurately predict results from a relatively small amount of training data. 

I would recommend using the EasyEnsembleClassifier as part of LendingClub's approach to loan approvals. The model can recall 92% of high risk loans but this means that 8% of high risk loans are missed by the algorithm. The model also identifies a large number of low risk loans as high risk, as shown by the high risk precision score of 0.09. I would use this model to identify loans for further analysis, but it should not be used as an exclusive method for loan approvals.