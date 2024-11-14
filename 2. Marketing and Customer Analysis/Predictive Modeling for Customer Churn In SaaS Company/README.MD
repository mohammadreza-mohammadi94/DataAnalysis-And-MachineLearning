# Customer Churn Prediction Project

## Project Overview

This project aims to build a predictive model to determine the likelihood of customer churn based on various customer engagement metrics and behavioral patterns. The model's predictions are designed to be interpretable, allowing stakeholders to identify key drivers of churn and take action to retain customers. The model is optimized to be deployable in real-time or near-real-time environments for integration into business platforms.

## Objectives

The primary objectives of this project include:

- **Developing a Predictive Model**: Using machine learning to predict customer churn with high accuracy.
- **Feature Engineering**: Extracting and engineering features to improve model performance, especially focusing on engagement and behavioral metrics.
- **Interpretability**: Ensuring that the model's predictions are explainable to facilitate data-driven decisions by stakeholders.
- **Deployable Solution**: Preparing the model for real-time or near-real-time deployment within client platforms.
- **Performance Optimization**: Balancing precision, recall, and interpretability to meet specific business benchmarks.

## Dataset

The dataset used in this project contains customer-related features, such as demographic information, customer engagement metrics, and historical behavior. Key features include `Churn Score`, `Tenure Months`, `Total Charges`, `Avg Monthly Spend`, `CLTV`, and others. 

### Features Selected for the Model:
- `Churn Score`
- `Tenure Months`
- `Total Charges`
- `Avg Monthly Spend`
- `CLTV`
- `Latitude`
- `Longitude`
- `Dependents`
- `Upsell Potential`


## Usage

1. **Data Preparation**: Preprocess the dataset by handling missing values, encoding categorical features, and feature scaling (if necessary).
   
2. **Model Training**: Train the model using the features identified above. Models used include **RandomForestClassifier** and **CatBoostClassifier**. Hyperparameters are optimized using cross-validation.
   
3. **Model Evaluation**: Evaluate the model using metrics such as accuracy, precision, recall, and AUC score. Additionally, use interpretability methods (like SHAP) to understand feature contributions to predictions.

4. **Generate Churn Probabilities**: After training, calculate churn probabilities for each customer and add them as a new column in the dataset.

5. **Deployment**: The model is containerized using Docker to facilitate deployment in real-time environments.

## Model Interpretability

The model’s interpretability is enhanced using **SHAP** values, providing insights into feature importance and their impact on churn predictions. Use SHAP plots to visualize individual and global feature impacts.

## Results

- **Feature Importance**: Key factors contributing to customer churn include `Churn Score`, `Total Charges`, and `Tenure Months`.

## Contributing

Contributions to improve this project are welcome. Please open an issue to discuss potential changes or submit a pull request with detailed notes on the modifications.
