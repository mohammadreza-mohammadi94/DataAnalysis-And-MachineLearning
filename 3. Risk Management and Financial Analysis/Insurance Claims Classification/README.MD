# Insurance Claims Classification Project

This repository contains an end-to-end solution for predicting insurance claim statuses based on customer and vehicle information. The project includes data preprocessing, exploratory data analysis (EDA), feature engineering, and the implementation of machine learning models such as Support Vector Classifier (SVC) and Random Forest with hyperparameter tuning.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Preprocessing](#preprocessing)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis)
5. [Modeling](#modeling)
6. [Results](#results)
7. [Future Improvements](#future-improvements)

## Project Overview <a id="project-overview"></a>
The goal of this project is to classify whether a customer will file an insurance claim based on various features such as vehicle specifications, customer demographics, and subscription details. The dataset has both numerical and categorical variables, which are preprocessed and used to train machine learning models.

## Dataset <a id="dataset"></a>
The dataset used in this project contains customer information and vehicle specifications. It is loaded from a CSV file named `Insurance claims data.csv`.

Key columns:
- **Claim Status**: The target variable indicating whether a claim has been made (0 = No, 1 = Yes).
- **Numerical Features**: Vehicle age, customer age, subscription length, etc.
- **Categorical Features**: Region code, vehicle segment, fuel type, transmission type, etc.

## Preprocessing <a id="preprocessing"></a>
### 1. Handling Missing and Duplicate Data
- Missing values are checked and handled.
- Duplicate records are identified and removed.

### 2. Encoding Categorical Variables
- **OneHotEncoding** is applied to binary categorical variables (e.g., `is_esc`, `is_parking_sensors`).
- **LabelEncoding** is used for multi-class categorical features like `region_code`, `fuel_type`, and `engine_type`.

### 3. Feature Extraction
- For `max_torque` and `max_power` columns, numerical values and RPM data are extracted using regular expressions.

### 4. Data Cleaning
- Irrelevant columns are dropped after encoding and feature extraction.

## Exploratory Data Analysis (EDA) <a id="exploratory-data-analysis"></a>
Several visualization techniques are used to gain insights into the dataset:
1. **Countplots**: Visualize the distribution of categorical features.
2. **Histograms**: Show the distribution of numerical features.
3. **Correlation Matrix**: Display relationships between variables.
4. **Boxplots**: Compare numerical features across claim statuses.

## Modeling <a id="modeling"></a>
### 1. Train-Test Split
The data is split into training and testing sets using an 80/20 ratio.

### 2. Oversampling with SMOTE
The Synthetic Minority Oversampling Technique (SMOTE) is applied to handle class imbalance in the dataset.

### 3. Standardization
Numerical features are standardized using `StandardScaler`.

### 4. Models
- **Support Vector Classifier (SVC)**: A classifier using the sigmoid kernel and regularization parameter `C=1`.
- **Random Forest Classifier**: A baseline Random Forest model is used for comparison.

### 5. Hyperparameter Tuning
- Hyperparameter tuning is performed on the Random Forest model using `RandomizedSearchCV` to optimize parameters such as `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`.

## Results <a id="results"></a>
The models are evaluated based on the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**

The Random Forest model, after tuning, showed better performance compared to SVC.

## Future Improvements <a id="future-improvements"></a>
- Implement additional machine learning algorithms such as XGBoost or LightGBM.
- Fine-tune models using GridSearchCV for more precise hyperparameter tuning.
- Explore feature selection techniques to reduce dimensionality and improve model performance.
- Add more feature engineering techniques for better predictive power.
