# Bank Customer Churn Prediction

This project focuses on predicting customer churn for a bank using various data analysis and machine learning techniques. The goal is to identify customers who are at risk of leaving the bank and to understand the factors contributing to their decision.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Conclusion](#conclusion)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

Predicting customer churn is crucial for banks to maintain their customer base and ensure steady revenue. This project applies machine learning algorithms to predict which customers are likely to leave the bank. The project includes steps such as data preprocessing, exploratory data analysis (EDA), model training and evaluation.

## Dataset

The dataset used for this project contains information about the bank's customers, including demographic details, account information, and transaction history. Key features include:

- CustomerID
- CreditScore
- Geography
- Gender
- Age
- Tenure
- Balance
- NumberOfProducts
- HasCrCard
- IsActiveMember
- EstimatedSalary
- Exited (Target variable)

## Data Preprocessing

Data preprocessing steps include:

- Handling missing values
- Encoding categorical variables using OneHotEncoder
- Scaling numerical features
- Splitting the data into training and testing sets

## Exploratory Data Analysis

EDA involves:

- Visualizing the distribution of features
- Analyzing correlations between features
- Identifying patterns and insights related to customer churn


## Modeling

Various machine learning models are applied and compared, including:

- Logistic Regression
- Random Forest
- Gradient Boosting

## Evaluation

Model performance is evaluated using metrics such as:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

## Conclusion

The project concludes with a summary of findings, the best-performing model, and potential strategies for reducing customer churn based on the model's insights.

## How to Run

To run the project, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/mohammadreza-mohammadi94/Data_Analysis_Machine_Learning.git
   cd Data_Analysis_Machine_Learning/2.%20Marketing%20and%20Customer%20Analysis/Bank%20Customer%20Churn%20Prediction
   ```

2. Run the Jupyter notebook:
   ```sh
   jupyter notebook
   ```
   Open the `Bank_Customer_Churn_Prediction.ipynb` notebook and execute the cells.


## Requirements

- Python 3.6+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- Jupyter Notebook

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.


