# Medical Insurance Prediction

![Medical Insurance](https://i.ibb.co/kMc3QQw/what-makes-a-good-medical-insurance.jpg)

Welcome to the Medical Insurance Prediction project! This repository contains a Jupyter Notebook that performs data analysis and machine learning to predict medical insurance costs based on various features.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Applied Process](#applied_process)
- [Model Evaluation](#model-evaluation)
- [Contributing](#contributing)

## Project Overview
The goal of this project is to predict medical insurance charges using a variety of regression techniques. This involves exploratory data analysis (EDA), data preprocessing, feature engineering, model training, and evaluation.

## Dataset
The dataset used in this project contains information about individuals such as age, sex, BMI, number of children, smoking status, region, and medical charges. This dataset can be found in the `insurance.csv` file.

### Features
- **Age:** Age of the individual.
- **Sex:** Gender of the individual (male/female).
- **BMI:** Body Mass Index.
- **Children:** Number of children/dependents covered by insurance.
- **Smoker:** Smoking status of the individual (yes/no).
- **Region:** Residential area (northwest, southeast, southwest, northeast).
- **Charges:** Medical insurance costs.

## Installation
To run this project locally, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/mohammadreza-mohammadi94/Data_Analysis_Machine_Learning.git
    cd Data_Analysis_Machine_Learning/4.%20Public%20Health%20and%20Safety/Medical%20Insurance%20Prediction
    ```

## Applied Process
- **Data Cleaning and Preprocessing:** Handle missing values, encode categorical variables, and scale numerical features.
- **Exploratory Data Analysis (EDA):** Visualize data distributions, relationships, and correlations.
- **Feature Engineering:** Create new features from existing ones to improve model performance.
- **Model Training:** Train multiple regression models including Linear Regression, Ridge Regression, and Lasso Regression.
- **Model Evaluation:** Evaluate models using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared.

## Model Evaluation
The models are evaluated based on their performance on the test set. Metrics used include:
- **Mean Absolute Error (MAE):** Measures the average magnitude of errors in a set of predictions, without considering their direction.
- **Mean Squared Error (MSE):** Measures the average of the squares of the errors—that is, the average squared difference between the estimated values and the actual value.
- **R-squared:** Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.

## Contributing
Contributions are welcome! If you have any improvements or suggestions, feel free to fork the repository, create a new branch, and submit a pull request.

