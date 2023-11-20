# Titanic Machine Learning Project

This project involves data analysis and preprocessing for the Titanic dataset to prepare it for machine learning tasks.

## Libraries Used

- pandas: Data manipulation and analysis
- numpy: Numerical operations
- matplotlib: Data visualization
- seaborn: Data visualization
- scipy: Scientific computing and statistics
- sklearn: Machine learning utilities

## Dataset

The dataset used in this project can be found [here](https://raw.githubusercontent.com/PeterLOVANAS/Titanic-machine-learning-project/main/datasets/Titanic_dataset_com.csv). It contains information about Titanic passengers.

## Data Preprocessing
This repository contains two version of preprocessing. second analysis has been added with Model Analysis.
Link to [Kaggle](https://www.kaggle.com/code/jigsaw13/titanic-data-analysis/notebook?scriptVersionId=151538575)

### Solving Missing Values

#### Drop Rows with All NaNs

We start by removing rows that contain all NaN values to clean the dataset.

#### Age Imputation

We impute missing 'Age' values based on the median age of passengers with the same 'Sex' and 'Pclass'.

#### Fare

We drop rows with missing 'Fare' values.

#### Cabin

Missing 'Cabin' values are filled based on the mode of cabins for passengers in the same class ('Pclass'). We also create a 'Cabin_Deck' column and a 'Has_Cabin' column.

#### Embarked

We fill missing 'Embarked' values with the mode value.

#### Boat

Missing 'Boat' values are imputed based on conditions. 

### Data Reduction & Transformation

#### Data Discretization

We discretize 'Age' and 'Fare' columns into ordinal bins using the KBinsDiscretizer.

#### One-Hot Encoding

We perform one-hot encoding for the 'Sex' column.

#### Label Encoding

We apply label encoding to the 'Cabin_Deck' column.

#### Dropping Unnecessary Columns

We remove the 'ticket' and 'PassengerId' columns as they are unnecessary.

#### Correcting Data Types

We adjust data types to ensure proper compatibility.

#### Name Format

We split the 'name' column into 'last_name', 'title', 'first_name' for better analysis.

#### Columns Order

We reorder columns for better organization of the dataset.

## Usage

You can use this code as a reference for preprocessing the Titanic dataset for machine learning tasks.

Feel free to explore the Jupyter Notebook file for a step-by-step walkthrough of the code and data analysis.
