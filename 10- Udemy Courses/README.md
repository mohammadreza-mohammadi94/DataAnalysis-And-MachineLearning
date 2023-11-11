# Udemy Courses Analysis

## Overview

This repository contains a Jupyter notebook (udemy-analysis.ipynb) that analyzes a dataset of Udemy courses. The notebook explores the dataset, performs data cleaning, and conducts various analyses using Python's pandas, matplotlib, and seaborn libraries.

### Requirements
* Python 3.x  
* Pandas  
* matoplitlib
* seaborn

### Dataset

The Udemy dataset (Udemy-Dataset.csv) was loaded into the notebook using pandas. The initial examination includes displaying the first few rows, general information about the dataset, column names, and some statistical analyses.

### Data Cleaning

Data cleaning steps include checking for duplicates, correcting data types, and handling missing values. The notebook addresses the correct data type for the 'published_timestamp' column, separates date from the timestamp, and corrects the 'price' and 'content_duration' columns.

### Exploratory Data Analysis (EDA)

EDA involves visualizing data distribution, exploring unique values in the 'subjects' column, identifying subjects with the most courses, and analyzing free and paid courses. The notebook also explores top-selling courses, courses with zero subscribers, and those related to Python.

## Visualization

Various visualizations are included, such as a correlation heatmap, stacked bar plots for subject/level distribution, courses per year, box plots for outlier detection, histograms for numeric features, distribution of subjects, and a pie chart for the distribution of paid and free courses.

## Usage
The notebooks can be run directly on Google Colab or the code can be reused on other datasets.

## Conclusion

The Udemy courses analysis notebook provides insights into the dataset, offering a comprehensive overview of the courses available on Udemy and their characteristics. The analysis is documented with comments for clarity and understanding.

