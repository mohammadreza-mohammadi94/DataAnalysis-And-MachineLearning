# Stock's Prices Dataset Analysis

This Python script performs an analysis on a stock dataset using pandas, numpy, matplotlib, seaborn, and scipy libraries. The dataset used is a CSV file containing stock information for a different companies over a 2006-2018. I Will complete and analyze other stock's prices over the time.

## Instructions

1. Make sure you have Python installed on your machine.
2. Install the required dependencies using the following command:

```bash
pip install pandas numpy matplotlib seaborn scipy
```

3. Download the stock dataset (CSV file) and update the file path in the script.

4. Run the script using the jupyter notebook.


## Overview

The script covers the following aspects:

### 1. Dataset Overview
 
- Loading the dataset into a pandas DataFrame.
- Displaying the first few rows of the dataset.
- Showing information about the dataset, including data types and missing values.

### 2. Data Distribution

- Visualizing the distribution of closing, opening, high, and low prices using histograms.

### 3. Prices Over Time

- Plotting the changes in closing, opening, high, and low prices over time.

### 4. Correlation Matrix

- Displaying the correlation matrix to identify relationships between different features.

### 5. Boxplot - To Identify Outliers

- Utilizing boxplots to identify outliers in closing, opening, high, and low prices.


### 6. Volume Trends

- Displaying the trend of trading volume over time.

### 7. Season Decomposition

- Using seasonal decomposition to analyze the seasonality component in closing prices.

### 8. Scatter Plots

- Creating scatter plots to explore relationships between different features (e.g., volume vs. closing price, high vs. closing price, low vs. closing price).

### 9. Inferential Analysis

- Estimating the mean closing price and constructing confidence intervals.

### 10. Mean Hypothesis Test (t-test)

- Performing a one-sample t-test to test if the estimated population mean is equal to the true population mean.
