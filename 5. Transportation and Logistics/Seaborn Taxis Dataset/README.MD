# Taxis Dataset Analysis using Seaborn

This notebook presents an analysis of the Taxis dataset using various visualization techniques with Seaborn library in Python.

## Dataset Overview

- The dataset contains information about taxi trips, including the number of passengers, distance traveled, fare amount, tip amount, tolls paid, and total fare.

### Data Cleaning

- Checked dataset information, missing values, and descriptive statistics.
- Dropped rows with missing values.
- Detected and removed outliers using boxplots and the Interquartile Range (IQR) method.

### Feature Engineering

- Extracted day, month, and year from the 'pickup' datetime feature.
- Calculated the tip percentage of the fare.

## Exploratory Data Analysis (EDA)

### Visualizations

- Boxplots to visualize outliers in numerical features.
- Histograms and Kernel Density Estimation (KDE) plots to visualize distributions.
- Frequency tables to display the count of categorical features.
- Line plots to show hourly and daily demand for taxi trips.
- Barplots to visualize the distribution of pickup zones, dropoff zones, and taxi colors.
- Scatter plots with regression lines to analyze the relationships between features and the total fare.

### Insights

- The analysis revealed the distribution of taxi trips across different pickup and dropoff zones, boroughs, and colors.
- It highlighted the hourly and daily demand patterns, helping to identify peak hours and days.
- Correlation heatmap provided insights into the relationships between numerical features.
- Regression plots showed the relationships between the total fare and other numerical features, such as distance, fare amount, tip amount, and tolls paid.

## Conclusion

- The analysis provides valuable insights into taxi trip patterns, fare distributions, and factors influencing fare amounts.
- These insights can be used for further analysis, predictive modeling, or optimization of taxi services.
