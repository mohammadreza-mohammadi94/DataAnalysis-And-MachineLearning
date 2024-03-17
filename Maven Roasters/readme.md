# Maven Roasters Coffee Shop Sales Revenue Analysis

This repository contains a Jupyter Notebook (`Maven_Roasters_DataAnalysis.ipynb`) that explores the sales revenue dataset of Maven Roasters Coffee Shop through Exploratory Data Analysis (EDA). The notebook covers various aspects of data analysis, including data cleaning, visualization, and deriving insights from the data.

## Dataset
The dataset used in this analysis contains information about sales revenue at Maven Roasters Coffee Shop, including features such as store location, product category, product type, product detail, transaction date, transaction time, transaction quantity, and unit price.

### Features:
- Store Location
- Product Category
- Product Type
- Product Detail
- Transaction Date
- Transaction Time
- Transaction Quantity
- Unit Price

## Analysis Overview
The notebook covers the following aspects of data analysis:

1. Data Import and Cleaning:
   - Importing the dataset from Google Drive.
   - Checking for missing values and duplicates.
   - Converting transaction date and time to datetime format.
   - Extracting day, month, year, and hour of transaction for analysis.
   - Calculating total sales by multiplying transaction quantity and unit price.

2. Exploratory Data Analysis (EDA):
   - Visualizing the count of products per category.
   - Identifying best-selling products and categories.
   - Analyzing the top expensive products.
   - Examining total sales per store location.
   - Investigating average sales per day and month.
   - Exploring total sales by transaction day, product category, and store location.
   - Analyzing sales trends over time using line plots.
   - Generating a word cloud from product details to visualize frequently sold items.

## Repository Contents
- `Maven_Roasters_DataAnalysis.ipynb`: Jupyter Notebook containing the complete analysis.
- `coffee-shop-sales-revenue.csv`: Dataset used for analysis.

## Dependencies
The analysis requires the following Python libraries:
- pandas
- numpy
- matplotlib
- seaborn
- wordcloud


