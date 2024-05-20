# California Housing Analysis

This repository contains a Jupyter Notebook (`CaliforniaHousing.ipynb`) that explores the California Housing dataset. The notebook covers various aspects of data analysis including data visualization, data preprocessing, statistical analysis, and machine learning modeling.

## Dataset
The dataset used in this analysis consists of housing information for various blocks in California. It includes the following features along with the target variable `median_house_value`:

- Longitude: Geographic coordinate specifying the east–west position of the block.
- Latitude: Geographic coordinate specifying the north–south position of the block.
- Housing Median Age: Median age of houses in a block.
- Total Rooms: Total number of rooms within a block.
- Total Bedrooms: Total number of bedrooms within a block.
- Population: Total number of people residing within a block.
- Households: Total number of households for a block.
- Median Income: Median income for households within a block of houses (measured in tens of thousands of US Dollars).
- Median House Value: Median house value for households within a block (measured in US Dollars).
- Ocean Proximity: Distance from the ocean.

## Analysis Overview
The notebook covers the following aspects of data analysis:

1. Data Exploration:
   - Checking dataset information and shape.
   - Descriptive statistics of the dataset.
   - Handling missing values using SimpleImputer.

2. Exploratory Data Analysis (EDA):
   - Visualizing the distribution of features.
   - Analyzing feature correlations using a heatmap.
   - Exploring the distribution of `ocean_proximity`.
   - Scatter plots and pairplots for understanding relationships.

3. Statistical Analysis:
   - Estimating the mean of various features and the target variable.
   - Performing hypothesis testing to determine the significance of sample means.

4. Machine Learning Modeling:
   - Implementing linear regression models to understand relationships between features and the target variable.
   
5. Visualization:
   - Visualizing regression lines for feature-target relationships.

## Repository Contents
- `CaliforniaHousing.ipynb`: Jupyter Notebook containing the complete analysis.
- `housing.csv`: Dataset used for analysis.

## Dependencies
The analysis requires the following Python libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy

## Usage
1. Clone the repository.
2. Install the required dependencies.
3. Open and run the Jupyter Notebook `CaliforniaHousing.ipynb` to replicate the analysis.

