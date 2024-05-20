# Automobile Dataset Analysis

This repository contains a Jupyter Notebook (`Automobile.ipynb`) that explores the Automobile dataset. The notebook covers various aspects of data analysis including data cleaning, visualization, and exploratory data analysis (EDA).

## Dataset
The dataset used in this analysis contains information about automobiles, including various features such as make, fuel type, engine size, horsepower, price, etc.

### Features:
- make
- fuel_type
- num_of_doors
- body_style
- drive_wheels
- engine_location
- wheel_base
- curb_weight
- engine_type
- num_of_cylinders
- engine_size
- fuel_system
- compression_ratio
- peak_rpm
- city_mpg
- highway_mpg
- horsepower
- bore
- stroke
- normalized_losses
- price
- symboling

## Analysis Overview
The notebook covers the following aspects of data analysis:

1. Dataset Analysis:
   - Checking dataset information, shape, and missing values.
   - Descriptive statistics of the dataset.
   - Data Cleaning: Handling missing values and replacing "?" with appropriate values.
   
2. Outlier Detection and Imputation:
   - Identifying outliers using Interquartile Range (IQR).
   - Replacing outliers with the median value.

3. Exploratory Data Analysis (EDA):
   - Visualizing data using various plots like bar plots, histograms, scatter plots, box plots, pair plots, and count plots.
   - Analyzing the relationships between features.

4. Visualization:
   - Visualizing the distribution of categorical features.
   - Creating histograms and box plots for numeric features.
   - Plotting a correlation heatmap.
   - Generating scatter plots and pair plots for numeric features.
   - Utilizing facet grids for visualization based on fuel type and number of doors.

## Repository Contents
- `Automobile.ipynb`: Jupyter Notebook containing the complete analysis.
- `Automobile_data.csv`: Dataset used for analysis.

## Dependencies
The analysis requires the following Python libraries:
- pandas
- numpy
- matplotlib
- seaborn

## Usage
1. Clone the repository.
2. Install the required dependencies.
3. Open and run the Jupyter Notebook `Automobile.ipynb` to replicate the analysis.

