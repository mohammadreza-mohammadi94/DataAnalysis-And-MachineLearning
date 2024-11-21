#---------------------------------------------------------------------#
#                     Code By Mohammadreza Mohammadi                  #
#      Github:   https://github.com/mohammadreza-mohammadi94          #
#      LinkedIn: https://www.linkedin.com/in/mohammadreza-mhmdi/      #
#---------------------------------------------------------------------#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import tqdm

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, classification_report

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.ensemble import IsolationForest

"""# 3. Import Dataset <a id=3></a>

***Define Columns***
"""

def define_cols(sensor_count):
    """
    Define column names based on the number of sensors.
    """
    base_columns = ["unit_number", "time_in_cycle", "ops_setting_1", "ops_setting_2", "ops_setting_3"]
    # sensor_columns = [f"sensor_measurement_{i}" for i in range(1, sensor_count + 1)]
    sensor_columns = ["(Fan inlet temperature) (◦R)",
            "(LPC outlet temperature) (◦R)",
            "(HPC outlet temperature) (◦R)",
            "(LPT outlet temperature) (◦R)",
            "(Fan inlet Pressure) (psia)",
            "(bypass-duct pressure) (psia)",
            "(HPC outlet pressure) (psia)",
            "(Physical fan speed) (rpm)",
            "(Physical core speed) (rpm)",
            "(Engine pressure ratio(P50/P2)",
            "(HPC outlet Static pressure) (psia)",
            "(Ratio of fuel flow to Ps30) (pps/psia)",
            "(Corrected fan speed) (rpm)",
            "(Corrected core speed) (rpm)",
            "(Bypass Ratio) ",
            "(Burner fuel-air ratio)",
            "(Bleed Enthalpy)",
            "(Required fan speed)",
            "(Required fan conversion speed)",
            "(High-pressure turbines Cool air flow)",
            "(Low-pressure turbines Cool air flow)"
        ]
    return base_columns + sensor_columns

"""***Import Datasets***"""

BASE_DIR = "/content/CMaps/"

def load_datasets(dataset_id, sensor_count):
    # File paths
    train_file = os.path.join(BASE_DIR, f"train_{dataset_id}.txt")
    test_file = os.path.join(BASE_DIR, f"test_{dataset_id}.txt")
    rul_file = os.path.join(BASE_DIR, f"RUL_{dataset_id}.txt")

    # Define column names
    columns = define_cols(sensor_count)

    # Load training data
    train_data = pd.read_csv(
        train_file,
        delim_whitespace=True,  # Correct handling of space-separated values
        header=None,
        names=columns
    )
    # Drop unnecessary blank columns
    train_data.dropna(axis=1, inplace=True)

    # Load test data
    test_data = pd.read_csv(
        test_file,
        delim_whitespace=True,  # Correct handling of space-separated values
        header=None,
        names=columns
    )
    # Drop unnecessary blank columns
    test_data.dropna(axis=1, inplace=True)

    # Load RUL values for the test set
    true_rul = pd.read_csv(rul_file, delim_whitespace=True, header=None, names=["RUL"])

    # Reset indices to ensure proper alignment
    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    true_rul.reset_index(drop=True, inplace=True)

    # Return datasets
    return train_data, test_data, true_rul

# Load datasets
datasets = {}

for dataset_id, sensor_count in zip(["FD001", "FD002", "FD003", "FD004"], [21, 21, 26, 26]):
    train_data, test_data, true_rul = load_datasets(dataset_id,
                                                   sensor_count)
    datasets[dataset_id] = {
        "train_data": train_data,
        "test_data": test_data,
        "true_rul": true_rul
        }

# FD001
fd001_train = datasets["FD001"]["train_data"]
fd001_test = datasets["FD001"]["test_data"]
fd001_rul = datasets["FD001"]["true_rul"]

fd001_rul['unit_number'] = fd001_rul.index + 1

# FD002
fd002_train = datasets["FD002"]["train_data"]
fd002_test = datasets["FD002"]["test_data"]
fd002_rul = datasets["FD002"]["true_rul"]

fd002_rul['unit_number'] = fd002_rul.index + 1

# FD003
fd003_train = datasets["FD003"]["train_data"]
fd003_test = datasets["FD003"]["test_data"]
fd003_rul = datasets["FD003"]["true_rul"]

fd003_rul['unit_number'] = fd003_rul.index + 1

# FD004
fd004_train = datasets["FD004"]["train_data"]
fd004_test = datasets["FD004"]["test_data"]
fd004_rul = datasets["FD004"]["true_rul"]

fd004_rul['unit_number'] = fd004_rul.index + 1

"""# 4. Analyze Dataset <a id=4></a>

***FD001***
"""

fd001_train.info()

fd001_train.describe().T

fd001_test.info()

fd001_test.describe().T

fd001_rul.info()

"""***FD002***"""

fd002_train.info()

fd002_train.describe().T

"""***FD003***"""

fd003_train.info()

fd003_train.describe().T

fd003_test.info()

fd003_test.describe().T

"""***FD004***"""

fd004_train.info()

fd004_train.describe().T

fd004_test.info()

fd004_test.describe().T

"""## 4.1 Merge Datasets <a id=4.1></a>"""

FD001 = fd001_train.merge(fd001_rul, on='unit_number', how='left')
FD002 = fd002_train.merge(fd002_rul, on='unit_number', how='left')
FD003 = fd003_train.merge(fd003_rul, on='unit_number', how='left')
FD004 = fd004_train.merge(fd004_rul, on='unit_number', how='left')

FD001.head(1)

"""## 4.2 Define Sensors Dcitionary <a id=4.2></a>"""

dict_list = ["(Fan inlet temperature) (◦R)",
            "(LPC outlet temperature) (◦R)",
            "(HPC outlet temperature) (◦R)",
            "(LPT outlet temperature) (◦R)",
            "(Fan inlet Pressure) (psia)",
            "(bypass-duct pressure) (psia)",
            "(HPC outlet pressure) (psia)",
            "(Physical fan speed) (rpm)",
            "(Physical core speed) (rpm)",
            "(Engine pressure ratio(P50/P2)",
            "(HPC outlet Static pressure) (psia)",
            "(Ratio of fuel flow to Ps30) (pps/psia)",
            "(Corrected fan speed) (rpm)",
            "(Corrected core speed) (rpm)",
            "(Bypass Ratio) ",
            "(Burner fuel-air ratio)",
            "(Bleed Enthalpy)",
            "(Required fan speed)",
            "(Required fan conversion speed)",
            "(High-pressure turbines Cool air flow)",
            "(Low-pressure turbines Cool air flow)"
        ]

sensor_dictionary={}

i=1
for x in dict_list :
    sensor_dictionary['sensor_measurement_'+str(i)] = x
    i+=1

sensor_dictionary

"""# 5. Explore Dataset <a id=5></a>

## 5.1 Distribution Of Engine Life & RUL <a id=5.1></a>
"""

life_distribution = fd001_train.groupby('unit_number')['time_in_cycle'].max()
sns.set_style('dark')
plt.figure(figsize=(6, 5))
life_distribution.hist(bins=30, color='darkblue', grid=False)
plt.title('Distribution of Engine Life - FD001')
plt.xlabel('Cycles')
plt.ylabel('Number of Engines')
plt.show()

plt.figure(figsize=(6, 5))
sns.set_style('dark')
plt.hist(fd001_rul['RUL'], bins=20, color='darkblue', alpha=0.7)
plt.title('Distribution of True RUL - FD001')
plt.xlabel('RUL')
plt.ylabel('Frequency')
plt.show()

life_distribution = fd002_train.groupby('unit_number')['time_in_cycle'].max()
sns.set_style('dark')
plt.figure(figsize=(6, 5))
life_distribution.hist(bins=30, color='darkblue', grid=False)
plt.title('Distribution of Engine Life - FD002')
plt.xlabel('Cycles')
plt.ylabel('Number of Engines')
plt.show()

plt.figure(figsize=(6, 5))
sns.set_style('dark')
plt.hist(fd002_rul['RUL'], bins=20, color='darkblue', alpha=0.7)
plt.title('Distribution of True RUL - FD002')
plt.xlabel('RUL')
plt.ylabel('Frequency')
plt.show()

life_distribution = FD003.groupby('unit_number')['time_in_cycle'].max()
sns.set_style('dark')
plt.figure(figsize=(6, 5))
life_distribution.hist(bins=30, color='darkblue', grid=False)
plt.title('Distribution of Engine Life - FD003')
plt.xlabel('Cycles')
plt.ylabel('Number of Engines')
plt.show()

plt.figure(figsize=(6, 5))
sns.set_style('dark')
plt.hist(fd003_rul['RUL'], bins=20, color='darkblue', alpha=0.7)
plt.title('Distribution of True RUL - FD003')
plt.xlabel('RUL')
plt.ylabel('Frequency')
plt.show()

life_distribution = fd004_train.groupby('unit_number')['time_in_cycle'].max()
sns.set_style('dark')
plt.figure(figsize=(6, 5))
life_distribution.hist(bins=30, color='darkblue', grid=False)
plt.title('Distribution of Engine Life - FD004')
plt.xlabel('Cycles')
plt.ylabel('Number of Engines')
plt.show()

plt.figure(figsize=(6, 5))
sns.set_style('dark')
plt.hist(fd004_rul['RUL'], bins=20, color='darkblue', alpha=0.7)
plt.title('Distribution of True RUL - FD004')
plt.xlabel('RUL')
plt.ylabel('Frequency')
plt.show()

"""## 5.2 Turbofan Engines Lifetime <a id=5.2></a>"""

max_time_cycles=fd001_train.groupby('unit_number').max()
sns.set_style('dark')
plt.figure(figsize=(10,40))
max_time_cycles['time_in_cycle'].sort_values(ascending=False).plot(kind='barh',width=0.5, stacked=True, align='center', color='darkblue')
plt.title('Turbofan Engines LifeTime', fontweight='bold', size=25)
plt.xlabel('Time Cycle',fontweight='bold',size=20)
plt.ylabel('Unit Number',fontweight='bold',size=20)
plt.tight_layout()
plt.show()

"""## 5.3 Visualize The Behavior Of Sensors By Unit Number <a id=5.3></a>

### FD001
"""

def plot_unit_sensors(data,
                      unit_id,
                      line_color='red'):
    """
    Plots all sensor measurements versus time_in_cycle for a specific unit ID.

    Parameters:
    - data: DataFrame containing the dataset (e.g., train_data).
    - unit_id: Integer representing the specific unit_number to plot.
    - sensor_dict: Dictionary mapping sensor column names to their descriptions.
    """
    unit_data = data[data['unit_number'] == unit_id]
    if unit_data.empty:
        print(f"No data found for Unit ID: {unit_id}")

    # Initialize the plot
    plt.figure(figsize=(10, 7))

    # Iterate over sensor names to plot each one
    for i, name in enumerate(data.columns[1:], 1):
        plt.figure(figsize=(10, 5))
        subset = data[data['unit_number'] == unit_id]
        plt.plot(subset['time_in_cycle'],
                 subset[name],
                 label=f'Unit {unit_id}',
                 color=line_color)

        # Define informations on plot
        plt.title(f"Sensor Measurement of Unit ID {unit_id}", fontsize=16, fontweight='bold')
        plt.xlabel("Time in Cycle", fontsize=12, fontweight='bold')
        plt.ylabel(f"Sensor Measurement {name}", fontsize=12, fontweight='bold')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

plot_unit_sensors(data=FD001,
                  unit_id=3)

# Lets check another unit number
plot_unit_sensors(data=FD001,
                  unit_id=100,
                  line_color='darkblue')

"""## 5.4 Sensor's Distribution <a id=5.4></a>"""

def hist(data,
        fig_size=(20, 22),
        bins=30,
        color='darkblue'):
    """
    Plots regression plots for each sensor variable against the Remaining Useful Life (RUL).

    This function generates a grid of scatter plots with regression lines to visualize
    the relationship between each sensor's measurements and the RUL. Each subplot corresponds
    to a sensor variable, with labels provided by the `sensor_dict`.

    Parameters:
    ----------
    data : pandas.DataFrame
        The dataset containing sensor measurements and the `RUL` column.
        Each column represents a sensor variable.
    sensor_dict : dict
        A dictionary where keys correspond to column names of sensors in the `data` DataFrame,
        and values are descriptive names of the sensors for labeling the plots.
    fig_size : tuple, optional (default=(20, 25))
        The size of the figure to be created (width, height).

    Returns:
    -------
    None
        Displays the regression plots in a grid format.

    Notes:
    ------
    - The grid layout is predefined as 7 rows and 3 columns, which suits up to 21 sensors.
    - If the `sensor_dict` contains more than 21 entries, additional plots will be omitted.
    - The regression line is colored red with a linewidth of 2 by default.
    - This visualization is helpful for identifying linear or non-linear relationships between
      sensor variables and RUL.

    Example:
    --------
    >>> regplot(data=FD001_train, sensor_dict=sensor_dictionary)
    """

    rows, cols = 9, 3
    plt.figure(figsize=fig_size)

    for i, name in enumerate(data.columns[1:], 1):
        plt.subplot(rows, cols, i)
        sns.histplot(data=data,
                     x=name,
                    kde=True,
                    bins=bins,
                    color=color)
        plt.title(f"Distribution Of {name}", fontsize=12, fontweight='bold')
        plt.xlabel(name, fontsize=10, fontweight='bold')
        plt.ylabel("Frequency", fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.show()

"""### FD001"""

hist(FD001)

"""### FD003

We can perfome same operation for other dataset as well.
"""

hist(data=FD003)

"""## 5.5 Boxplot <a id=5.5></a>"""

def boxplot(data,
            fig_size=(10, 30)):
    """
    Plots boxplots for each sensor variable in the dataset.

    This function generates a grid of boxplots for visualizing the distribution and
    outliers of sensor measurements in the dataset. Each subplot corresponds to a
    sensor variable, with labels provided by the `sensor_dict`.

    Parameters:
    ----------
    data : pandas.DataFrame
        The dataset containing sensor measurements. Each column represents a sensor variable.
    sensor_dict : dict
        A dictionary where keys correspond to column names of sensors in the `data` DataFrame,
        and values are descriptive names of the sensors for labeling the plots.
    fig_size : tuple, optional (default=(15, 30))
        The size of the figure to be created (width, height).

    Returns:
    -------
    None
        Displays the boxplots in a grid format.

    Notes:
    ------
    - The grid layout is predefined as 7 rows and 3 columns, which suits up to 21 sensors.
    - If the `sensor_dict` contains more than 21 entries, additional plots will be omitted.
    - Outliers, if present in the data, will be visualized as individual points on the boxplot.

    Example:
    --------
    >>> boxplot(data=FD001_train, sensor_dict=sensor_dictionary)
    """
    rows, cols = 9, 3
    plt.figure(figsize=fig_size)

    for i, name in enumerate(data.columns[1:], 1):
        plt.subplot(rows, cols, i)
        sns.boxplot(data=data,
                    y=name)
        plt.title(f"{name} Boxplot", fontsize=12, fontweight='bold')
        plt.xlabel(name, fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.show()

boxplot(data=FD001)

"""## 5.6 Regplot <a id=5.6></a>"""

def regplot(data,
            fig_size=(20, 25)):
    """
    Plots boxplots for each sensor variable in the dataset.

    This function generates a grid of boxplots for visualizing the distribution and
    outliers of sensor measurements in the dataset. Each subplot corresponds to a
    sensor variable, with labels provided by the `sensor_dict`.

    Parameters:
    ----------
    data : pandas.DataFrame
        The dataset containing sensor measurements. Each column represents a sensor variable.
    sensor_dict : dict
        A dictionary where keys correspond to column names of sensors in the `data` DataFrame,
        and values are descriptive names of the sensors for labeling the plots.
    fig_size : tuple, optional (default=(15, 30))
        The size of the figure to be created (width, height).

    Returns:
    -------
    None
        Displays the boxplots in a grid format.

    Notes:
    ------
    - The grid layout is predefined as 7 rows and 3 columns, which suits up to 21 sensors.
    - If the `sensor_dict` contains more than 21 entries, additional plots will be omitted.
    - Outliers, if present in the data, will be visualized as individual points on the boxplot.

    Example:
    --------
    >>> boxplot(data=FD001_train, sensor_dict=sensor_dictionary)
    """
    rows, cols = 9, 3
    plt.figure(figsize=fig_size)

    for i, name in enumerate(data.columns[1:], 1):
        plt.subplot(rows, cols, i)
        sns.regplot(data=data,
                    x='RUL',
                    y=name,
                    line_kws={'color': 'red', 'linewidth': 2})
        plt.title(f"{name} Vs RUL", fontsize=12, fontweight='bold')
        plt.xlabel('RUL', fontsize=10, fontweight='bold')
        plt.ylabel(name, fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.show()

regplot(data=FD001)

"""### 5.6.1 Analysis Of regplot <a id=5.6.1></a>
* Most variables do not exhibit a clear linear relationship with the Remaining Useful Life (RUL).
* The red regression lines are almost flat in most cases, indicating a weak or no correlation between the sensor values and the RUL.
* Some variables might contain a significant amount of noise, obscuring their relationship with RUL.

## 5.7 Rolling Mean (Smoothing Plot) <a id=5.7></a>
"""

def smoothing_plot(data,
                   fig_size=(10, 5),
                   from_unit=0,
                   to_unit=5):
    """
    Plots smoothed sensor data over time for different units.

    This function takes sensor data and applies a rolling mean smoothing technique
    to the data for each unit, plotting the smoothed values over time in a figure.
    The function handles multiple sensors and allows customization of the number of
    units and figure size.

    Parameters:
    - data (pandas.DataFrame): The dataset containing sensor data.
    - sensor_dict (dict): A dictionary where keys are sensor names and values are
                          the names to display in the plot's title.
    - fig_size (tuple): The size of the figure (default is (10, 5)).
    - n_unit (int): The number of units to plot (default is 5). Only the first `n_unit`
                    units will be plotted.

    Returns:
    - None: This function directly plots the figures for each sensor.

    Example:
    - smoothing_plot(FD001, {'sensor1': 'Sensor 1', 'sensor2': 'Sensor 2'})
    """
    for name in data.columns[1:]:
        plt.figure(figsize=fig_size)
        for unit in FD001['unit_number'].unique()[from_unit:to_unit]:
            subset = FD001[FD001['unit_number'] == unit]
            plt.plot(subset['time_in_cycle'], subset[name].rolling(window=10).mean(), label=f"Unit {unit}")
        plt.title(f"Smoothed {name} vs Time in Cycle")
        plt.xlabel("Time in Cycle")
        plt.ylabel(name)
        plt.legend()
        plt.show()

smoothing_plot(FD001)

smoothing_plot(FD001, from_unit=15, to_unit=18)

"""## 5.8 Compute Rolling Aggregated Metrics <a id=5.8></a>"""

FD001_copy = FD001.copy()

def compute_rolling_features(df, sensor_cols, window_size=5):
    """
    Computes rolling mean and standard deviation for specified sensor columns
    in the dataframe, grouped by 'unit_number'.

    Parameters:
    ----------
    df : pandas.DataFrame
        The input dataframe containing sensor data and 'unit_number'.
    sensor_cols : list
        List of column names corresponding to sensor data for which rolling
        features will be computed.
    window_size : int, optional (default=5)
        The size of the rolling window over which metrics are computed.

    Returns:
    -------
    pandas.DataFrame
        A dataframe with new columns added for rolling mean and standard deviation
        for each sensor column in `sensor_cols`. The new column names are of the
        format `<sensor>_roll_mean` and `<sensor>_roll_std`.

    Example:
    --------
    sensors = ['sensor_1', 'sensor_2']
    df = compute_rolling_features(df, sensors, window_size=3)
    """
    for sensor in sensor_cols:
        df[f"{sensor}_roll_mean"] = df.groupby("unit_number")[sensor].rolling(window=window_size).mean().reset_index(drop=True)
        df[f"{sensor}_roll_std"] = df.groupby("unit_number")[sensor].rolling(window=window_size).std().reset_index(drop=True)
    return df

FD001_statistical_info = compute_rolling_features(FD001_copy, dict_list)

def plot_rolling_features(data,
                        unit_id,
                        sensor,
                        figsize=(10, 5)):
    """
    Visualizes raw sensor data and its rolling metrics (mean and standard deviation)
    for a single unit from the dataset.

    Parameters:
    ----------
    data : pandas.DataFrame
        The dataframe containing sensor data and precomputed rolling features.
    unit_id : int
        The ID of the unit to visualize.
    sensor : str
        The column name of the sensor to plot.
    figsize : tuple, optional (default=(10, 5))
        Figure size for the plot.

    Returns:
    -------
    None
        Displays a plot of the raw sensor data, rolling mean, and rolling standard
        deviation for the specified unit and sensor.

    Example:
    --------
    single_unit_rolling(data=df, unit_id=1, sensor="sensor_1")
    """
    unit_data = data[data["unit_number"] == unit_id]
    plt.figure(figsize=(12, 6))
    plt.plot(unit_data["time_in_cycle"], unit_data[sensor], label="Raw Sensor Data")
    plt.plot(unit_data["time_in_cycle"], unit_data[f"{sensor}_roll_mean"], label="Rolling Mean")
    plt.fill_between(
        unit_data["time_in_cycle"],
        unit_data[f"{sensor}_roll_mean"] - unit_data[f"{sensor}_roll_std"],
        unit_data[f"{sensor}_roll_mean"] + unit_data[f"{sensor}_roll_std"],
        color='gray', alpha=0.3, label="Rolling Std Dev"
    )
    plt.xlabel("Time in Cycle")
    plt.ylabel(sensor)
    plt.title(f"Rolling Metrics for Unit {unit_id}")
    plt.legend()
    plt.show()

plot_rolling_features(FD001_statistical_info, 61, "(HPC outlet temperature) (◦R)")

plot_rolling_features(FD001_statistical_info, 11, "(HPC outlet temperature) (◦R)")

plot_rolling_features(FD001_statistical_info, 17, "(HPC outlet temperature) (◦R)")

plot_rolling_features(FD001_statistical_info, 1, "(Corrected fan speed) (rpm)")

plot_rolling_features(FD001_statistical_info, 1, "(Corrected core speed) (rpm)")

"""## 5.9 Clustering Based on Operational Settings <a id=5.9></a>"""

operations_settings = ['ops_setting_1', 'ops_setting_2', 'ops_setting_3']

def add_operational_clusters(data,
                             ops_cols,
                             n_clusters=3):
    """
    Adds operational condition clusters to the dataset based on specified columns.

    This function applies K-Means clustering on scaled operational setting columns
    to group similar operational conditions into distinct clusters. The resulting
    cluster labels are added as a new column (`operational_cluster`) in the dataset.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing operational data.
    ops_cols : list of str
        List of column names representing the operational settings to be used for clustering.
    n_clusters : int, optional, default=3
        The number of clusters to form.

    Returns:
    -------
    pd.DataFrame
        The input DataFrame with an additional column `operational_cluster` indicating
        the cluster assignments for each data point.

    Example:
    -------
    >>> ops_columns = ["ops_setting_1", "ops_setting_2", "ops_setting_3"]
    >>> df = add_operational_clusters(df, ops_columns, n_clusters=3)
    >>> print(df["operational_cluster"].unique())
    [0, 1, 2]
    """
    scaler = StandardScaler()
    ops_data_scaled = scaler.fit_transform(data[ops_cols])

    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(ops_data_scaled)

    data["operational_cluster"] = clusters
    return data

FD001_statistical_info = add_operational_clusters(FD001_statistical_info, operations_settings)

def plot_operational_clusters(df):
    """
    This function plots a scatter plot of two operational settings (ops_setting_1 and ops_setting_2)
    from the given dataframe and colors the points according to the operational clusters
    identified in the 'operational_cluster' column.

    The plot helps visualize how the data points are distributed across different operational clusters,
    providing insights into the relationship between operational settings and cluster membership.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe containing the operational settings (ops_setting_1, ops_setting_2, etc.)
        and the 'operational_cluster' column indicating the cluster assignments.

    Returns:
    --------
    None
        Displays the scatter plot showing the relationship between the selected operational settings
        and the operational clusters.
    """
    # Create a scatter plot with different colors for each operational cluster
    plt.figure(figsize=(8, 4))

    # Scatter plot of two operational settings, color-coded by the operational cluster
    sns.scatterplot(data=df, x='ops_setting_1', y='ops_setting_2', hue='operational_cluster', palette='Set1', marker='o')

    # Adding labels and title
    plt.xlabel('Operational Setting 1')
    plt.ylabel('Operational Setting 2')
    plt.title('Operational Clusters Based on Settings')

    # Show the plot
    plt.legend(title='Cluster', loc='upper right')
    plt.show()

plot_operational_clusters(FD001_statistical_info)

def assess_clusters_on_rul(df):
    """
    This function assesses how operational clusters (from the 'operational_cluster' column)
    affect the Remaining Useful Life (RUL) by grouping the data by clusters and visualizing
    the distribution of RUL within each cluster.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe containing the 'operational_cluster' and 'RUL' columns.

    Returns:
    --------
    None
        Displays summary statistics and a box plot for RUL by operational cluster.
    """

    # 1. Compute summary statistics of RUL for each operational cluster
    cluster_stats = df.groupby("operational_cluster")["RUL"].describe()
    print(cluster_stats)

    # 2. Plot a box plot of RUL distribution for each operational cluster
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="operational_cluster", y="RUL", palette='Set1')
    plt.title('RUL Distribution by Operational Cluster')
    plt.xlabel('Operational Cluster')
    plt.ylabel('Remaining Useful Life (RUL)')
    plt.show()

    # 3. Correlation between operational clusters and RUL
    correlation = df[["operational_cluster", "RUL"]].corr()
    print("\nCorrelation between Operational Cluster and RUL:")
    print(correlation)

assess_clusters_on_rul(FD001_statistical_info)

"""## 5.10 Correlation Matrix <a id=5.10></a>"""

correlation_matrix = FD001[dict_list + ['RUL']].corr()

# Visualize correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation of Sensors with RUL')
plt.show()

"""# 6. Anomaly Detection In Sensor Behavior <a id=6></a>"""

def detect_sensor_anomalies(data,
                            sensor_cols,
                            figsize=(8, 5),
                            standardize=False,
                            contamination=0.05):
    """
    Detects anomalies in sensor data using the Isolation Forest algorithm and visualizes the results.

    Parameters:
    - data (DataFrame): The input DataFrame containing sensor data.
    - sensor_cols (list): A list of sensor columns to be analyzed for anomalies.
    - figsize (tuple): The size of the plot for visualization. Default is (16, 10).
    - contamination (float): The proportion of data points expected to be anomalies. Default is 0.05 (5%).

    Returns:
    - data (DataFrame): The original DataFrame with a new 'anomaly' column indicating detected anomalies.
    """

    # Train Isolation Forest to detect anomalies
    iso_forest = IsolationForest(contamination=contamination, random_state=42)

    if standardize:
        standard_scaler = StandardScaler()
        data[sensor_cols] = standard_scaler.fit_transform(data[sensor_cols])

    # Fit the model and predict anomalies for each sensor
    for sensor in sensor_cols:
        anomalies = iso_forest.fit_predict(data[[sensor]])

        # -1 indicates anomalies, 1 indicates normal data
        data[f'anomaly_{sensor}'] = anomalies

        # Plot the anomalies for the current sensor
        plt.figure(figsize=figsize)
        plt.plot(data['time_in_cycle'], data[sensor], label=f'{sensor} Sensor Data')
        plt.scatter(data['time_in_cycle'][data[f'anomaly_{sensor}'] == -1],
                    data[sensor][data[f'anomaly_{sensor}'] == -1],
                    color='red', label='Anomalies')
        plt.xlabel('Time in Cycle')
        plt.ylabel(f'{sensor} Sensor Value')
        plt.legend()
        plt.title(f'Anomaly Detection with Isolation Forest - {sensor}')
        plt.show()

FD001_anomaly = detect_sensor_anomalies(FD001_copy, dict_list)
FD001_anomaly



