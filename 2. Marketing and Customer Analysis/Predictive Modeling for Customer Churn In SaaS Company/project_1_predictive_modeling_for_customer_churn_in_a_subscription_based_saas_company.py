#---------------------------------------------------------------------#
#                     Code By Mohammadreza Mohammadi                  #
#      Github:   https://github.com/mohammadreza-mohammadi94          #
#      LinkedIn: https://www.linkedin.com/in/mohammadreza-mhmdi/      #
#---------------------------------------------------------------------#


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')



df = pd.read_excel('/content/Telco_customer_churn.xlsx')
df.head()

"""# 3. Data Description <a id=3></a>"""

df.info()


df.describe().T

"""# 4. Frequently Used Methods <a id=4></a>"""

# Define a dictionary in order to use for plots to
# have better visualization.

# For plot's title
plot_title_dict = {'font': 'DejaVu Sans', 'weight': 'bold', 'fontsize': 15, 'color': 'darkblue'}
# For plot's axis
plot_axis_dict = {'font': 'DejaVu Sans', 'weight': 'bold', 'fontsize': 12, 'color': 'darkred'}

def plot_categorical_bar(df,
                         categorical_column,
                         rot=None,
                         bar_width=0.5,
                         palette='dark',
                         figsize=(8, 5)):
    """
    This function creates a bar plot for a categorical variable using Seaborn with a deep color palette.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    categorical_column (str): The name of the categorical column to plot.

    Returns:
    None: Displays the bar plot.
    """
    # Set the Seaborn style for the plot
    sns.set(style="darkgrid")

    # Create the bar plot
    plt.figure(figsize=figsize)
    bar_plot = sns.countplot(x=categorical_column,
                             data=df,
                             width=bar_width,
                             palette=palette)

    # Add title and labels
    plt.title(f'Distribution of {categorical_column}',
              fontdict = plot_title_dict)
    plt.xlabel(categorical_column,
              fontdict = plot_axis_dict)
    plt.ylabel('Count',
              fontdict = plot_axis_dict)
    # Rotate x-axis labels for better readability (optional, depending on the data)
    plt.xticks(rotation=rot, ha='right')

    # Annotate counts above bars
    for p in bar_plot.patches:
        bar_plot.annotate(format(p.get_height(), '.0f'),
                          (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center', va='center',
                          xytext=(0, 5),
                          textcoords='offset points',
                          fontweight='bold',
                          color='black')
    # Show the plot
    plt.tight_layout()
    plt.show()

def hist_plot(
            df,
            x,
            rot=None,
            bins=150,
            figsize=(8, 5)):
    """
    Plots a histogram for a specified numerical column from a DataFrame.

    This function creates a histogram using Seaborn's `histplot` to show the
    distribution of a numerical variable, with options to customize the
    number of bins, figure size, and rotation of x-axis labels.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data to plot.
    x : str
        The name of the numerical column to plot the histogram for.
    rot : int or None, optional
        The angle to rotate the x-axis labels, useful for readability with long labels.
    bins : int, default=150
        The number of bins for the histogram.
    figsize : tuple, default=(8, 5)
        The width and height of the figure in inches.

    Returns:
    --------
    None
        Displays the histogram plot with title and axis labels formatted based on predefined style dictionaries.
    """
    plt.figure(figsize=figsize)
    sns.set_style('darkgrid')

    # Use sns.histplot instead of sns.distplot
    sns.histplot(data=df, x=x, bins=bins)

    # Add title and labels with font styling
    plt.title(f"Distribution of {x}", fontdict=plot_title_dict)
    plt.xlabel(f"{x}", fontdict=plot_axis_dict)
    plt.ylabel("Frequency", fontdict=plot_axis_dict)

    # Adjust layout
    plt.tight_layout()
    plt.xticks(rotation=rot)  # Rotate x-ticks if specified
    plt.show()

def scatter_plot(df,
                 x_column,
                 y_column,
                 hue_column=None,
                 size=5,
                 palette='dark'):
    """
    Plots a scatter plot using Seaborn to show the relationship between two variables.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data to plot.
    x_column : str
        The name of the column for the x-axis.
    y_column : str
        The name of the column for the y-axis.
    hue_column : str or None, optional
        The column to use for color encoding, useful for adding an additional categorical variable.
    size : int, default=5
        The size of the markers in the scatter plot.
    palette : str, default='rocket'
        The color palette to use if hue is specified.

    Returns:
    --------
    None
        Displays the scatter plot with axis labels and a title.
    """
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df,
                    x=x_column,
                    y=y_column,
                    hue=hue_column,
                    palette=palette,
                    s=size*10)

    # Add title and labels with styling
    plt.title(f'Scatter Plot of {y_column} vs {x_column}', fontdict=plot_title_dict)
    plt.xlabel(x_column, fontdict=plot_axis_dict)
    plt.ylabel(y_column, fontdict=plot_axis_dict)

    # Display legend if hue is specified
    if hue_column:
        plt.legend(title=hue_column, loc='upper right')

    plt.tight_layout()
    plt.show()

def box_plot(df,
             x_col,
             y_col,
             hue_col=None,
             palette="dark",
             figsize=(12, 6)):
    """
    Creates a box plot to compare the distribution of a numeric variable
    across different categories and optionally grouped by a secondary
    categorical variable.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    x_col (str): The name of the categorical column on the x-axis.
    y_col (str): The name of the numeric column on the y-axis.
    hue_col (str, optional): A secondary categorical variable to further group
                             the data by color. Default is None.
    figsize (tuple, optional): Figure size for the plot. Default is (12, 6).

    Returns:
    None: Displays the box plot.

    """
    plt.figure(figsize=figsize)
    sns.boxplot(data=df,
                x=x_col,
                y=y_col,
                hue=hue_col,
                palette=palette)
    plt.title(f'{y_col} by {x_col} and {hue_col}', fontdict=plot_title_dict)
    plt.xlabel(x_col, fontdict=plot_axis_dict)
    plt.ylabel(y_col, fontdict=plot_axis_dict)
    plt.legend(title=hue_col)
    plt.show()

def count_grouped_by_status(df, status_value, filter_feature, groupby_feature="Churn Label"):
    """
    Filters the DataFrame based on a specified feature and value,
    then groups by another feature and counts the occurrences within each group.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - status_value (str): The value of the filter feature to filter on (e.g., 'Yes' or 'No').
    - groupby_feature (str): The column name to group by (e.g., 'Churn Label').
    - filter_feature (str): The column name to filter on (e.g., 'Senior Citizen').

    Returns:
    - pd.Series: A count of each group within the specified filter.
    """
    filtered_df = df[df[filter_feature] == status_value]
    grouped_counts = filtered_df.groupby(groupby_feature)[groupby_feature].count()
    return grouped_counts

"""# 5. Exploratory Data Analysis <a id=5></a>

## 5.1 Categorical Variable's Countplot <a id=5.1></a>
"""

plot_categorical_bar(df, 'Country', bar_width=0.3)

plot_categorical_bar(df, 'State', bar_width=0.3)

plot_categorical_bar(df, 'Gender')

plot_categorical_bar(df, 'Senior Citizen')

plot_categorical_bar(df, 'Partner')

plot_categorical_bar(df, 'Dependents')

plot_categorical_bar(df, 'Phone Service')

plot_categorical_bar(df, 'Multiple Lines')

plot_categorical_bar(df, 'Internet Service')

plot_categorical_bar(df, 'Online Security')

plot_categorical_bar(df, 'Online Backup')

plot_categorical_bar(df, 'Device Protection')

plot_categorical_bar(df, 'Tech Support')

plot_categorical_bar(df, 'Streaming TV')

plot_categorical_bar(df, 'Streaming Movies')

plot_categorical_bar(df, 'Contract')

plot_categorical_bar(df, 'Paperless Billing')

plot_categorical_bar(df, 'Payment Method', rot=20, figsize=(8, 6))

plot_categorical_bar(df, 'Churn Label')

"""* All customers are from Unites States and California.
* Customers are from 1129 unique cities from CA, which top ten cities are: 'Los Angeles', 'San Diego', 'San Jose', 'Sacramento', 'San Francisco','Fresno','LongBeach', 'Oakland', 'Stockton', 'Bakersfield'
* Customer's gender is almost equal. Male customers are 3555 and Female are 3488.
* There 3042 customers that living with their partner. Moreover 1627 of them are living with one of their relatives (Child, Parents,...). From this we can assume that 1928 of customers are couples. We will try to check this further.
* More than 90% of customers use Phone Service.
* 2971 of customers using multiple phone lines, while 3390 of customers do not. 682 of customers does not have any phone service.
* About 90% of customers are using Internet Service from company while 1526 of them do not use internet service. 2421 are using DSL and 3096 are using Fiber Optic.
* Distribution of Online Security indicates that about 50% of customers are not using Online Security features. Also 43% of customers does not have Online Backup.
* Furthermore from plots we can understand that 1526 of customers do not have Internet Service , so they do not have some services such as `Online Security, Online Backup, Device Protection, Tech Support, Streaming TV and Streaming Movies`. This about is about 21% of customers within this dataset.
* About 50% of customers are not using Tech Support and 30% of them are using this service. (About 21% of customer do not have Internet Service)
* Almost 40% of clients are using services such as Streaming TV and Streaming Movies while 40% of them do not use these services.(About 21% of customer do not have Internet Service)
* Contracts are as below:
    - Month to Month subscriptions are 3875 which is 55.01%
    - Two year subscriptions are 1695 which is 24.06%
    - One year subscriptions are 1473 which is 20.91%
* Subscription renewal are:
    - Mailed check: 1612 customers which is 22.88%
    - Electronic check: 2365 customers which is 33.57%
    - Bank transfer(automatic): 1544 customers which is 21.92
    - Credit card (automatic): 1522 customers which is 21.61%
* Finally, 5174 of customers remained with company which means 73.46% and 1869 of customers didnt renewed their subscription which means 26.54%.

## 5.2 Histplot <a id=5.2></a>
"""

hist_plot(df, 'Tenure Months')

hist_plot(df, 'Latitude')

hist_plot(df, 'Longitude')

hist_plot(df, 'Monthly Charges')

hist_plot(df, 'Churn Score')

hist_plot(df, 'CLTV')

"""## 5.3 Scatter Plot <a id=5.3></a>"""

# Correct datatype of Total Charges to float.
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')

scatter_plot(df, 'Monthly Charges', 'Total Charges', hue_column='Churn Label', size=3)

scatter_plot(df, 'Churn Score', 'Total Charges', hue_column='Churn Label', size=3)

"""* **Churn Score**:A value from 0-100 that is calculated using a predictive tool (e.g., IBM SPSS Modeler). The model incorporates multiple factors known to cause churn. The higher the score, the more likely the customer will churn.
  
> It seems that total charges does not have that much impact to clients to Churn, because we can see from above scatter plot that as Total Charges increase, number of clients that already Churn decreases. It show us that we have to search further in data.
"""

scatter_plot(df, 'Latitude', 'Longitude', hue_column='Churn Label', size=3)

scatter_plot(df, 'Tenure Months', 'Total Charges', hue_column='Churn Label', size=3)

"""## 5.4 Boxplot <a id=5.4></a>"""

box_plot(df, 'Gender', 'Total Charges', 'Churn Label')

count_grouped_by_status(df, 'Male', 'Gender')

count_grouped_by_status(df, 'Female', 'Gender')

"""* Average Of Total Charges for both Men and women, who churn and who not, is same."""

box_plot(df, 'Senior Citizen', 'Total Charges', 'Churn Label')

"""* Customers that are not Senior Citizen:
    * Customers that churn: Less than 1000$ per tenure. The total charges range between approx 10$ to 2000$ per tenure.
    * Customers that didnt churn: Abaout 1800$ per tenure. the total charges range between 250$ to 4000$ per tenure.
* Customers that are Senior Citizen:
    * Average total charges for customers who churn is about 1000$ and range between approx. 100$ to ~2000$.
    * Average total charges for customers who not churn is ~2700$ and range between 1200$ to 5800$.

> We can assume that Senior Citizen are more intend to stay loyal with company to keep themselfs entertained. But what about price ? does it have any impact on customers satisfaction ?
"""

count_grouped_by_status(df, 'Yes', 'Senior Citizen')

count_grouped_by_status(df, 'No', 'Senior Citizen')

"""* By analyzing above tables there are few points that we can understand.
    1. Totally 1142 of customers are Senior Citizen that 476 of them already churn. This means 41.68% of Senior Citizen were not loyal and they left.
    2. There are 5901 customer that are not Senior Citizen, while 1393 of them already churn which is about 23.60%.
> So the assumption that Senior Citizens are more loyal is rejected. One the main reason is that data of Senior and Non-Senior Citizen are imbalanced.
"""

box_plot(df, 'Partner', 'Total Charges', 'Churn Label')

"""* The number of clients that are living with a partner is almost balanced.
    * 3641 of clients are living alone.
    * 3402 of clinets are couple.
* By analyzing above chart we can see that customer who are couple are more intend to stay loyal.
"""

count_grouped_by_status(df, 'Yes', 'Partner')

count_grouped_by_status(df, 'No', 'Partner')

"""* As obiviously is clear from above tables that about half of customers who are single are very likely to Churn, while only 19.66% of customer whom are Cuple are likely to churn."""

box_plot(df, 'Phone Service', 'Total Charges', 'Churn Label')

"""
* **Churn and Total Charges Distribution**: Customers without phone service who did not churn tend to have lower total charges than those with phone service.
* **Churn Label Impact**: Customers with phone service show higher total charges on average compared to those without phone service, particularly among non-churned customers.
* **Boxplot Spread**: The range of total charges is wider for customers with phone service, indicating a higher variability in their billing compared to customers without phone service.
* **Outliers**: The "Yes" churn group with phone service has many high-charge outliers, suggesting some high-paying customers still choose to leave.
* **Median Total Charges**: For customers without phone service, the median total charge is relatively low in both churn and non-churn groups, possibly indicating less lucrative customer segments.

These insights can help target customer segments based on service type and churn behavior."""

count_grouped_by_status(df, "Yes", 'Phone Service')

count_grouped_by_status(df, "No", 'Phone Service')

box_plot(df, 'Multiple Lines', 'Total Charges', 'Churn Label')

count_grouped_by_status(df, 'No', 'Multiple Lines')

count_grouped_by_status(df, 'Yes', 'Multiple Lines')

count_grouped_by_status(df, 'No phone service', 'Multiple Lines')

box_plot(df, 'Internet Service', 'Total Charges', 'Churn Label')

"""* Churn Rate and Internet Service Type: Customers using Fiber Optic tend to have higher total charges and are more likely to churn compared to those using DSL.

* Lower Churn for No Internet Service: Customers without Internet Service have the lowest total charges and seem less likely to churn, indicating a possible relationship between internet service and churn.

* Distribution of Charges for DSL: For DSL users, churners generally have higher charges than non-churners, though this difference is not as pronounced as with Fiber Optic users.

* Charge Variability in Fiber Optic Users: Fiber Optic users show greater variability in total charges among churners, suggesting that high-cost plans might contribute to churn.

* Outliers: There are notable outliers, especially for Fiber Optic users who did not churn, showing some customers pay significantly higher charges without leaving.
"""

count_grouped_by_status(df, 'DSL', 'Internet Service')

count_grouped_by_status(df, 'Fiber optic', 'Internet Service')

count_grouped_by_status(df, 'No', 'Internet Service')

box_plot(df, 'Online Security', 'Total Charges', 'Churn Label')

"""* Customers without online security show a higher median total charge for those who have not churned compared to those who have churned, suggesting that online security could be linked to customer retention.

* Among customers with online security, the median total charges for churned and non-churned are closer in value, indicating a smaller impact on churn rate for this group.

* Customers without internet service have the lowest total charges across both churn and non-churn groups, likely due to fewer services utilized.
"""

count_grouped_by_status(df, 'Yes', 'Online Security')

count_grouped_by_status(df, 'No', 'Online Security')

count_grouped_by_status(df, 'No internet service', 'Online Security')

box_plot(df, 'Online Backup', 'Total Charges', 'Churn Label')

count_grouped_by_status(df, 'Yes', 'Online Backup')

count_grouped_by_status(df, 'No', 'Online Backup')

count_grouped_by_status(df, 'No internet service', 'Online Backup')

box_plot(df, 'Device Protection', 'Total Charges', 'Churn Label')

count_grouped_by_status(df, 'No', 'Device Protection')

count_grouped_by_status(df, 'Yes', 'Device Protection')

count_grouped_by_status(df, 'No internet service', 'Device Protection')

box_plot(df, 'Tech Support', 'Total Charges', 'Churn Label')

count_grouped_by_status(df, 'Yes', 'Tech Support')

count_grouped_by_status(df, 'No', 'Tech Support')

count_grouped_by_status(df, 'No internet service', 'Tech Support')

box_plot(df, 'Streaming TV', 'Total Charges', 'Churn Label')

count_grouped_by_status(df, 'No', 'Streaming TV')

count_grouped_by_status(df, 'Yes', 'Streaming TV')

count_grouped_by_status(df, 'No internet service', 'Streaming TV')

box_plot(df, 'Streaming Movies', 'Total Charges', 'Churn Label')

count_grouped_by_status(df, 'No', 'Streaming Movies')

count_grouped_by_status(df, 'Yes', 'Streaming Movies')

count_grouped_by_status(df, 'No internet service', 'Streaming Movies')

box_plot(df, 'Contract', 'Total Charges', 'Churn Label')

"""* The "Two year" contract shows the largest difference in total charges between "Yes" and "No" churn, indicating churn has the greatest financial impact for longer-term contracts.
* The "One year" contract has the lowest total charges overall, suggesting it may be the most cost-effective option for customers.
"""

count_grouped_by_status(df, 'Month-to-month', 'Contract')

count_grouped_by_status(df, 'Two year', 'Contract')

count_grouped_by_status(df, 'One year', 'Contract')

box_plot(df, 'Paperless Billing', 'Total Charges', 'Churn Label')

count_grouped_by_status(df, 'No', 'Paperless Billing')

count_grouped_by_status(df, 'Yes', 'Paperless Billing')

box_plot(df, 'Payment Method', 'Total Charges', 'Churn Label')

count_grouped_by_status(df, 'Mailed check', 'Payment Method')

count_grouped_by_status(df, 'Electronic check', 'Payment Method')

count_grouped_by_status(df, 'Bank transfer (automatic)', 'Payment Method')

count_grouped_by_status(df, 'Credit card (automatic)', 'Payment Method')

"""# 6. Feature Extraction <a id=6></a>"""

# Create a copy of dataframe to perfome feature extraction/generation
df_copy = df.copy()

df_copy.head()

df_copy['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')

# Drop some useless varibales from copied dataframe

columns_to_drop = [
    'CustomerID',
    'Count',
    'Lat Long',
    'Zip Code',
    'City',
    'Churn Label',
    'Country',
    'State'
]

df_copy = df_copy.drop(columns=columns_to_drop)
df_copy.head(1)

"""## 6.1 Clustering Based on Region <a id=6.1></a>"""

from sklearn.cluster import KMeans

# Example: Creating clusters based on latitude and longitude for region grouping
def add_region_clusters(df, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Region Cluster'] = kmeans.fit_predict(df[['Latitude', 'Longitude']])
    return df

df_copy = add_region_clusters(df_copy)
df_copy.head()

scatter_plot(df_copy, "Latitude", "Longitude", hue_column='Region Cluster')

"""## 6.2 Average Monthly Spend <a id=6.2></a>"""

df_copy['Total Charges'] = pd.to_numeric(df_copy['Total Charges'], errors='coerce')

def calculate_avg_monthly_spend(df):
    df['Avg Monthly Spend'] = df['Total Charges'] / df['Tenure Months']
    df['Avg Monthly Spend'].fillna(0, inplace=True)  # Handle division by zero
    return df

df_copy = calculate_avg_monthly_spend(df_copy)
df_copy.head()

"""The values of `Avg Monthly Spend` are almost same as `Monthly Charges`, but there are few cases that these two values might differ:

1. Billing Changes Over Time: If a customer’s monthly charges changed (e.g., due to plan upgrades, downgrades, or discounts), their average monthly spend would be different from the current monthly charges.

2. Promotional Offers or Discounts: If customers received discounts or promotional pricing in their initial months, their average spend might be lower than their current monthly charges.

3. Unpaid Months or Gaps: If a customer missed payments or had gaps in service, their average monthly spend may not align perfectly with their current monthly charges.
"""

# List of columns to plot
cols = ['Avg Monthly Spend', 'Monthly Charges']

# Set up the figure for side-by-side subplots
plt.figure(figsize=(12, 5))

# Loop through columns and create a subplot for each
for i, col in enumerate(cols):
    plt.subplot(1, 2, i + 1)  # Adjusting i + 1 to set the correct position
    sns.histplot(df_copy[col], bins=30, kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()  # Adjust spacing between plots
plt.show()

"""## 6.3 Serivce Bundling <a id=6.3></a>"""

service_columns = ['Phone Service', 'Multiple Lines', 'Internet Service',
                   'Online Security', 'Online Backup', 'Device Protection',
                   'Tech Support', 'Streaming TV', 'Streaming Movies']

def calculate_total_services(df):
    df['Total Services Subscribed'] = df[service_columns].apply(lambda row: sum(row == 'Yes'), axis=1)
    return df

df_copy = calculate_total_services(df_copy)

plot_categorical_bar(df_copy, 'Total Services Subscribed')

"""## 6.3 Payment Method Recurrence <a id=6.3></a>

If a customer’s payment method is either "Bank transfer (automatic)" or "Credit card (automatic)", they are considered to be using a recurring payment method. **This column provides information about whether each customer's payment method is automatic (recurring) or not.**
"""

def add_payment_recurrence(df):
    recurring_methods = ['Bank transfer (automatic)', 'Credit card (automatic)']
    df['Recurring Payment'] = df['Payment Method'].apply(lambda x: 1 if x in recurring_methods else 0)
    return df

df_copy = add_payment_recurrence(df_copy)

plot_categorical_bar(df_copy, 'Recurring Payment')

"""## 6.4 Auto-Pay Indicator <a id=6.4></a>

The code assigns a value of 1 to the Auto-Pay column if both conditions are met:
1. The Paperless Billing column is set to "Yes".
2. The Recurring Payment column (created in a previous step) is set to 1 (indicating an automatic payment method, such as "Bank transfer (automatic)" or "Credit card (automatic)").
If either condition is not met, the Auto-Pay column is assigned a value of 0.


>The Auto-Pay column acts as a binary indicator, where 1 signifies that the customer is using an automatic, paperless billing method, and 0 indicates they are not.
"""

def add_auto_pay_indicator(df):
    df['Auto-Pay'] = ((df['Paperless Billing'] == 'Yes') & (df['Recurring Payment'] == 1)).astype(int)
    return df

df_copy = add_auto_pay_indicator(df_copy)

plot_categorical_bar(df_copy, 'Auto-Pay')

"""## 6.4 Potential Upsell Flag <a id=6.4></a>

* Churn Score:	A value from 0-100 that is calculated using a predictive tool (e.g., IBM SPSS Modeler). The model incorporates multiple factors known to cause churn. The higher the score, the more likely the customer will churn.
* CLTV:	Customer Lifetime Value. A predicted CLTV is calculated using corporate formulas and existing data. Higher values indicate more valuable customers who should be monitored for churn.  

**The Upsell Potential column provides a binary flag (1 or 0) indicating customers who have both high churn risk and above-average lifetime value. This information can help identify high-value customers who may benefit from targeted offers or engagement strategies to increase revenue and retention.**
"""

def add_upsell_flag(df):
    df['Upsell Potential'] = ((df['Churn Score'] > 50) & (df['CLTV'] > df['CLTV'].mean())).astype(int)
    return df

df_copy = add_upsell_flag(df_copy)

plot_categorical_bar(df_copy, 'Upsell Potential')

"""## 6.5 Churn Probability Group <a id=6.5></a>"""

def add_churn_score_group(df):
    bins = [0, 33, 66, 100]
    labels = ['Low', 'Medium', 'High']
    df['Churn Score Group'] = pd.cut(df['Churn Score'], bins=bins, labels=labels)
    return df

df_copy = add_churn_score_group(df_copy)

plot_categorical_bar(df_copy, 'Churn Score Group')

"""## 6.6 Grouping Churn Reason <a id=6.6></a>"""

# Define a function to categorize Churn Reasons
def categorize_churn_reason(reason):
    if pd.isna(reason):
        return 'Missing'

    reason = reason.lower()

    # Competitor related reasons
    if 'competitor' in reason:
        return 'Competitor'

    # Price and service related reasons
    elif 'price' in reason or 'charges' in reason or 'extra data' in reason or 'long distance' in reason or 'affordable' in reason:
        return 'Price & Service'

    # Support and service quality issues
    elif 'attitude' in reason or 'expertise' in reason or 'support' in reason or 'self-service' in reason:
        return 'Support & Service Quality'

    # Product and network related reasons
    elif 'network' in reason or 'product' in reason or 'limited range' in reason:
        return 'Product & Network'

    # Life changes
    elif 'moved' in reason or 'deceased' in reason:
        return 'Life Changes'

    # Unknown reasons
    elif "don't know" in reason:
        return 'Unknown'

    # Catch-all for any other reasons
    else:
        return 'Other'

# Apply the function to the 'Churn Reason' column and create a new column 'Churn Reason Summary'
df_copy['Churn Reason Group'] = df_copy['Churn Reason'].apply(categorize_churn_reason)

df_copy['Churn Reason Group'].value_counts()

"""# 7. WorldCloud of Churn Reason <a id=7></a>"""

from wordcloud import WordCloud

# We need to filter out NaN values in 'Churn Reason' for the general word cloud
churn_reason_all = df_copy['Churn Reason'].dropna().str.cat(sep=' ')

# Generate the general word cloud
wordcloud_all = WordCloud(width=800, height=400, background_color='black').generate(churn_reason_all)

# Plotting the general word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_all, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for All Churn Reasons')
plt.show()

# Word Clouds Grouped by Churn Score Group
churn_score_groups = df_copy['Churn Score Group'].unique()

# Loop through each group in 'Churn Score Group' and generate a word cloud
for group in churn_score_groups:
    # Filter data for the current churn score group
    churn_reason_group = df_copy[df_copy['Churn Score Group'] == group]['Churn Reason'].dropna().str.cat(sep=' ')

    # Only generate a word cloud if there are valid churn reasons (non-empty)
    if churn_reason_group:
        wordcloud_group = WordCloud(width=800, height=400, background_color='black').generate(churn_reason_group)

        # Plotting the word cloud for the group
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_group, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for Churn Reason - {group}')
        plt.show()
    else:
        print(f'No churn reasons for group: {group}')

"""# 8. Modelling <a id=8></a>"""

from sklearn.model_selection import (learning_curve, RandomizedSearchCV,
                                     GridSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             roc_curve, roc_auc_score)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

"""## 8.1 Encoding Categorical Variable <a id=8.1></a>"""

features_to_ohe = ['Gender', 'Senior Citizen','Partner', 'Dependents', 'Phone Service',
   'Multiple Lines', 'Online Security',
   'Online Backup', 'Device Protection',
   'Tech Support', 'Streaming TV',
   'Streaming Movies', 'Paperless Billing']

ohe = OneHotEncoder(sparse_output=False, drop='first')
encoded_data = ohe.fit_transform(df_copy[features_to_ohe])
encoded_data_df = pd.DataFrame(encoded_data,
                               columns=ohe.get_feature_names_out(features_to_ohe))
df_encoded = df_copy.drop(columns=features_to_ohe)
df_encoded = pd.concat([df_copy, encoded_data_df], axis=1)

df_encoded.drop(columns=features_to_ohe, inplace=True)

# LabelEncoder
features_to_le = ['Contract', 'Churn Score Group',
                  'Internet Service', 'Payment Method', 'Churn Reason Group']

le = LabelEncoder()
for feature in features_to_le:
    df_encoded[feature] = le.fit_transform(df_encoded[feature])

df_encoded.drop(columns=features_to_le, inplace=True)

"""## 8.1 Split Dependent/Independent Variables <a id=8.1></a>"""

df_encoded.dropna(subset=['Total Charges'], axis=0, inplace=True)

X = df_encoded.drop(columns=['Churn Value', 'Churn Reason'], axis=1)
y = df_encoded['Churn Value']

print(f"X Shape: {X.shape}")
print(f"y Shape: {y.shape}")

"""## 8.2 Train/Test/Validation Sets <a id=8.2></a>"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Split Validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Check the shape of the splits
print(f"X_train Shape: {X_train.shape}")
print(f"X_val Shape: {X_val.shape}")
print(f"X_test Shape: {X_test.shape}")
print(f"y_train Shape: {y_train.shape}")
print(f"y_val Shape: {y_val.shape}")
print(f"y_test Shape: {y_test.shape}")

"""## 8.3 XGBClassifier <a id=8.3></a>"""

from sklearn.metrics import accuracy_score

# Initialize the model
xgb_model = XGBClassifier(random_state=42)

# Train the model on the training data
xgb_model.fit(X_train, y_train)

# Predict on the validation data
y_val_pred = xgb_model.predict(X_val)

# Evaluate accuracy on the validation set
cnf_matrix = confusion_matrix(y_val, y_val_pred)
sns.heatmap(cnf_matrix, annot=True, fmt="d")

val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Initial Model Validation Accuracy: {val_accuracy}")
print(f"\n\nClassification Report:\n\n {classification_report(y_val, y_val_pred)}")

"""___Hyperparameter Tuning___"""

# Define the hyperparameter grid
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 6, 10],
    'n_estimators': [100, 200, 300],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

random_search = RandomizedSearchCV(estimator=XGBClassifier(random_state=42),
                                   param_distributions=param_grid,
                                   scoring='accuracy',
                                   cv=5,
                                   verbose=1,
                                   n_jobs=-1)

random_search.fit(X_val, y_val)

"""___Train New Model___"""

# Get the best parameters from grid search
best_params = random_search.best_params_
best_estimator = random_search.best_estimator_
print(f"Best Hyperparameters: {best_params}")

xgb_tuned = XGBClassifier(**best_params, random_state=42)
xgb_tuned.fit(X_train, y_train)
y_pred_tuned = xgb_tuned.predict(X_val)

# Evaluate accuracy on the validation set
cnf_matrix = confusion_matrix(y_val, y_pred_tuned)
sns.heatmap(cnf_matrix, annot=True, fmt="d")

val_accuracy = accuracy_score(y_val, y_pred_tuned)
print(f"Initial Model Validation Accuracy: {val_accuracy}")
print(f"\n\nClassification Report:\n\n {classification_report(y_val, y_pred_tuned)}")

"""__Cross Validation__"""

cv_scores = cross_val_score(xgb_tuned, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross Validation Scores: {cv_scores}")
print(f"Cross Validation Mean: {cv_scores.mean()}")

"""**Learning Curve**"""

# Get learning curve data
train_sizes, train_scores, val_scores = learning_curve(
    xgb_tuned, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy', random_state=42
)

# Calculate mean and std for training and validation scores
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
val_scores_std = np.std(val_scores, axis=1)

# Plot the learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color="blue", label="Training score")
plt.plot(train_sizes, val_scores_mean, 'o-', color="green", label="Cross-validation score")
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, color="blue", alpha=0.1)
plt.fill_between(train_sizes, val_scores_mean - val_scores_std, val_scores_mean + val_scores_std, color="green", alpha=0.1)
plt.xlabel('Training Size')
plt.ylabel('Score')
plt.title('Learning Curve')
plt.legend(loc='best')
plt.show()

"""## 8.4 RandomForestClassifier <a id=8.4></a>"""

# RandomForestClassifier Instance
rf_model = RandomForestClassifier(random_state=42)
# Train RandomForest
rf_model.fit(X_train, y_train)
# Prediction on validation set
y_pred_val_rf = rf_model.predict(X_val)

# Evaluate accuracy of rf model
# confusion matrix & classification report
cnf_matrix = confusion_matrix(y_val, y_pred_val_rf)
sns.heatmap(cnf_matrix, annot=True, fmt="d")

print(f"\n\nClassification Report:\n\n {classification_report(y_val, y_pred_val_rf)}")

# accuracy score
val_accuracy = accuracy_score(y_val, y_pred_val_rf)
print(f"Initial Model Validation Accuracy: {val_accuracy}")

"""__Hyperparameter Tunning__"""

rf_tuned = RandomForestClassifier(random_state=42)


params = {
    'n_estimators': [100, 200, 300, 350],
    'max_depth': [2, 4, 8, 10,],
    'min_samples_split': [2, 4, 6, 8],
    'min_samples_leaf': [1, 3, 5, 7]
}

random_search = RandomizedSearchCV(estimator=rf_tuned,
                                   param_distributions=params,
                                   scoring='accuracy',
                                   cv=5,
                                   verbose=1,
                                   n_jobs=-1)

random_search.fit(X_val, y_val)

"""__Train Tuned Model__"""

best_params = random_search.best_params_
print(best_params)

# Randomforest model with tuned hyperparameters
rf_tuned = RandomForestClassifier(**best_params, random_state=42)
# Train tuned randomforstes model
rf_tuned.fit(X_train, y_train)
# Prediction on validation
y_pred_tuned_rf = rf_tuned.predict(X_val)

# Evaluate accuracy on the validation set
cnf_matrix = confusion_matrix(y_val, y_pred_tuned_rf)
sns.heatmap(cnf_matrix, annot=True, fmt="d")

val_accuracy = accuracy_score(y_val, y_pred_tuned_rf)
print(f"Initial Model Validation Accuracy: {val_accuracy}")
print(f"\n\nClassification Report:\n\n {classification_report(y_val, y_pred_tuned_rf)}")

cv_scores = cross_val_score(rf_tuned, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross Validation Scores: {cv_scores}")
print(f"Cross Validation Mean: {cv_scores.mean()}")

"""__Feature Importance__"""

importances = rf_model.feature_importances_

# Convert to DataFrame for readability
feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plotting
plt.figure(figsize=(10, 7))
plt.barh(feature_importances['Feature'], feature_importances['Importance'])
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importances in Random Forest')
plt.gca().invert_yaxis()  # Most important at the top
plt.show()

"""**Learning Curve**"""

# Get learning curve data
train_sizes, train_scores, val_scores = learning_curve(
    rf_tuned, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy', random_state=42
)

# Calculate mean and std for training and validation scores
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
val_scores_std = np.std(val_scores, axis=1)

# Plot the learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color="blue", label="Training score")
plt.plot(train_sizes, val_scores_mean, 'o-', color="green", label="Cross-validation score")
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, color="blue", alpha=0.1)
plt.fill_between(train_sizes, val_scores_mean - val_scores_std, val_scores_mean + val_scores_std, color="green", alpha=0.1)
plt.xlabel('Training Size')
plt.ylabel('Score')
plt.title('Learning Curve')
plt.legend(loc='best')
plt.show()

"""## 8.5 Balancing Classes Using SMOTE <a id=8.5></a>"""

# Creating SMOTE instance
smote = SMOTE(random_state=42)
# Perfomr Sample Balancing
X_res, y_res = smote.fit_resample(X, y)

# Split Train/Test/Validation Sets
X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
X_train_smote, X_val_smote, y_train_smote, y_val_smote = train_test_split(X_train_smote, y_train_smote, test_size=0.2, random_state=42)

"""**Train New Model With Balanced Data**"""

rf_tuned_smote = RandomForestClassifier(random_state=42)


params = {
    'n_estimators': [100, 200, 300, 350],
    'max_depth': [2, 4, 8, 10,],
    'min_samples_split': [2, 4, 6, 8],
    'min_samples_leaf': [1, 3, 5, 7]
}

random_search = RandomizedSearchCV(estimator=rf_tuned_smote,
                                   param_distributions=params,
                                   scoring='accuracy',
                                   cv=5,
                                   verbose=1,
                                   n_jobs=-1)

random_search.fit(X_val_smote, y_val_smote)

# Randomforest model with tuned hyperparameters
rf_tuned_smote = RandomForestClassifier(**best_params, random_state=42)
# Train tuned randomforstes model
rf_tuned_smote.fit(X_train_smote, y_train_smote)
# Prediction on validation
y_pred_tuned_rf = rf_tuned_smote.predict(X_val_smote)

# Evaluate accuracy on the validation set
cnf_matrix = confusion_matrix(y_val_smote, y_pred_tuned_rf)
sns.heatmap(cnf_matrix, annot=True, fmt="d")

val_accuracy = accuracy_score(y_val_smote, y_pred_tuned_rf)
print(f"Initial Model Validation Accuracy: {val_accuracy}")
print(f"\n\nClassification Report:\n\n {classification_report(y_val_smote, y_pred_tuned_rf)}")

# Get learning curve data
train_sizes, train_scores, val_scores = learning_curve(
    rf_tuned_smote, X_res, y_res, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy', random_state=42
)

# Calculate mean and std for training and validation scores
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
val_scores_std = np.std(val_scores, axis=1)

# Plot the learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color="blue", label="Training score")
plt.plot(train_sizes, val_scores_mean, 'o-', color="green", label="Cross-validation score")
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, color="blue", alpha=0.1)
plt.fill_between(train_sizes, val_scores_mean - val_scores_std, val_scores_mean + val_scores_std, color="green", alpha=0.1)
plt.xlabel('Training Size')
plt.ylabel('Score')
plt.title('Learning Curve')
plt.legend(loc='best')
plt.show()

"""# 9. Add Churn Probability To DataFrame <a id=9></a>"""

# Get the churn probability for each sample in the entire dataset
churn_probabilities = rf_tuned_smote.predict_proba(X)[:, 1]  # Column 1 represents the probability of the positive class (churn)

# Add churn probability as a new column to the original dataset
df_encoded['Churn_Probability'] = churn_probabilities

# Display the updated dataset with the new Churn_Probability column
df_encoded.head()

"""# 10. Save Models And Data <a id=10></a>"""

# Save Data
df_encoded.to_csv('predicted_churn.csv', index='ignore')

# Save Models
import joblib
# Save RandomForest Model With Data Balance
joblib.dump(rf_tuned_smote, 'rf_model_smote.pkl')

# Save RandomForest Model Without Data Balancing
joblib.dump(rf_tuned, 'rf_model.pkl')

