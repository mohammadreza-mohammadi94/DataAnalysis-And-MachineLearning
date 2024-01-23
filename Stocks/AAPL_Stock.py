#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import scipy.stats as stats


# In[4]:


df = pd.read_csv(r"C:\Users\mrmhm\Desktop\Stocks-Dataset\AAPL_2006-01-01_to_2018-01-01.csv")


# # Dataset Analysis

# In[5]:


df.head()


# In[6]:


df.info()


# In[9]:


df.shape


# In[10]:


df.size


# In[12]:


df.describe()


# # Data Distribution

# In[21]:


def dist(df, col):
    plt.figure(figsize=(12, 6))
    plt.hist(df[col], bins=365, color='blue', edgecolor='black')
    plt.title(f"Distribution of {col} Prices", fontsize = 15)
    plt.xlabel("Closing Price", fontsize = 12)
    plt.ylabel("Frequency")
    plt.show()


# In[23]:


for col in ['Close', 'Open', 'High', 'Low']:
    dist(df, col)


# # Prices Over Time

# In[24]:


# Changing dtype of Date column to datetime dtype.
df['Date'] = pd.to_datetime(df['Date'])


# In[27]:


def price_over_time(df, col):
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df[col])
    plt.title(f"{col} Changes Over Time", fontsize=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel(f"{col} Price", fontsize=12)
    plt.tight_layout()
    plt.show()


# In[28]:


for col in ['Close', 'Open', 'High', 'Low']:
    price_over_time(df, col)


# # Correlation Matrix

# In[34]:


plt.figure(figsize=(10, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, cmap="YlGnBu", annot=True)
plt.show()


# In[37]:


corr


# # Boxplot - To Identify Outliers

# In[44]:


def boxplot(df, col):
    plt.figure(figsize=(10, 5))
    sns.boxplot(df[col], showmeans=True)
    plt.title(f"{col} Boxplot", fontsize=15)
    plt.xlabel(f"{col}")
    plt.show()


# In[45]:


for col in ['Close', 'Open', 'High', 'Low']:
    boxplot(df, col)


# # Moving Average 50

# In[46]:


# Calculate and add MA_50
df['MA_50'] = df['Close'].rolling(window=50).mean()


# In[52]:


def moving_average(df, col):
    plt.figure(figsize=(16, 8))
    plt.plot(df['Date'], df[col], label=f"{col} Price")
    plt.plot(df["Date"], df['MA_50'], label='50-Day MA')
    plt.legend(fontsize=10)
    plt.tight_layout
    plt.show()


# In[53]:


for col in ['Close', 'Open', 'High', 'Low']:
    moving_average(df, col)


# # Daily Returns

# In[54]:


# Calulate and add Daily_Return
df['Daily_Ret'] = df['Close'].pct_change()


# In[56]:


plt.figure(figsize=(16, 8))
plt.plot(df['Date'], df['Daily_Ret'])
plt.title("Daily Return", fontsize=15)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Percentage Change")
plt.tight_layout()
plt.show()


# # Volume Trends

# In[57]:


plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Volume'])
plt.title("Vol Over Time", fontsize=15)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Volume", fontsize=12)
plt.show()


# # Season Decomposition

# In[58]:


from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df['Close'], model='multiplicative', period=30)
result.plot()
plt.show()


# # Scatter Plot

# In[59]:


def scatter(df, col1, col2):
    plt.figure(figsize=(12, 6))
    plt.scatter(df[col1], df[col2])
    plt.title(f"{col1} Vs. {col2}", fontsize=15)
    plt.xlabel(f"{col1}", fontsize=12)
    plt.ylabel(f"{col2}", fontsize=12)
    plt.show()


# In[60]:


scatter(df, 'Volume', 'Close')


# In[65]:


scatter(df, 'High', 'Close')


# In[66]:


scatter(df, 'Low', 'Close')


# In[67]:


scatter(df, 'High', 'Low')


# # Exploring Relationship between Close and Trading Volume

# In[96]:


plt.figure(figsize=(12, 6))
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Date')
ax1.set_ylabel('Closing Price', color=color)
ax1.plot(df['Date'], df['Close'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Volume', color=color)
ax2.plot(df['Date'], df['Volume'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Closing Price and Volume Trends')
plt.show()


# # Seasonal Changes

# In[99]:


monthly_mean = df.groupby(df['Date'].dt.month)['Close'].mean()
plt.figure(figsize=(12, 6))
plt.plot(monthly_mean.index, monthly_mean.values)
plt.title('Seasonal Changes in Closing Prices')
plt.xlabel('Month')
plt.ylabel('Mean Closing Price')
plt.show()


# # Inferential Analysis

# In[73]:


from scipy.stats import t, ttest_1samp


# # Estimate Mean Closing Price

# In[87]:


np.random.seed(42)

# Number of Samples and Sample size of 'Close' feature
num_sample = 10
sample_size = 100

# Recording data
sample_means = []
confidence_intervals = []
confidence_levels = []
pop_mean_estimators = []

# Generate and Analyze Samples
for _ in range(num_sample):
    # Generate a random sample
    sample = np.random.choice(df['Close'], sample_size, replace=True)

    # Calculate means
    sample_mean = np.mean(sample)
    sample_means.append(sample_mean)

    # Calculate Standard Error
    standard_error = stats.sem(sample)
    # Caculate Margin of Error
    cl = 0.95
    margin_of_error = stats.norm.ppf((1 + cl) / 2) * standard_error
    # Confidence Interval
    confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)
    confidence_intervals.append(confidence_interval)

# Print results
for i in range(num_sample):
    print(f"Sample {i + 1}:")
    print(f"  Sample Mean: {sample_means[i]}")
    print(f"  Margin of Error: {margin_of_error}")
    print(f"  Confidence Interval: {confidence_intervals[i]}\n")


# In[84]:


estimate_pop_mean = np.mean(sample_means)
estimate_pop_std = np.std(sample_means)
population_mean_ci = stats.norm.interval(cl, loc=estimate_pop_mean, scale=estimate_pop_std)


# In[86]:


print(f"Estimate Mean of Population: {estimate_pop_mean}")
print(f"Confidence Interval: {population_mean_ci}")
print(f"True Population Mean: {np.mean(df['Close'])}")


# 1. For each sample, we generate a sample mean, calculates the margin of error, and constructs a 95% confidence interval.
# 2. Then we calculate an estimate for the population mean based on the sample means.
# 3. The estimated population mean is 65.72923 with a 95% confidence interval of (60.79, 70.67).
# 4. The true population mean of the 'Close' feature in the original dataset is 64.66.
# 5. The estimate for the population mean falls within the confidence interval, indicating consistency between the estimate and the sample data.
# 6. The true population mean is also within the calculated confidence interval, supporting the accuracy of the estimation.val, with the estimate aligning well with the true population mean from the original dataset.e estimation.

# # Mean Hypothesis Test (t-test)

# __H0: The estimated population mean is equal to the true population mean.__   
# 
# __H1: The estimated population mean is not equal to the true population mean.__

# In[91]:


# One-Sample ttest
t_stats, p_value = stats.ttest_1samp(sample_means, np.mean(df['Close']))
# Alpha
alpha = 0.05

# Print results
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

# Compare p-value with significance level
if p_value < alpha:
    print("Reject the null hypothesis. There is evidence that the estimated population mean is different from the true population mean.")
else:
    print("Fail to reject the null hypothesis. There is not enough evidence to suggest a difference between the estimated and true population means.")


# In[ ]:




