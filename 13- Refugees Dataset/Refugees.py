#!/usr/bin/env python
# coding: utf-8

# # Refugees Dataset (2010 - 2022)
# [Link To Dataset](https://www.kaggle.com/datasets/sujaykapadnis/refugees)  

# # Dataset First Evaluation
# 
# **year** -> The year.  
# **coo_name** -> Country of origin name.  
# **coo**	-> Country of origin UNHCR code.  
# **coo_iso**	-> Country of origin ISO code.  
# **coa_name** -> Country of asylum name.  
# **coa**	-> Country of asylum UNHCR code.  
# **coa_iso**	-> Country of asylum ISO code.  
# **refugees** -> The number of refugees.  
# **asylum_seekers** -> The number of asylum-seekers.  
# **returned_refugees** -> The number of returned refugees.  
# **idps** -> The number of internally displaced persons.  
# **returned_idps** -> The number of returned internally displaced persons.  
# **stateless** -> The number of stateless persons.  
# **ooc**	->	The number of others of concern to UNHCR.  
# **oip**	->	The number of other people in need of international protection.  
# **hst**	->	The number of host community members.  

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[69]:


# read dataset to df
df = pd.read_csv('population.csv')

# check first rows
df.head()


# In[3]:


# check df shape
df.shape


# In[4]:


# check df dtypes
df.dtypes


# In[5]:


# check general information of df
df.info()


# **6 Features are `object`**  
# **8 Features are `int`**  
# **2 Features are `float`**

# In[6]:


# check NaNs
df.isna().sum()


# ___Total number of rows are 64809. There are lot of NaNs in `oip` and `hst` columns___

# In[19]:


# check number of duplicate rows
df.duplicated().sum()


# In[7]:


# check statistical values of dataframe
df.describe()


# # Handling Missing Values

# In[70]:


# check missing values of all features where there are at least 1 missing values
df[df.isna().any(axis = 1)]


# In[ ]:


# there are lots of NaNs within `oip` and `hst` features, due to this issue , we can simply
# drop these columns becuase they dont provide much insightful information caused by NaNs


# In[71]:


df.drop(columns=['oip', 'hst'], inplace = True) # removing oip and hst column


# In[72]:


df.info()


# In[ ]:


# also we can drop coo_iso & coa_iso in order to decrease dataframe size


# In[73]:


df.drop(columns=['coo_iso', 'coa_iso'], inplace=True)


# In[74]:


df.head()


# # Exploratory Data Analysis

# In[78]:


df.hist(figsize=(10, 10))


# In[79]:


df.sort_values(by='refugees', ascending=False).head(20)


# ### Total Refugees by Country of Origin (COO)

# In[106]:


total_refugees_by_coo = df.groupby('coo_name')['refugees'].sum().sort_values(ascending=False).to_frame().head(10)


# In[108]:


total_refugees_by_coo


# ### Average Asylum Seekers by Country of Asylum (COA)

# In[109]:


avg_asylum_seeker_coa = df.groupby('coa_name')['asylum_seekers'].mean().sort_values(ascending=False).to_frame().head(10)
avg_asylum_seeker_coa


# ### Total IDPs (Internally Displaced Persons) by Year:

# In[110]:


total_idps_by_year = df.groupby('year')['idps'].sum().to_frame().head(10)
total_idps_by_year


# ### Total Refugees and Asylum Seekers by Year and COO

# In[118]:


total_by_year_coo = df.groupby(['year', 'coo_name', 'coa'])[['refugees', 'asylum_seekers']].sum().reset_index()
total_by_year_coo


# ### Total Refugees by Country of Origin and Asylum (COO and COA)

# In[125]:


total_refugees_by_coo_coa = df.groupby(['coo_name', 'coa_name'])['refugees'].sum().sort_values(ascending=False).to_frame().head(10)
total_refugees_by_coo_coa


# ### Top Countries with the Highest Number of Asylum Seekers

# In[128]:


top_asylum_countries = df.groupby('coa_name')['asylum_seekers'].sum().sort_values(ascending=False).to_frame().head(20)
top_asylum_countries


# ### Distribution of Refugees Over the Years

# In[131]:


plt.figure(figsize=(12, 6))
sns.lineplot(x='year', y='refugees', data=df, estimator=sum, errorbar=None)
plt.title('Distribution of Refugees Over the Years')
plt.xlabel('Year')
plt.ylabel('Total Refugees')
plt.show()


# ### Correlation Heatmap

# In[133]:


corr_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# ### Refugees vs. IDPs Scatter Plot

# In[134]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x='refugees', y='idps', data=df)
plt.title('Refugees vs. IDPs')
plt.xlabel('Refugees')
plt.ylabel('IDPs')
plt.show()


# ### Merge DataFrames to Compare Total Refugees and Asylum Seekers

# In[136]:


coo_grouped = df.groupby('coo_name')['refugees'].sum().reset_index()
coa_grouped = df.groupby('coa_name')['asylum_seekers'].sum().reset_index()


# In[138]:


merged_df = pd.merge(coo_grouped, coa_grouped , left_on='coo_name', right_on='coa_name', suffixes=('_coo', '_coa'))
merged_df.head()


# ### Total OOC (Others of Concern) by Country

# In[142]:


total_ooc_by_country = df.groupby('coo_name')['ooc'].sum().sort_values(ascending=False).to_frame().head(10)


# In[143]:


total_ooc_by_country


# ### Concatenation of Top Refugee and Asylum Countries:

# In[144]:


top_refugee_countries = df.groupby('coo_name')['refugees'].sum().sort_values(ascending=False).head(5)
top_asylum_countries = df.groupby('coa_name')['asylum_seekers'].sum().sort_values(ascending=False).head(5)
concatenated_df = pd.concat([top_refugee_countries, top_asylum_countries], axis=1)


# In[145]:


concatenated_df


# ### Refugee Trends by Country Over the Years:

# In[147]:


plt.figure(figsize=(14, 8))
sns.lineplot(x='year', y='refugees', hue='coo_name', data=df, estimator=sum, errorbar=None)
plt.title('Refugee Trends by Country Over the Years')
plt.xlabel('Year')
plt.ylabel('Total Refugees')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# ### Comparison of Total Refugees and Asylum Seekers

# In[148]:


plt.figure(figsize=(12, 6))
sns.barplot(x='refugees', y='coo_name', data=df.groupby('coo_name')['refugees'].sum().reset_index().sort_values(by='refugees', ascending=False).head(10))
plt.title('Top 10 Countries with the Highest Number of Refugees')
plt.xlabel('Total Refugees')
plt.ylabel('Country')
plt.show()


# ### Distribution of IDPs (Internally Displaced Persons) Over the Years

# In[152]:


plt.figure(figsize=(12, 6))
sns.lineplot(x='year', y='idps', data=df, estimator=sum, errorbar=None)
plt.title('Distribution of IDPs Over the Years')
plt.xlabel('Year')
plt.ylabel('Total IDPs')
plt.show()


# ### Refugee to Asylum Seeker Ratio

# In[154]:


df['refugee_asylum_ratio'] = df['refugees'] / df['asylum_seekers']
average_ratio = df['refugee_asylum_ratio'].mean()


# In[156]:


df


# ### Top Countries with the Highest Number of Returned Refugees

# In[160]:


top_returned_countries = df.groupby('coo_name')['returned_refugees'].sum().sort_values(ascending=False).to_frame().head(10)
top_returned_countries


# In[ ]:




