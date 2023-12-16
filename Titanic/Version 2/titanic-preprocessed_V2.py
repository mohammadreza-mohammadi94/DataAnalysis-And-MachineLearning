#!/usr/bin/env python
# coding: utf-8

# # Loading Libraries

# In[204]:


# Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler


# # Load and Check Dataset 

# In[174]:


# Laoding Dataset 
df = pd.read_csv('https://raw.githubusercontent.com/PeterLOVANAS/Titanic-machine-learning-project/main/datasets/Titanic_dataset_com.csv')


# In[175]:


df.info()


# In[176]:


df.columns


# In[177]:


df.describe()


# # Solving Missing Values

# **Check total NaNs Values on each column**

# In[178]:


df.isnull().sum()


# ___Totally we have 1310 rows and there is 1 row which is completely NaN with all features___

# ### Drop rows which are all NaNs

# In[179]:


df.dropna(how = 'all', inplace = True)


# ### Check Percentage of missing values

# In[180]:


def NansPercentage(col, dataframe):
    perc = dataframe[col].isnull().sum() / dataframe.shape[0]
    print(f"Missing Values Percentage of {col} Feature is {round(perc * 100,3)} %")


# In[181]:


columns = ['PassengerId', 'pclass', 'survived', 'name', 'sex', 'age', 'sibsp',
       'parch', 'ticket', 'fare', 'cabin', 'embarked', 'boat', 'body',
       'home.dest']

for col in columns:
    NansPercentage(col, df)


# **It is clear that there are lots of NaNs on `cabin`, `boat`, `body`, `home.dest` features. So we can ignore and drop these features**

# In[182]:


# function to show number of missing values of each feature
def getMissingValues(cols, dataframe):
        print(f"Number of Missing Values of {col} is {df[col].isnull().sum()}")


# In[183]:


columns = ['PassengerId', 'pclass', 'survived', 'name', 'sex', 'age', 'sibsp',
       'parch', 'ticket', 'fare', 'cabin', 'embarked', 'boat', 'body',
       'home.dest']
for col in columns:
    getMissingValues(col, df)


# ### Drop cabin, boat, body, home.dest Columns

# In[184]:


df.drop(columns = ['cabin', 'boat', 'body', 'home.dest'], axis = 1, inplace = True)


# In[185]:


# checking missing values percentage after droping columns with most NaNs
columns = ['PassengerId', 'pclass', 'survived', 'name', 'sex', 'age', 'sibsp',
       'parch', 'ticket', 'fare',  'embarked']

for col in columns:
    NansPercentage(col, df)


# ### Filling Missing Values

# ___For `fare`, `embarked` and `age` we will use mean to fill missing values___

# In[186]:


df["fare"].fillna(value = df["fare"].mean(), inplace=True)
df["embarked"].fillna(df["embarked"].value_counts().idxmax(), inplace=True)
df["age"].fillna(value = df["age"].mean(), inplace=True)


# In[187]:


columns = ['PassengerId', 'pclass', 'survived', 'name', 'sex', 'age', 'sibsp',
       'parch', 'ticket', 'fare',  'embarked']
for col in columns:
    getMissingValues(col, df)


# ***There are no Missing Values anymore***

# # Data Visualization

# ### Survival Distribution

# In[188]:


sns.set(style="whitegrid")
sns.countplot(x = 'survived', data = df, palette = 'Set1')    # creating coundplot
plt.title("Survival Distribution (0: No, 1: Yes)")            # set title for countplot
plt.ylabel("Survived")                                        # title of y-axis
plt.xlabel("Count")                                           # title of x-axis
plt.show()


# ### pclass Distribution

# In[189]:


sns.set(style = 'whitegrid')
sns.countplot(x = 'pclass', data = df , palette = 'Set1')
plt.title("Passenger Class Distribution")
plt.ylabel('Count')
plt.xlabel("Class")
plt.show()


# ### Crrelation Matrix

# In[190]:


plt.figure(figsize = (14, 6))
ax = sns.heatmap(corr_df, annot = True, fmt=".1g", vmin = -1, vmax = 1, center = 0, cmap='inferno', linecolor='Black')
plt.title("Correlation Heatmap")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)


# ### `parch` Vs `survived` Correlation

# In[191]:


plt.figure(figsize = (10, 6))
sns.regplot(data = df, x = 'parch', y = 'survived', color = 'b')
plt.title("parch Vs survived Correlation")
plt.show()


# ### `sibsp` Vs `pclass` Correlation

# In[192]:


plt.figure(figsize = (10, 6))
sns.regplot(data = df, x = 'sibsp', y = 'pclass', color = 'b')
plt.title("parch Vs survived Correlation")
plt.show()


# ### Boxplot for Numeric Features

# In[193]:


num_df = df.select_dtypes(include = 'number')
plt.style.use('seaborn')

names = list(num_df.columns)

plot_per_row = 2

f, axes = plt.subplots(round(len(names)/plot_per_row), plot_per_row, figsize = (15, 25))

y = 0;

for name in names:
    i, j = divmod(y, plot_per_row)
    sns.boxplot(x=num_df[name], ax=axes[i, j], palette = 'Set3')
    y = y + 1

plt.tight_layout()
plt.show()


# # Data Reduction & Data Transformation

# ## LabelEncoding

# ***Perfom LabelEncoding on categorical features***

# In[194]:


def labelencoder(df, columns):
    le = LabelEncoder()
    for col in columns:
        df[col] = le.fit_transform(df[col])
    return df


# In[195]:


df = labelencoder(df, columns=['name', 'embarked', 'ticket'])


# In[196]:


df.head()


# ## One-Hot Encoding

# ***Perfom One-Hot Encoding on `sex` Feature***

# In[197]:


def one_hot(df, columns):
    return pd.get_dummies(df, columns=columns, dtype='int')


# In[198]:


df = one_hot(df, ['sex'])
df.head()


# # Model Analysis

# In[202]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 

X = df.drop("survived", axis=1)
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = LogisticRegression(max_iter=5000)
model.fit(X_train,y_train)
p_predict = model.predict(X_test)

print("The accuracy is", round(accuracy_score(p_predict, y_test) * 100,2))


# In[203]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

confusionMatrix = pd.crosstab(y_test, p_predict)
classificationReport = classification_report(y_test, p_predict)


fx = sns.heatmap(confusionMatrix, annot=True, cmap="Blues", fmt="d")
fx.set_title("Confusion matrix\n\n");
fx.set_xlabel("\nValues model predicted")
fx.set_ylabel("True Values ")
plt.show()
print(f"Classification Report\n{classificationReport}")


# In[ ]:




