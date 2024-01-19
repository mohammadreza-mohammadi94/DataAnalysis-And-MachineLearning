#!/usr/bin/env python
# coding: utf-8

# # Import Libs

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# # Dataset Evaluation

# In[2]:


# import dataset
df = pd.read_csv('flights.csv')
# check head of df
df.head()


# ### Dataset Description
# 
# | __Variable__ | __Description__ |
# |     :---      |       :---      |      
# | __id__ | A unique identifier assigned to each flight record in this dataset. |                
# | __year__ | The year in which the flight took place. The dataset includes flights from the year 2013 |                        
# | __month__ | The month of the year in which the flight occurred, represented by an integer ranging from 1 (January) to 12 (December) |
# | __day__ | The day of the month on which the flight took place, represented by an integer from 1 to 31 |
# | __dep_time__ | The actual departure time of the flight, represented in 24-hour format (hhmm) |                     
# | __sched_dep_time__ | The locally scheduled departure time of the flight, presented in a 24-hour format (hhmm) |
# | __dep_delay__ | The delay in flight departure, calculated as the difference (in minutes) between the actual and scheduled departure times. Positive values indicate a delay, while negative values indicate an early departure. |  
# | __arr_time__ | The actual arrival time of the flight, represented in 24-hour format (hhmm) |                      
# | __sched_arr_time__ | The locally scheduled arrival time of the flight, presented in a 24-hour format (hhmm) |
# | __arr_delay__ |  The delay in flight arrival, calculated as the difference (in minutes) between the actual and scheduled arrival times. Positive values indicate a delay, while negative values indicate an early arrival |
# | __carrier__ |  A two-letter code representing the airline carrier responsible for the flight |                      
# | __flight__ | The designated number of the flight |              
# | __tailnum__ | A unique identifier associated with the aircraft used for the flight |                      
# | __origin__ | A three-letter code signifying the airport from which the flight departed |
# | __dest__ | A three-letter code representing the airport at which the flight arrived |
# | __air_time__ | The duration of the flight, measured in minutes |                 
# | __distance__ | The total distance (in miles) between the origin and destination airports | 
# | __hour__ | The hour component of the scheduled departure time, expressed in local time | 
# | __minute__ | The minute component of the scheduled departure time, expressed in local time | 
# | __time_hour__ | The scheduled departure time of the flight, represented in local time and formatted as "yyyy-mm-dd hh:mm:ss" | 
# | __name__ | The full name of the airline carrier responsible for the flight | 

# In[3]:


df.shape


# In[4]:


# check datatype and features of dataset using df.info()
df.info()


# In[5]:


df.isna().sum()


# * ___This Dataset Has___:
#     * 336,776 Rows
#     * 21 features (Column)
#     * 5 feature with float dtype
#     * 10 feature with int dtype
#     * 6 feature with object dtype
# * ___There are missing values on___:
#     * `dep_time` : 8255
#     * `dep_delay` : 8255
#     * `arr_time` : 8713
#     * `arr_delay` : 9430
#     * `tailnum` : 2512
#     * `air_time` : 9430

# In[6]:


df.duplicated().sum()
# There are no duplicated rows in dataset


# In[7]:


# Checking statistical information and analysis of dataset using describe() method
df.describe().T


# - `dep_delay`, `arr_delay`: These are our target variables. They show the departure and arrival delays in minutes. The values range from negative (early departure or arrival) to positive (late departure or arrival).  
# - `year`: All records are belongs to 2013
# - `air_time` : It is flight duration in minutes. it starts from 20min up to 695min.
# - `distance`: This is the total distance between the origin and destination airports. It varies from 17 to 4983 miles.

# In[8]:


# Checking statistical summary for categorical features
df.describe(include='object')


# * `tailnum` is unique identifier therefore there are lot of unique values.  
# * `name`, `carrier` are airlines carrier code and names and there are 16 unique values therefore we have information about 16 airlines.
# * `time_hour` is date and time of scheduled departure time and it is in format of  "YYYY-MM-DD HH:MM:SS"

# # EDA

# In[9]:


# This function will show distribution of features using histplot of seaborn library.
def hist_plot(column, bins, title, xlabel, fontsize=8, rotation=0):
    plt.figure(figsize=(12, 6))
    plt.hist(column, bins=bins, color='green', edgecolor='black')
    plt.title(title, fontsize=15)
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.xticks(rotation=rotation, ha='right', fontsize=fontsize)
    plt.show()


# __`hist_plot`__ will be used to show distribution of numerical features

# In[10]:


# This function will use bar plot to show distribution of categorical features
def bar_plot(column, title, xlabel, fontsize=8):
    plt.figure(figsize=(12, 6))
    sns.barplot(x=column.value_counts().index, y=column.value_counts(), color='darkblue', edgecolor='black')
    plt.title(title, fontsize=15)
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel("Frequency", fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=fontsize)
    plt.show()


# __`bar_plot`__ will be used to show frequency of categorical features

# __________

# In[12]:


hist_plot(df['month'], bins=12, title='Month', xlabel='Month of Flight')
# We considere 12 for bins as we are plotting Months so we have 12 bins.


# * _Month Of Flight Distribution_:
#         Distribution of flights on each month seems to be Unifrom

# ______________

# In[13]:


# hist_plot of `day` feature
# We will pass value of 31 to bins parameter.
hist_plot(df['day'], bins=31, title="Day", xlabel='Day of Flight')


# * _Days of Flight Distribution_:
#         This feature seems to be Uniform distribution, with obvious decreasing on 31th day as some months have less than 31 days.

# ___________

# In[14]:


hist_plot(df['dep_time'].dropna(), bins=24, title="Departure Time (24-Hour Format)", xlabel="Time")


# * _Depature Time_:
#         This feature seems to be bimodal distributed. showing two peak period for departure time. First peak starts at 06:00 AM and second peak starts around 16:00 PM
# More about [Bimodal Distribution](https://de.wikipedia.org/wiki/Bimodale_Verteilung#:~:text=Eine%20bimodale%20Verteilung%20ist%20eine,besonders%20ausgepr%C3%A4gte%20lokale%20Maxima%20aufweist.)

# _____________

# In[15]:


hist_plot(df['sched_dep_time'], bins=24, title="Schedule Departure Time (24-hour format)", xlabel="Time")


# * _Schedule Departure Time_:
#         This plot is similar to the "Departure Time" histogram. It shows two peak periods for scheduled flight departures as well as actual departure time.

# ______________

# In[16]:


hist_plot(df['dep_delay'].dropna(), bins=30, title="Departure Delay", xlabel="Departure Delay (Minutes)")


# * _Departure Delay_ :
#         This histogram indicates that most flights departed close to their scehdule departure time, however there are many flight which departed with delays.
# 
# ***Note: The delay in flight departure, calculated as the difference (in minutes) between the actual and scheduled departure times***

# ______________

# In[17]:


hist_plot(df['arr_time'], bins=24, title="Actual Arrival Time", xlabel="Actual Time")


# * _Actual Arrival Time_ :
#         This histogram is bimodal distribution similar to Departure Time histogram with two peak periods for arrival time.

# ____________

# In[19]:


hist_plot(df['sched_arr_time'], bins=24, title="Schedule Arrival Time", xlabel="Actual Time")


# * _Schedule Arrival Time_:
#         This histogram is similiar to Actual Arrival Time.

# __________

# In[20]:


hist_plot(df['arr_delay'], bins=30, title="Arrival Delay (Minutes)", xlabel="Arrival Delay")


# * _Arrival Delay_ :
#          The histogram shows that most flight arrived close to their actual arrival time. These numbers are difference between Schedule Arrival Time and Actual Time.

# ______________________

# In[24]:


hist_plot(df['air_time'].dropna(), bins=30, title='Air Time', xlabel='Flight Duration (minutes)')


# * _Air Time_:
#     
# From this histogram plot we can understand that most flights duration are between from 50minutes up to 200 minutes. There are few flights having longer air time.

# _________________

# In[26]:


hist_plot(df['distance'], bins=30, title="Distance Plot", xlabel="Distance (Measured in Miles)")


# * _Distance_ :
# 
# This histogram shows distance between origin and destination measured in miles.We can clearly understand that most flight's distance are around 400 up to 1200 miles. There are few flights which travels higher distance.

# __________

# In[28]:


hist_plot(df['hour'], bins=24, title="Hour", xlabel="Hour of Schedule Departure Time")


# * _Hour_ :
#     
#     The histogram shows two peak periods for departure time related to flights which were operated at morning and evening.

# ____________

# In[21]:


bar_plot(df['carrier'], "Carriers Distribution", 'Carrier Code')


# * _Carriers_:  
# 
# This bar plot shows that most flights in this dataset were operated by __UA__, __B6__, __EV__ and __DL__ .

# __________

# In[22]:


bar_plot(df['origin'], title="Origin Point", xlabel="Origin Code")


# * _Origin Point_:
# 
#     This bar plot shows that most flight in this dataset were operated from EWR , JFK and LGA. 

# In[23]:


bar_plot(df['dest'], "Destination Plot", "Destination Code")


# * _Destionation Point_ :
# 
# This plot shows that most commmon destinations are ORD, ATL and LAX.

# _________

# In[29]:


bar_plot(df['name'], title="Airlines Title", xlabel="Airlines")


# * _Names_ :
# 
# This bar plot shows number of flights which were operated by each airlines. We can see that most flights were operated by _Unined Airlines_, _Jetblue_ and _Expressjet Airlines_. This plot distribution is same as Carrier Code's plot.

# __________________

# ___We can analyze relationship between features by targetin `arr_delay` as our main target. For this scenario we can scatter plots for numeric feature , and violin plot for categorical features.___
# 
# __As `id`, `flight`, `tailnum` and `time_hour` does not provide meaningful information we can ignore them in analysis.__

# In[83]:


# Scatter plot function.
def scatter_plot(x, y, title, xlabel, ylabel):
    sns.set_style(style='darkgrid')
    plt.figure(figsize=(10, 4))
    sns.scatterplot(x=x, y=y, hue=y, palette='coolwarm')
    plt.title(title, fontsize=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    plt.show()


# In[59]:


# Violin plot function.
def violin_plot(x, y, title, xlable, ylabel, fontisze=8):
    plt.figure(figsize=(10, 4))
    sns.set_style(style='darkgrid')
    sns.violinplot(x=x, y=y, palette="Set2")
    plt.title(title, fontsize=15)
    plt.xlabel(xlable, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=90, fontsize=fontisze)
    plt.show()


# _________

# In[60]:


violin_plot(df['year'], df['arr_delay'], "Year Vs Arrival Delay", "Year", "Arrival Delay")


# _Year Vs Arrival Delay_: 
# 
# This plot does not provide much information as this dataset contains flights information for 2013 only.

# _________________

# In[61]:


violin_plot(df['month'], df['arr_delay'], "Month Vs Arrival Delay", "Month", "Arrival Delay")


# _Month Vs Arrival Delay_:
# 
# This plot shows that distribution of arrival delays varies by each month. Some Months such as June, July and December have wider distribution indicating a higher variability in arrival delays.

# ________

# In[62]:


violin_plot(df['day'], df['arr_delay'], "Day Vs Arrival Delay", "Day", "Arrival Delay")


# _Day Vs Arrival Delay_ :
# 
# This plot shows that there are not much effects on arrival delays for each day.

# _________________

# In[74]:


scatter_plot(df['dep_time'], df['arr_delay'], "Departure Time Vs Arrival Delay", "Departure Time", "Arrival Delay")


# _Departure Time Vs Arrival Delay_ :
# 
# This scatter plots show that flights departing later in the day tend to have more delays. 

# ________

# In[77]:


scatter_plot(df['sched_dep_time'], df['arr_delay'], "Schedule Departure Time Vs Arrival Delay", "Schedule Departure", "Arrival Delays")


# _Schedule Departure Time Vs Arrival Delay_:
# 
# This plots show information as "Departure Time Vs Arrival Delay" plot.

# _________

# In[78]:


scatter_plot(df['dep_delay'], df['arr_delay'], "Departure Delay Vs Arrival Delay", "Departure Delay", "Arrival Delay")


# _Departure Delay Vs Arrival Delay_ :
# 
# As it is obivious , there is strong relationship between depature delay and arrival delay. There more flights which have more departure delay would have more arrival delay.

# ________________

# In[82]:


scatter_plot(df["arr_time"], df['arr_delay'], "Arrival Time Vs Arrival Delay", "Arrival Time", "Arrival Delay")


# _Arrival Time Vs Arrival Delay_:
# 
# This scatter plots indicates that flights arriving later in the day tend to have more delays.

# __________

# In[81]:


scatter_plot(df['sched_arr_time'], df['arr_delay'], "Schedule Arrival Time Vs Arrival Delay", "Sch Arrival Time", "Arrival Delay")


# _Sch Arrival Time Vs Arrival Delay_:
# 
# This plots show that flight scheduled to arrive later in the day tend to have more delays.

# ___________

# In[85]:


violin_plot(df['carrier'], df['arr_delay'], "Carrier Vs Arrival Delay", "Carrier", "Arrival Delay")


# _Carrier Vs Arrival Delay_:
# 
# This violin plot shows that different airlies have different arrival delays.

# In[86]:


violin_plot(df['origin'], df['arr_delay'], "Origin Vs Arrival Delay", "Origin", "Arrival Delay")


# _Origin Vs Arrival Delay_:
# 
# This plot shows that delays in each airport are varies.

# _______________

# In[87]:


violin_plot(df['dest'], df['arr_delay'], "Destination Vs Arrival Delay", "Destination", "Arrival Delay")


# _Destination Vs Arrival Delay_ :
# 
# This plot shows that each destination airport tend to have different delays. Some destination have more delays in their arrival time

# _______________

# In[88]:


scatter_plot(df['air_time'], df['arr_delay'], "Air Time Vs Arrival Delay", "Air Time", "Arrival Delay")


# _Air Time Vs Arrival Delay_:
# 
# This scatter plot does not any clear trend and relationship between these two feature.

# ______

# In[89]:


scatter_plot(df['distance'], df['arr_delay'], "Distance Vs Arrival Delays", "Distance", "Arrival Delay")


# _Distance Vs Arrival Delay_:
# 
# Also this plot does not show clear trend and relationship between these two feature.

# ___________________

# In[90]:


violin_plot(df['hour'], df['arr_delay'], "Hour Vs Arrival Delay", "Hour", "Arrival Delay")


# _Hour Vs Arrival Delay_:
# 
# We can understand from this violin plot that those flight which departing later in the day tend to have higher variability in arrival delays.

# _______________

# In[101]:


# Correlation heatmap between numeric feature
plt.figure(figsize=(18, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='viridis')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("Correlation Heatmap", fontsize=15)
plt.show()


# Based on analyzing plots, below feature have a noticeable affect on arrival delay:
# 
# 1. Month
# 2. Departure Time
# 3. Departure Delay
# 4. Arrival Time
# 5. Carrier
# 6. Origin
# 7. Destination
# 8. Hour
