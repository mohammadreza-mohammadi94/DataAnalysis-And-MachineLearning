#!/usr/bin/env python
# coding: utf-8

# __Datasets__:  
# https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db  


# # Import `tracks.csv` dataset and necessary libraries

# * id: unique identifier for each track used by Spotify (randomly generated alphanumeric string)  
# * name: track name  
# * popularity: song popularity score as of March 2021 on a normalized scale [0-100] where 100 is the most popular  
# * duration_ms: duration of track in milliseconds  
# * explicit: true or false if the song contains explicit content.  
# * artists: name of the main artist  
# * id_artists: unique identifier for each artist used by Spotify  
# * release_date: when the album was released (date format: yyyy/mm/dd)  
# * danceability: describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.  
# * energy: measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy.  
# * key: The estimated overall key of the track. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is set to -1.  
# * loudness: The overall loudness of a track in decibels (dB). Values typical range between -60 and 0 db.  
# * mode: Mode indicates the modality (major=1 or minor=0) of a track, the type of scale from which its melodic content is derived.  
# * speechiness: measures from 0.0 to 1.0 and detects the presence of spoken words in a track. If the speechiness of a song is above 0.66, it is probably made of spoken words, a score between 0.33 and 0.66 is a song that may contain both music and words, and a score below 0.33 means the song does not have any speech.  
# * acousticness: confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic  
# * instrumentalness: measure from 0.0 to 1.0 and represents the amount of vocals in the song. The closer it is to 1.0, the more instrumental the song is.  
# * liveness: likelihood measure from 0.0 to 1.0 and indicates the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live.  
# * valence: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track.  
# * tempo: The overall estimated tempo of a track in beats per minute (BPM)  
# * time_signature: An estimated overall time signature of a track. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure).  ach bar (or measure).

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


# In[103]:


df_tracks = pd.read_csv('tracks.csv')


# # Check Structure of `df_tracks` dataframe

# In[104]:


# check head to dataframe
df_tracks.head(5)


# In[108]:


# an overvoew of features and their dtypes
df_tracks.info()


# from this output of info() method we understand that:  
# - There are __9__ Features with `float` dtype
# - There are __6__ Features with `int` dtype
# - There are __5__ Features with `object` dtype

# In[48]:


df_tracks.shape


# In[90]:


# check total number of null values in df_tracks
df_tracks.isnull().sum()


# - there are 71 missing values on `name` column

# In[109]:


# check if dataframe has duplicated rows
df_tracks.duplicated().sum()


# In[110]:


# df_tracks.describe()
df_tracks.describe().transpose()


# - `popularity` has mean of 27 and it is skewed to left which shows that mosts songs are unpopular.
# - `loudness` has negative values.
# - `speechiness`, `instrumentalness`, `liveness` are skewed to left.
# - `duration_ms` has large numbers and its better to convert its unit to seconds. 

# In[111]:


# check distribution of df_tracks
df_tracks.hist(bins = 100, color = 'green', figsize = (20, 14))


# ## Check categorical featuires

# In[128]:


# create a new df for cateforical features only
category_df = df_tracks.select_dtypes(include = 'object')


# In[129]:


category_df.info()


# ### Check values of each categorical features

# In[131]:


for col in category_df.columns:
    print(f"{col} : Has {category_df[col].nunique()} Unique Values")


# # Data Cleaning

# ## `name` Column

# ___Renaming `name` column to `track_name` for better understanding of its nature___

# In[112]:


df_tracks.rename(columns={'name' : 'track_name'}, inplace = True)


# ## `duration_ms` Column

# ___We will convert songs duration unit to seconds and renaming its column to `duration`___

# In[113]:


df_tracks['duration_ms'] = df_tracks['duration_ms'].apply(lambda x: round(x/1000))


# In[114]:


df_tracks.rename(columns = {'duration_ms': 'duration'}, inplace = True)


# In[115]:


df_tracks.info()


# ## `explicit` Column

# ___Generally, this means offensive words or curse words. These are the words that are generally found to be offensive and that are not normally said in regular or polite conversation.___
# ___This column is kind of categorical which contains only 0 and 1. So i will change its dtype to category dtype___

# In[225]:


df_tracks['explicit'].plot(kind = 'bar')


# In[117]:


df_tracks['explicit'] = df_tracks['explicit'].astype('category')


# ## `artist` Column

# In[118]:


df_tracks['artists'][0:10]


# ___In the "artists" column of the DataFrame, the artist names are currently represented as strings with square brackets around them.To clean up this representation, we want to extract the actual artist names and remove the square brackets___

# In[119]:


df_tracks['artists'] = df_tracks['artists'].str.strip("[]").str.replace("'", "")


# In[120]:


df_tracks['artists'][0:10] 


# ## `release_date` Column

# ___Convert `release_date` column's dtype to datetime___

# In[121]:


df_tracks['release_date'][:20] # to display the 20 first rows for better undestanding to its values.


# In[122]:


# change dtype to datetime
df_tracks['release_date'] = pd.to_datetime(df_tracks['release_date'], format = 'mixed')


# In[123]:


df_tracks['release_date'][:20] # to check whether converting is done correctly.


# 

# # `df_tracks` Analysis

# ## Popular Artists

# In[135]:


plt.figure(figsize = (20, 14))

def visualize_word_counts(counts):
    wc = WordCloud(max_font_size=130, min_font_size=25, colormap='tab20', background_color='white', prefer_horizontal=.95, width=2100, height=700, random_state=0)
    cloud = wc.generate_from_frequencies(counts)
    plt.figure(figsize=(18,15))
    plt.imshow(cloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


# In[137]:


# create a new df for top 20 artists in aspect of their number of songs
top_artist = df_tracks['artists'].value_counts().head(20)


# In[138]:


top_artist


# In[139]:


visualize_word_counts(top_artist)


# ### `top_artists` on barplot

# In[176]:


fig, ax = plt.subplots(figsize = (12, 6))
ax = sns.barplot(x = top_artist.values, y=top_artist.index, palette='rocket_r')
ax.set_xlabel("Sum of Songs", fontsize = 12)
ax.set_ylabel('Artist Name', fontsize = 12)
ax.set_title("Most Popular Artist", c = 'black', fontsize = 15)
plt.show()


# ## Most Popular Songs

# In[169]:


def popularity(overall, n):
    #n: represents number of rows shown
    return df_tracks.query(f'popularity > {overall}')[['track_name', 'popularity', 'artists', 'release_date']].sort_values(by = 'popularity', ascending = False).reset_index(drop=-1).head(n)


# In[170]:


popularity(90, 5)


# In[174]:


top_songs = popularity(90, 20)[['track_name', 'popularity']]


# In[175]:


top_songs


# In[185]:


fig, ax = plt.subplots(figsize = (12, 6))
ax = sns.barplot(y= top_songs['track_name'], x = top_songs['popularity'], color='skyblue')
ax.set_xlabel("Popularity", fontsize = 15)
ax.set_ylabel("Song Name", fontsize = 15)
ax.set_title("Top 20 Popular Songs")
plt.show()


# ## Top 10 danceability songs

# In[195]:


most_dance = df_tracks.sort_values(by = 'danceability', ascending = False).head(10)
most_dance


# In[202]:


plt.figure(figsize = (12, 6))
sns.barplot(y = most_dance['track_name'], x = most_dance['danceability'], color = 'lightgreen')
plt.ylabel('Track Name', fontsize = 15)
plt.xlabel('Danceability', fontsize = 15)
plt.title("Top 10 Dancable Songs")
plt.show()


# ## Top 10 energic songs

# In[208]:


most_energy = df_tracks.sort_values(by = 'energy', ascending = False)[['track_name', 'energy', 'artists']].reset_index(drop = -1).head(10)


# In[209]:


most_energy


# ## Distributions

# In[226]:


# Selecting numeric columns
numeric_columns = df_tracks.select_dtypes(include=['float64', 'int64']).columns

# Setting up subplots with dynamic rows based on the number of numeric columns
num_rows = (len(numeric_columns) + 1) // 2
fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(12, 4 * num_rows))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Plotting histograms for each numeric column
for i, column in enumerate(numeric_columns):
    sns.histplot(df_tracks[column], bins=50, kde=True, ax=axes[i], color='skyblue')
    axes[i].set_title(f'Distribution of {column}')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Count')

# Removing empty subplots if any
for j in range(len(numeric_columns), len(axes)):
    fig.delaxes(axes[j])

# Adjust layout
plt.tight_layout()
plt.show()


# ## Correlation Heatmap

# In[210]:


corr_df = df_tracks.drop(['key', 'mode', 'explicit'], axis = 1).corr(method = 'pearson', numeric_only=True)
corr_df


# In[214]:


plt.figure(figsize = (14, 6))
ax = sns.heatmap(corr_df, annot = True, fmt=".1g", vmin = -1, vmax = 1, center = 0, cmap='inferno', linecolor='Black')
plt.title("Correlation Heatmap")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)


# In[43]:


sample_df = df_tracks.sample(int(0.004 * len(df_tracks)))


# `sample` method in pandas to randomly select a fraction of rows from the DataFrame `df_tracks`.  
# `0.004 * len(df_tracks)`: Calculates 0.4% of the total number of rows.  
#  the code is creating a random sample (approximately 0.4% of the original size) of rows from the DataFrame df_tracks. This is commonly done to obtain a smaller, representative subset of data for analysis or exploration without having to work with the entire dataset.

# In[44]:


len(sample_df)


# ### Loudness Vs. Energy Correlation

# In[47]:


# regression plot for loudness vs energy
plt.figure(figsize = (10, 6))
sns.regplot(data = sample_df, y = "loudness", x = 'energy', color = 'g').set(title = "Loudness Vs. Energy Correlation")
plt.show()


# ### Popularity Vs. Acousticness Correlation

# In[52]:


# regression plot 
plt.figure(figsize = (10, 6))
sns.regplot(data = sample_df, y = "popularity", x = 'acousticness', color = 'c').set(title = "Popularity Vs. Acousticness Correlation")
plt.show()


# ### Duration Vs. Instrumentalness Correlation

# In[56]:


# regression plot 
plt.figure(figsize = (10, 6))
sns.regplot(data = sample_df, y = "duration", x = 'instrumentalness', color = 'r').set(title = "Duration Vs. Instrumentalness Correlation")
plt.show()


# ## Number of songs per year distribution

# In[217]:


years = df_tracks['release_date'].dt.year


# In[219]:


# songs distribution plot since 1922
sns.displot(years, discrete = True, aspect = 2, height = 5, kind = 'hist').set(title='Number of Songs Per Year')


# ## Duration Vs Years

# In[65]:


duration = df_tracks['duration']
fig_dims = (18, 7)
fig, ax = plt.subplots(figsize = fig_dims)
fig = sns.barplot(x = years, y = duration, ax = ax, errwidth = False).set(title="Year Vs Duration")
plt.xticks(rotation = 90)


# ## Duration Vs Years

# In[67]:


# duration
sns.set_style(style = 'whitegrid')
fig_dims = (10, 5)
fig, ax = plt.subplots(figsize = fig_dims)
fig = sns.lineplot(x = years, y = duration, ax = ax).set(title='Year Vs Duration')
plt.xticks(rotation = 90)


# ## Boxplots

# In[223]:


num_df = df_tracks.select_dtypes(include = 'number')
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


# ## Pairplot of `popularity`, `duration`, `danceability`, `energy`

# In[228]:


sns.pairplot(df_tracks[['popularity', 'duration', 'danceability', 'energy']])
plt.show()


# In[ ]:





# # Dataset #2

# In[68]:


df_genre = pd.read_csv('SpotifyFeatures.csv')


# In[69]:


df_genre.head()


# In[71]:


df_genre['genre'].value_counts()


# In[73]:


plt.title('Duration of The Songs in Different Genres')
sns.color_palette('rocket', as_cmap = True)
sns.barplot(y = 'genre', x = 'duration_ms', data = df_genre)
plt.xlabel("Duration")
plt.ylabel("Genres")


# In[ ]:





# In[ ]:





# In[74]:


sns.set_style(style = 'darkgrid')
plt.figure(figsize = (10, 5))
famous = df_genre.sort_values('popularity', ascending = False).head(10)
sns.barplot(y = 'genre', x = 'popularity', data = famous).set(title = 'Top5 Genres by Popularity')


# In[ ]:




