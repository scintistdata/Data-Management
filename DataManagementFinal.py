#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/SameulAH/Data_Science/blob/main/Untitled14.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[4]:


get_ipython().system('pip install WordCloud')


# In[1]:


import pandas as pd


# In[2]:


import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


# In[3]:


df = pd.read_csv('./Downloads/grandFinaleD.csv')


# In[7]:


df.columns


# In[4]:


df.drop('Unnamed: 0',axis=1, inplace=True)


# In[5]:


df.columns


# In[46]:


df.shape, df.columns, df.info()


# In[9]:


# checking for null values
df.isnull().sum()


# In[ ]:





# In[47]:


df.Genre


# In[ ]:





# In[ ]:





# In[4]:


import pandas as pd


expanded_df = pd.DataFrame()


for index, row in df.iterrows():
    
    genre = row["Genre"]
    genre = genre.replace("'", "")
    
    
   
    genres_list = genre.strip("[]'").split(", ")
    for genre_item in genres_list:
        new_row = row.copy()  
        new_row["Genre"] = genre_item 
        expanded_df = expanded_df.append(new_row) 


expanded_df = expanded_df.reset_index(drop=True)


#expanded_df.to_csv("expanded_dataset.csv", index=False)  


# In[5]:


expanded_df.head()


# In[6]:


genre_counts = expanded_df['Genre'].value_counts()
print(genre_counts)


# In[ ]:





# In[94]:


import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the count of each genre
genre_counts = expanded_df['Genre'].value_counts()

# Select the top 30 most popular genres
top_30_genres = genre_counts.head(30)

# Plot the count of songs for the top 30 genres
plt.figure(figsize=(12, 8))
sns.barplot(x=top_30_genres.index, y=top_30_genres.values)
plt.xlabel('Genre')
plt.ylabel('Song Count')
plt.title('Top 30 Most Popular Genres')
plt.xticks(rotation=90)
plt.show()


# In[128]:





df['Year'] = pd.to_datetime(df['Release Date']).dt.year

# Group the dataset by year and genre and calculate the genre counts
yearly_genre_counts = expanded_df.groupby(['Year', 'Genre']).size().reset_index(name='Count')

# Iterate over each year and plot the top 30 genres
for year in df['Year'].unique():
    # Filter the dataset for the current year
    year_data = yearly_genre_counts[yearly_genre_counts['Year'] == year]

    # Sort the genres by count in descending order
    sorted_genres = year_data.sort_values('Count', ascending=False).head(10)

    # Plot the top 10 genres
    plt.figure(figsize=(12, 6))
    plt.bar(sorted_genres['Genre'], sorted_genres['Count'])
    plt.xlabel('Genre')
    plt.ylabel('Count')
    plt.title(f'Top 10 Genres - {year}')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


# In[ ]:





# In[121]:


expanded_df['Year'] = pd.to_datetime(expanded_df['Release Date']).dt.year

# Group the dataset by year and genre and calculate the genre counts
yearly_genre_counts = expanded_df.groupby(['Year', 'Genre']).size().reset_index(name='Count')

# Create a figure and axis object
fig, ax = plt.subplots(figsize=(12, 6))

# Iterate over each year and plot the top 30 genres on the same plot
for year in expanded_df['Year'].unique():
    # Filter the dataset for the current year
    year_data = yearly_genre_counts[yearly_genre_counts['Year'] == year]

    # Sort the genres by count in descending order and select the top 10
    sorted_genres = year_data.sort_values('Count', ascending=False).head(10)

    # Plot the top 30 genres as a bar chart
    ax.bar(sorted_genres['Genre'], sorted_genres['Count'], label=str(year))

# Set labels and title
ax.set_xlabel('Genre')
ax.set_ylabel('Count')
ax.set_title('Top 10 Genres by Year')
ax.legend(title='Year', bbox_to_anchor=(1.02, 1), loc='upper left')

# Rotate x-axis labels for better visibility
plt.xticks(rotation=90)

# Adjust spacing and display the plot
plt.tight_layout()
plt.show()


# In[130]:



# Convert the 'Release Date' column to datetime and extract the year
expanded_df['Year'] = pd.to_datetime(expanded_df['Release Date']).dt.year

# Group the dataset by year and genre and calculate the mean popularity
yearly_genre_popularity = expanded_df.groupby(['Year', 'Genre'])['Popularity'].mean().reset_index()

# Iterate over each year
for year in df['Year'].unique():
    # Filter the dataset for the current year
    year_data = yearly_genre_popularity[yearly_genre_popularity['Year'] == year]

    # Sort the genres by popularity in descending order
    sorted_genres = year_data.sort_values('Popularity', ascending=False).head(10)

    # Plot the popularity vs. genre for the current year
    plt.figure(figsize=(12, 6))
    plt.bar(sorted_genres['Genre'], sorted_genres['Popularity'], color='b', alpha=0.5)
    plt.xlabel('Genre')
    plt.ylabel('Popularity')
    plt.title(f'Popularity by Genre - {year}')
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Show the plot
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[10]:


fig=px.imshow(df.corr(),text_auto=True,height=800,width=800,color_continuous_scale=px.colors.sequential.Greens,aspect='auto',title='<b>correlation of columns')
fig.update_layout(title_x=0.5)
fig.show()


# In[ ]:





# Year By year song collection

# In[11]:


# fig=px.area(df.groupby('Year',as_index=False).count().sort_values(by='Track_Name',ascending=False).sort_values(by='Year'),x='Year',y='Track_Name',markers=True,labels={'Track_name':'Total songs'},color_discrete_sequence=['green'],title='<b>Year by Year Songs collection')
# fig.update_layout(hovermode='x',title_x=0.5)


# In[ ]:





# In[11]:


fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))

axes[0, 0].hist(df['Popularity'])
axes[0, 0].set_title('Popularity')

axes[0, 1].hist(df['Danceability'])
axes[0, 1].set_title('Danceability')

axes[0, 2].hist(df['Energy'])
axes[0, 2].set_title('Energy')

axes[1, 0].hist(df['Loudness'])
axes[1, 0].set_title('Loudness')

axes[1, 1].hist(df['speechiness'])
axes[1, 1].set_title('Speechiness')

axes[1, 2].hist(df['acousticness'])
axes[1, 2].set_title('Acousticness')

axes[2, 0].hist(df['liveness'])
axes[2, 0].set_title('Liveness')

axes[2, 1].hist(df['valence'])
axes[2, 1].set_title('Valence')

axes[2, 2].hist(df['tempo'])
axes[2, 2].set_title('Tempo')

plt.tight_layout()
plt.show()


# In[ ]:





# In[12]:


cm = df.corr(method = 'pearson')
plt.figure(figsize=(14,6))
map = sns.heatmap(cm, annot = True, fmt = '.1g', vmin=-1, vmax=1, center=0, cmap='inferno', linewidths=1, linecolor='Black')
map.set_title('Correlation Heatmap between Variable')
map.set_xticklabels(map.get_xticklabels(), rotation=90)


# In[ ]:





# In[14]:


# Assuming 'years' is a pandas Series or DataFrame column containing the years

plt.figure(figsize=(10, 5))
df['Year'].value_counts().sort_index().plot(kind='bar')
plt.title('Number of songs per year')
plt.xlabel('Year')
plt.ylabel('Count')
plt.show()


# In[ ]:





# In[13]:


df_grouped = df.groupby('Genre', as_index=False).count().sort_values(by='Track_Name', ascending=False)

fig = px.bar(df_grouped, x='Genre', y='Track_Name', color_discrete_sequence=['green'], template='plotly_dark', title='<b>Total songs based on genres</b>')
fig.update_layout(title_x=0.5)

box_fig = px.box(df, x='Genre', y='Track_Name')
fig.add_trace(box_fig.data[0])

fig.show()


# In[ ]:





# In[14]:


df_grouped = df.groupby('Artist', as_index=False).count().sort_values(by='Track_Name', ascending=False).head(50)

fig = px.bar(df_grouped, x='Artist', y='Track_Name', labels={'Track_Name': 'Total Songs'}, width=1000, color_discrete_sequence=['green'], text='Track_Name', title='<b>List of Songs Recorded by Each Singer')
fig.update_layout(title_x=0.5)

fig.show()


# In[ ]:





# In[15]:


df_grouped = df.groupby('Artist', as_index=False).sum().sort_values(by='Popularity', ascending=False).head(30)

fig = px.bar(df_grouped, x='Artist', y='Popularity', color_discrete_sequence=['lightgreen'], template='plotly_dark', text='Popularity', title='<b>Top 30 Popular Singers')
fig.update_layout(title_x=0.5)

fig.show()


# In[ ]:





# In[91]:


df_sorted = expanded_df.sort_values(by='Popularity', ascending=False).head(30)

fig = px.line(df_sorted, x='Track_Name', y='Popularity', hover_data=['Artist'], color_discrete_sequence=['green'], markers=True, title='<b>Top 30 songs in Spotify')
fig.show()


# In[ ]:





# In[90]:


fig = px.treemap(expanded_df, path=[px.Constant('Artist'), 'Artist', 'Genre', 'Track_Name'], values='Popularity', title='<b>TreeMap of Singers Playlist')
fig.update_traces(root_color='lightgreen')
fig.update_layout(title_x=0.5)

fig.show()


# In[ ]:





# In[8]:


fig = px.scatter(df, x='tempo', y='Popularity', color='tempo', color_continuous_scale=px.colors.sequential.Plasma, template='plotly_dark', title='<b>Tempo Versus Popularity')
fig.show()


# In[ ]:





# In[9]:


px.scatter(df,x='speechiness',y='Popularity',color='speechiness',color_continuous_scale=px.colors.sequential.Plasma,template='plotly_dark',title='<b> Speechiness Versus Popularity')


# In[ ]:





# In[10]:


px.scatter(df,x='Energy',y='Danceability',color='Danceability',color_continuous_scale=px.colors.sequential.Plotly3,template='plotly_dark',title='<b>Energy Versus Danceability')


# In[ ]:





# In[11]:


px.scatter(df,x='Energy',y='Loudness',color_discrete_sequence=['lightgreen'],template='plotly_dark',title='<b>Energy versus Loudness correlation')


# In[ ]:





# In[24]:


# plt.figure(figsize=(10,6))
# sns.regplot(data=sam, y='Loudness', x='Energy', color='c').set(title='Loudness vs Energy Correlation')


# In[ ]:





# In[25]:


# total_dr = tracks.groupby('Year')['duration'].sum().reset_index()
# years = total_dr['Year']
# total_duration = total_dr['duration']

# fig_dims = (18, 7)
# fig, ax = plt.subplots(figsize=fig_dims)

# sns.barplot(x=years, y=total_duration, ax=ax, errwidth=False)
# plt.xticks(rotation=90)
# plt.title('Years vs Duration')
# plt.xlabel('Year')
# plt.ylabel('Total Duration')
# plt.tight_layout()
# plt.show()


# In[12]:


df.columns


# In[ ]:





# #create a bar plot showing the relationship between years and song durations
# 

# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt

total_dr = df.groupby('Year')['duration_ms'].sum().reset_index()
years = total_dr['Year']
total_duration = total_dr['duration_ms']

fig_dims = (18, 7)
fig, ax = plt.subplots(figsize=fig_dims)

sns.barplot(x=years, y=total_duration, ax=ax, errwidth=False)
plt.xticks(rotation=90)
plt.title('Years vs Duration')
plt.xlabel('Year')
plt.ylabel('Total Duration')
plt.tight_layout()
plt.show()


# In[ ]:





# In[28]:


# artist_streams = df.groupby(['Year', 'Artist'])['Streams'].sum().reset_index(name='Total Streams')


# In[29]:


# import matplotlib.pyplot as plt

# # Bar plot for artist counts
# plt.figure(figsize=(12, 6))
# plt.bar(sorted_counts['Artist'], sorted_counts['Song Count'])
# plt.xlabel('Artist')
# plt.ylabel('Song Count')
# plt.title('Popular Artists by Song Count')
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.show()


# In[ ]:





# In[30]:


df.columns


#  #calculate the average popularity score for each year using the mean() function.

# In[138]:


popularity_trends = df.groupby('Year')['Popularity'].mean()


# #1. Plot the average popularity of the top 400 songs each year to see if there are any significant changes or shifts in music preferences.

# In[139]:


import matplotlib.pyplot as plt

# Line plot
plt.plot(popularity_trends.index, popularity_trends.values)
plt.xlabel('Year')
plt.ylabel('Average Popularity')
plt.title('Popularity Trends of Songs (2010-2022)')
plt.show()


# 

# # 2.Analyze the genre distribution among the top 400 songs each year from 2010 to 2022

# #create a bar chart or pie chart to visualize the proportion of different genres over time and observe any shifts or emerging trends.

# In[ ]:





# In[33]:


#Count the occurrences of each genre: Use the value_counts() function to count the occurrences of each genre in the subset dataframe


# In[140]:


genre_counts = df['Genre'].value_counts()


# In[141]:


genre_counts


# In[36]:


#Plot the genre distribution using a bar chart or a pie chart.


# In[ ]:





# 

# # 3.Analyze the danceability and energy scores of the top 400 songs each year from 2010 to 2022,

# Calculate the average danceability and energy scores for the top 400 songs each year

# Plotting these scores over time can help you identify any changes in the overall mood and energy of popular songs.

# In[ ]:





# In[ ]:





# In[38]:


danceability_trends = df.groupby('Year')['danceability'].mean()
energy_trends = df.groupby('Year')['Energy'].mean()


# In[39]:


# Group by year and calculate average danceability and energy scores using mean


# In[40]:


# Visualize the danceability and energy trends using line plots or bar charts.


# In[41]:


import matplotlib.pyplot as plt

# Line plot for danceability
plt.plot(danceability_trends.index, danceability_trends.values)
plt.xlabel('Year')
plt.ylabel('Average Danceability')
plt.title('Danceability Trends of Top Songs (2010-2022)')
plt.show()

# Line plot for energy
plt.plot(energy_trends.index, energy_trends.values)
plt.xlabel('Year')
plt.ylabel('Average Energy')
plt.title('Energy Trends of Top Songs (2010-2022)')
plt.show()


# 

# # 4.keyword extraction from the lyrics of the top 400 songs each year from 2010 to 2022 using techniques like TF-IDF or TextRank

# In[42]:


#Clean and preprocess the lyrics data to remove any noise or irrelevant information. This can involve steps such as removing punctuation, converting text to
#lowercase, and removing stopwords.


# In[43]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import nltk
nltk.download('punkt')

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Function to preprocess lyrics
def preprocess_lyrics(lyrics):
    # Convert to lowercase
    lyrics = lyrics.lower()

    # Remove punctuation
    lyrics = lyrics.translate(str.maketrans('', '', string.punctuation))

    # Tokenize the lyrics
    tokens = word_tokenize(lyrics)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Join the tokens back into a string
    preprocessed_lyrics = ' '.join(tokens)

    return preprocessed_lyrics

# Apply preprocessing to the lyrics column
df['Preprocessed_Lyrics'] = df['Lyrics'].apply(preprocess_lyrics)


# In[44]:


df['Preprocessed_Lyrics']


# In[45]:


#Perform keyword extraction: You can use either TF-IDF or TextRank to extract keywords from the preprocessed lyrics. Here's an example for each technique:
#a. TF-IDF (Term Frequency-Inverse Document Frequency):


# In[46]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the preprocessed lyrics
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Preprocessed_Lyrics'])

# Get the feature names (keywords)
feature_names = tfidf_vectorizer.get_feature_names_out()


# In[47]:


#The tfidf_matrix will represent the lyrics in a matrix format where rows correspond to songs and columns correspond to unique keywords.
#The feature_names list contains the extracted keywords from the lyrics.


# In[48]:


tfidf_matrix


# In[49]:


#Calculate the average TF-IDF score for each keyword:


# In[50]:


# Calculate the average TF-IDF score for each keyword
average_tfidf_scores = tfidf_matrix.mean(axis=0).tolist()[0]
keyword_scores = list(zip(feature_names, average_tfidf_scores))

# Sort the keyword scores by the TF-IDF score in descending order
keyword_scores = sorted(keyword_scores, key=lambda x: x[1], reverse=True)


# In[51]:


#The keyword_scores list contains tuples with the keyword and its corresponding average TF-IDF score.

#Visualize the TF-IDF scores:


# In[52]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Create a dictionary of keyword scores
keyword_scores_dict = {keyword: score for keyword, score in keyword_scores}

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(keyword_scores_dict)

# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('TF-IDF Word Cloud')
plt.show()


# In[53]:


#This will generate a word cloud visualization where the size of each keyword represents its TF-IDF score.


# In[54]:


#b. Bar Chart:


# In[55]:


# Get the top N keywords and their scores
top_n = 10  # Choose the desired number of top keywords to display
top_keywords = [keyword for keyword, _ in keyword_scores[:top_n]]
top_scores = [score for _, score in keyword_scores[:top_n]]

# Plot the bar chart
plt.figure(figsize=(10, 5))
plt.bar(top_keywords, top_scores)
plt.xlabel('Keyword')
plt.ylabel('Average TF-IDF Score')
plt.title('Top {} Keywords by TF-IDF Score'.format(top_n))
plt.xticks(rotation=45)
plt.show()


# This will create a bar chart visualization showing the top N keywords and their corresponding average TF-IDF scores.

# In[ ]:





# In[ ]:





# ##5. To perform an artist analysis and explore the artists who have appeared most frequently in
# #count the number of occurrences for each artist using the value_counts() function.

# In[56]:


artist_counts = df['Artist'].value_counts()
artist_counts


# #visualize the Artist analysis

# In[57]:


import matplotlib.pyplot as plt

# Select the top N artists
top_n = 10  # Choose the desired number of top artists to display
top_artists = artist_counts.head(top_n)

# Plot the bar chart
plt.figure(figsize=(10, 5))
plt.bar(top_artists.index, top_artists.values)
plt.xlabel('Artist')
plt.ylabel('Song Appearances')
plt.title('Top {} Artists in the Top Songs (2010-2022)'.format(top_n))
plt.xticks(rotation=45)
plt.show()


# #World CLoud

# In[58]:


from wordcloud import WordCloud

# Create a dictionary of artist counts
artist_counts_dict = {artist: count for artist, count in artist_counts.items()}

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(artist_counts_dict)

# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Artist Word Cloud (2010-2022)')
plt.show()


# #6. To perform a release date analysis and examine the distribution of song releases throughout the year

# Extract month and season from the release date

# In[59]:


import pandas as pd

# Convert the 'Release Date' column to datetime format
df['Release Date'] = pd.to_datetime(df['Release Date'])

# Extract month and season from the release date
df['Month'] = df['Release Date'].dt.month
df['Season'] = pd.cut(df['Month'], bins=[0, 3, 6, 9, 12], labels=['Winter', 'Spring', 'Summer', 'Fall'])


# #Calculate the song release distribution:

# a. By Month

# In[60]:


# Calculate the number of songs released each month
release_month_distribution = df['Month'].value_counts().sort_index()

# Plot the line chart or bar chart for release month distribution
plt.plot(release_month_distribution.index, release_month_distribution.values)
plt.xlabel('Month')
plt.ylabel('Number of Songs Released')
plt.title('Song Release Distribution by Month (2010-2022)')
plt.show()


# b. By season:

# In[61]:


# Calculate the number of songs released in each season
release_season_distribution = df['Season'].value_counts()

# Plot the bar chart for release season distribution
plt.bar(release_season_distribution.index, release_season_distribution.values)
plt.xlabel('Season')
plt.ylabel('Number of Songs Released')
plt.title('Song Release Distribution by Season (2010-2022)')
plt.show()


# #7. Audio features comparison and compare the average values of audio features such as danceability, energy, loudness, and valence for the top 400 songs each year from 2010 to 2022,

# #Group by year and calculate average audio feature values:

# In[62]:


audio_features = ['danceability', 'Energy', 'loudness', 'valence']
audio_feature_comparison = df.groupby('Year')[audio_features].mean()


# In[63]:


audio_feature_comparison


# Plot the line charts or bar charts to compare the average values of the audio features over time.

# a. Line charts:

# In[64]:


import matplotlib.pyplot as plt

# Plot the line charts for audio feature comparison
for feature in audio_features:
    plt.plot(audio_feature_comparison.index, audio_feature_comparison[feature], label=feature)

plt.xlabel('Year')
plt.ylabel('Average Value')
plt.title('Audio Features Comparison (2010-2022)')
plt.legend()
plt.show()


# line chart for each audio feature, where the x-axis represents the years, and the y-axis represents the average value of the audio feature.

# 

# In[ ]:





# bar chart for each audio feature, where the x-axis represents the years, and the y-axis represents the average value of the audio feature.

# # 8. perform a correlation analysis and explore the correlations between different audio features of the top 400 songs each year from 2010 to 2022,

# let's consider the audio features 'danceability', 'energy', 'loudness', and 'valence'.

#  Compute the correlation matrix for the selected audio features using the corr() function.

# In[66]:


audio_features = ['danceability', 'Energy', 'loudness', 'valence']
# audio_feature_comparison = df.groupby('Year')[audio_features].mean()/
correlation_matrix = df[audio_features].corr()


# In[67]:


# import seaborn as sns
# import matplotlib.pyplot as plt

# Create a heatmap of the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Correlation Matrix of Audio Features')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


df.columns


# In[ ]:





# In[16]:


def add_spacy_data(dataset, feature_column):
    '''
    Grabs the verb, adverb, noun, and stop word Parts of Speech (POS) 
    tokens and pushes them into a new dataset. returns an 
    enriched dataset.
    
    Parameters:
    
    dataset (dataframe): the dataframe to parse
    feature_column (string): the column to parse in the dataset.
    
    Returns: 
    dataframe
    '''
    
    verbs = []
    nouns = []
    adverbs = []
    corpus = []
    nlp = spacy.load('en_core_web_sm')
    ##
    for i in range (0, len(dataset)):
        print("Extracting verbs and topics from record {} of {}".format(i+1, len(dataset)), end = "\r")
        song = dataset.iloc[i][feature_column]
        doc = nlp(song)
        spacy_dataframe = pd.DataFrame()
        for token in doc:
            if token.lemma_ == "-PRON-":
                    lemma = token.text
            else:
                lemma = token.lemma_
            row = {
                "Word": token.text,
                "Lemma": lemma,
                "PoS": token.pos_,
                "Stop Word": token.is_stop
            }
            spacy_dataframe = spacy_dataframe.append(row, ignore_index = True)
        verbs.append(" ".join(spacy_dataframe["Lemma"][spacy_dataframe["PoS"] == "VERB"].values))
        nouns.append(" ".join(spacy_dataframe["Lemma"][spacy_dataframe["PoS"] == "NOUN"].values))
        adverbs.append(" ".join(spacy_dataframe["Lemma"][spacy_dataframe["PoS"] == "ADV"].values))
        corpus_clean = " ".join(spacy_dataframe["Lemma"][spacy_dataframe["Stop Word"] == False].values)
        corpus_clean = re.sub(r'[^A-Za-z0-9]+', ' ', corpus_clean)   
        corpus.append(corpus_clean)
    dataset['Verbs'] = verbs
    dataset['Nouns'] = nouns
    dataset['Adverbs'] = adverbs
    dataset['Corpus'] = corpus
    return dataset


# In[145]:


get_ipython().system('pip install spacy')


# In[149]:


expanded_df.columns


# In[ ]:





# In[ ]:





# In[18]:


import spacy
import re


# In[163]:


get_ipython().system('pip install spacy')
get_ipython().system('python -m spacy download en')


# In[ ]:





# In[ ]:





# In[8]:


expanded_df.columns


# In[10]:


expanded_df.Year.unique()


# In[15]:



filtered_df = df[(df['Year'] >= 2010) & (df['Year'] <= 2022)]


selected_elements = pd.DataFrame()


for year in range(2010, 2023):
   
    year_df = filtered_df[filtered_df['Year'] == year]
    
    if len(year_df) >= 30:
        
        year_elements = year_df.sample(n=30, random_state=42)
        
        selected_elements = selected_elements.append(year_elements)

selected_elements


# In[ ]:





# # Evaluation Phase - Word Counts and Repetition

# In[14]:


selected_elements.head()


# In[20]:


prepared_songs_dataset = add_spacy_data(selected_elements, 'Lyrics')


# In[21]:


prepared_songs_dataset = prepared_songs_dataset.drop(columns = ['Unnamed: 0'])


# In[22]:


word_counts = []
unique_word_counts = []
for i in range (0, len(prepared_songs_dataset)):
    word_counts.append(len(prepared_songs_dataset.iloc[i]['Lyrics'].split()))
    unique_word_counts.append(len(set(prepared_songs_dataset.iloc[i]['Lyrics'].split())))
prepared_songs_dataset['Word Counts'] = word_counts
prepared_songs_dataset['Unique Word Counts'] = unique_word_counts


# In[23]:


prepared_songs_dataset.to_csv('prepped_data.csv')


# In[24]:


display(prepared_songs_dataset.iloc[0]['Adverbs'])


# In[28]:


characteristics = prepared_songs_dataset.groupby('Year').count()


# In[29]:


summary_dataset = pd.DataFrame()
years = prepared_songs_dataset['Year'].unique().tolist()
for i in range(0, len(years)):
    row = {
        "Year": years[i],
        "Average Words": prepared_songs_dataset['Word Counts'][prepared_songs_dataset['Year'] == years[i]].mean(),
        "Unique Words": prepared_songs_dataset['Unique Word Counts'][prepared_songs_dataset['Year'] == years[i]].mean()
    }
    summary_dataset = summary_dataset.append(row, ignore_index=True)
summary_dataset["Year"] = summary_dataset['Year'].astype(int)


# In[26]:


characteristics = prepared_songs_dataset.groupby('Year').count()


# In[31]:


plt.figure(figsize=(20,10), dpi=200)
plt.plot(summary_dataset['Year'], summary_dataset['Average Words'].values, color="red", label="Average Words")
plt.plot(summary_dataset['Year'], summary_dataset['Unique Words'].values, color="green", label = "Unique Words")
plt.plot(characteristics['Popularity'], color="blue", label = "Number of Songs Per Year")
plt.xticks(summary_dataset['Year'], rotation=90)
plt.grid()
plt.legend()
plt.show()


# In[33]:


plt.figure(figsize=(20,10), dpi=200)
plt.bar(summary_dataset['Year'], summary_dataset['Average Words'].values, color="red", label="Average Words")
plt.bar(summary_dataset['Year'], summary_dataset['Unique Words'].values, color="green", label = "Unique Words")
plt.bar(summary_dataset['Year'], characteristics['Popularity'], color="blue", label = "Number of Songs Per Year")
plt.xticks(summary_dataset['Year'], rotation=90)
plt.legend()
plt.show()


# In[35]:


fig, ax1 = plt.subplots(figsize=(20,10), dpi=200)
#plt.figure(figsize = (20,15))

color = 'tab:red'
ax1.set_xlabel('Years')
ax1.set_ylabel('Average Words', color=color)
ax1.plot(summary_dataset['Year'], summary_dataset['Average Words'].values, color=color)
ax1.tick_params(axis='y', labelcolor=color)
plt.xticks(summary_dataset['Year'], rotation=90)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:green'
ax2.set_ylabel('Unique Words', color=color)  # we already handled the x-label with ax1
ax2.plot(summary_dataset['Year'], summary_dataset['Unique Words'].values, color=color)
ax2.tick_params(axis='y', labelcolor=color)

color = 'tab:blue'
ax3 = ax2.twinx()  # instantiate a second axes that shares the same x-axis
ax3.set_ylabel("Number of Songs per Year", color = color)
ax3.plot(characteristics['Popularity'])
ax3.tick_params(axis='y', labelcolor=color)
#plt.plot(characteristics['Rank'])
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.xticks(summary_dataset['Year'], rotation=90)

plt.grid()
plt.show()


# In[41]:


prepared_songs_dataset.sort_values(by=['Unique Word Counts'], ascending=False).head(10)


# In[45]:


import wordcloud


# In[54]:


import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

for year in range(2010, 2023):
    year_df = prepared_songs_dataset[prepared_songs_dataset['Year'] == year]

    nouns_text = ' '.join(year_df['Nouns'].astype(str))

    if not nouns_text:
        # Skip generating the word cloud if nouns_text is empty
        continue

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(nouns_text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(str(year))
    plt.axis('off')
    plt.show()


# # Calculating Term Frequencies

# In[56]:


from collections import Counter


# In[57]:


word_frequencies = pd.DataFrame()
for i in range (0, len(years)):
    year_corpus = str(prepared_songs_dataset['Corpus'][prepared_songs_dataset['Year'] == years[i]].tolist())
    tokens = year_corpus.split(" ")
    counts = Counter(tokens)
    word_frequencies = word_frequencies.append({
        "Year": years[i],
        "Most Common Words": counts.most_common(n=100)
    }, ignore_index=True)
word_frequencies['Year'] = word_frequencies['Year'].astype(int)


# In[59]:


def map_popular_terms(dataset, feature_column, year_column):
    '''Function that counts the frequency of occurences of words in a dataset
    column. Returns a new dataset with those frequencies'''
    frequencies = pd.DataFrame()
    years = dataset[year_column].unique().tolist()
    for i in range (0, len(years)):
        year_corpus = str(dataset[feature_column][dataset[year_column] == years[i]].tolist())
        tokens = year_corpus.split(" ")
        counts = Counter(tokens)
        frequencies = frequencies.append({
            "Year": years[i],
            "Most Common Terms": counts.most_common(n=100)
        }, ignore_index=True)
    frequencies['Year'] = frequencies['Year'].astype(int)
    return frequencies


# In[60]:


adverb_frequencies = map_popular_terms(prepared_songs_dataset, "Adverbs", "Year")
noun_frequencies = map_popular_terms(prepared_songs_dataset, "Nouns", "Year")
verb_frequencies = map_popular_terms(prepared_songs_dataset, "Verbs", "Year")
word_frequencies = map_popular_terms(prepared_songs_dataset, "Corpus", "Year")


# In[62]:


def map_common_words(dataset):
    '''Maps common words from across multiple columns in a dataset to 
    identify terms that show up in all columns. Normally used with the 
    outputs of map_popular_terms. returns the common words'''
    common_words = []
    for words in dataset['Most Common Terms'][0]:
        common_words.append(words[0])

    for i in range (0, len(dataset)):
        check_list = []
        year_list = dataset['Most Common Terms'][i]
        for words in year_list:
            check_list.append(words[0])
        common_words = [x for x in common_words if x  in check_list]
    return common_words


# In[64]:


common_adverbs = map_common_words(adverb_frequencies)
common_nouns = map_common_words(noun_frequencies)
common_words = map_common_words(word_frequencies)
common_verbs = map_common_words(verb_frequencies)


# In[68]:


import numpy as np
def get_common_frequency(term_list, frequency_list):
    '''Finds the frequency of occurence of terms in a list and then
    returns them in a new dataframe organized by year'''
    common_word_frequency_per_year = pd.DataFrame()
    for i in range(0, len(term_list)):
        word_frequency = []
        for j in range(0, len(frequency_list)):
            current_year = frequency_list['Year'][j]
            current_year_terms = frequency_list['Most Common Terms'][j]
            for words in current_year_terms:
                    if term_list[i] in words[0]:
                        word_frequency.append(words[1])
                        #print(words[1])
                        break
        current_word = term_list[i]
        common_word_frequency_per_year[str(current_word)] = word_frequency
    common_word_frequency_per_year["Year"] = np.arange(2010,2022)
    common_word_frequency_per_year = common_word_frequency_per_year.set_index("Year")
    return common_word_frequency_per_year


# In[69]:


common_adverb_counts = get_common_frequency(common_adverbs, adverb_frequencies)
common_noun_counts = get_common_frequency(common_nouns, noun_frequencies)
common_verb_counts = get_common_frequency(common_verbs, verb_frequencies)
common_word_counts = get_common_frequency(common_words, word_frequencies)


# In[70]:


top_five_nouns = []
for i in range (0, len(noun_frequencies)):
    for j in range (0, 10):
        top_five_nouns.append(noun_frequencies.iloc[i]['Most Common Terms'][j][0])
top_five_nouns = " ".join(top_five_nouns)
top_five_nouns = re.sub("what", "", top_five_nouns)


# In[72]:





# In[76]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

nouns_text = ' '.join(top_five_nouns)

wordcloud = WordCloud(max_words=100)
wordcloud.generate(nouns_text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[77]:


common_adverb_counts.plot(figsize=(35,15), title="Adverbs", grid="true", xticks=common_adverb_counts.index)
common_noun_counts.plot(figsize=(35,15), title="Nouns", grid="true", xticks=common_noun_counts.index)
common_verb_counts.plot(figsize=(35,15), title="Verbs", grid="true", xticks=common_verb_counts.index)
common_word_counts.plot(figsize=(35,15), title="Words", grid="true", xticks=common_word_counts.index)
plt.show()


# In[78]:


plt.figure(figsize=(20,15))
sns.heatmap(data=common_noun_counts)
plt.show()


# In[ ]:




