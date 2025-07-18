#!/usr/bin/env python
# coding: utf-8

# SENTIMENT ANALYSIS OF TWEETS
# Introduction
# Sentiment analysis of tweets task will use tweets sent to six airlines and attempt to identify whether they are positive, negative or neutral. This is a natural language processing and a classification task.
# 
# Sentiment analysis process will try two methods and select one with the most accuracy.
# 
# About the dataset
# The tweets dataset comprises of 13 features including:
# 
# airline_sentiment
# airline_sentiment_confidence
# negativereason
# negativereason_confidence
# airline
# airline_sentiment_gold
# name
# negativereason_gold
# retweet_count
# text
# tweet_coord
# tweet_created
# tweet_location
# user_timezone
# Import libraries and preview dataset

# In[1]:


get_ipython().system('pip install langdetect')
get_ipython().system('pip install textatistic')


# In[2]:


import os
for dirname, _, filenames in os.walk('C:/Users/JoanCarles/iCloudDrive/Documents/_Universitats/_Bachelors/Salle/data-science/Moduls/Module 5/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


import pandas as pd

# Read dataset
tweets_df = pd.read_csv('Airline-Sentiment-2-w-AA.csv', index_col='tweet_id')

# Display the first few rows of the DataFrame
tweets_df.head()


# In[5]:


# View dataset features
tweets_df.info()


# Descriptive Analysis¶
# Duration of tweets
# The tweets were sent between 16th Feb 2015 to 24th Feb 2015

# In[6]:


# Extracting features that we will use
tweets_df = tweets_df[['text','tweet_created','airline_sentiment']]
tweets_df


# In[7]:


tweets_df['tweet_created'] = pd.to_datetime(tweets_df.tweet_created, format = '%m/%d/%y %H:%M')
tweets_df.tweet_created.agg(['min','max'])


# Polarity of the tweets
# The tweets have been pre-classified into different emotions. Among the tweets, 62% are categorized as negative, indicating a predominantly negative sentiment, while 16% are classified as positive, suggesting a smaller proportion of positive sentiments. Additionally, 21% of the tweets are categorized as neutral, representing a neutral emotional tone. These pre-assigned emotion categories will serve as the feature against which we will evaluate the accuracy of our predictions.

# In[8]:


import matplotlib.pyplot as plt
# Plotting the pie chart
fig = plt.figure(figsize=(10,6))
tweets_df.airline_sentiment.value_counts().plot(kind='pie', label='', autopct='%1.1f%%', explode=[0.03,0.03,0.03], textprops={'fontsize':17, 'color':'k'}, startangle=60)
plt.title('Polarity of the Tweets')
plt.show()


# Preprocessing
# 
# Feature extraction
# Creating new features for:
# 
# *   Number of hashtag
# *   Number of mentions
# *   Number of characters
# *   Number of words
# 
# Number of hashtags in the tweet
# 

# In[9]:


import re
# Define the regular expression pattern for hashtags
pattern = r'#[a-zA-Z0-9]+'

# Apply the lambda function to count the number of hashtags in each text
hashtags = tweets_df.text.apply(lambda x: len(re.findall(pattern, x)))

# Add the 'hashtags' column to the 'tweets_df' DataFrame
tweets_df = tweets_df.assign(hashtags=hashtags.values)

# Display the first 5 rows
tweets_df.head(5)


# Number of mentions in the tweet

# In[10]:


# Define the regular expression pattern for mentions
pattern = r'@[a-zA-Z0-9]+'

# Apply the lambda function to count the number of mentions in each text
mentions = tweets_df.text.apply(lambda x: len(re.findall(pattern, x)))

# Add the 'mentions' column to the 'X' DataFrame
tweets_df = tweets_df.assign(mentions=mentions.values)

tweets_df.head(5)


# Number of characters in the tweet

# In[11]:


# Apply the lambda function to count the number of characters in each text
no_of_chars = tweets_df.text.apply(lambda x: len(x))

# Add the 'no_of_chars' column to the 'X' DataFrame
tweets_df = tweets_df.assign(no_of_chars=no_of_chars)

tweets_df.head()


# Number of words in the tweet

# In[12]:


get_ipython().system('pip install nltk')
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
# Tokenize the text in each tweet to generate a list of tokens
tokens = [word_tokenize(tweet) for tweet in tweets_df.text]

# Count the number of alphanumeric tokens in each tokenized tweet
words = [len([token for token in sent if token.isalnum()]) for sent in tokens]

# Add the 'no_of_words' column to the 'tweets_df' DataFrame
tweets_df = tweets_df.assign(no_of_words=words)

# Display the first few rows
tweets_df.head()


# Extracting the airline name¶
# We have tweets to six major airlines.

# In[13]:


# Identify text starting with the @ sign
pattern = r'@[a-zA-Z0-9]+'
tweets_df['airline_extracted'] = tweets_df.text.apply(lambda x: re.findall(pattern, x))

# Extract the name of the airline
tweets_df['airline_extracted'] = [item[0].lower() for item in tweets_df.airline_extracted]

# Remove the '@' sign from the name
tweets_df['airline_extracted'] = tweets_df['airline_extracted'].apply(lambda x: re.sub("@", '',str(x)))

tweets_df['airline_extracted'].value_counts()


# Cleaning airlines¶
# We will remove tweets not addressed to the six airlines.

# In[14]:


# Get the list of unique airlines in the 'airline' column of the 'X' DataFrame
not_airline = tweets_df['airline_extracted'].value_counts().index.tolist()

# Iterate over a list of specific airlines to remove them from the 'not_airline' list
for airline in ['united', 'usairways', 'americanair', 'southwestair', 'jetblue', 'virginamerica']:
    not_airline.remove(airline)


# In[15]:


import numpy as np
# Print the number of rows before the operation
print('Number of rows before =', tweets_df.shape[0])

# Set the 'airline_extracted' values to NaN for rows where 'airline_extracted' is in the 'not_airline' list
tweets_df.loc[tweets_df['airline_extracted'].isin(not_airline), 'airline_extracted'] = np.nan

# Drop rows with missing values in the 'airline_extracted' column
tweets_df.dropna(subset=['airline_extracted'], inplace=True)

# Print the number of rows after the operation
print('Number of rows after =', tweets_df.shape[0])


# Tokenization/lemmatization
# Tokenization and lemmatization are two key concepts in natural language processing (NLP) and text analysis.
# 
# Tokenization:
# 
# Definition: Tokenization is the process of breaking down text into smaller units, called tokens. These tokens can be words, numbers, or punctuation marks. In essence, it involves splitting sentences and phrases into individual words or meaningful elements.
# Purpose: The main purpose of tokenization is to simplify the text data for analysis. By breaking text into smaller parts, algorithms can more easily process and analyze the data. For example, in sentiment analysis, tokenization allows a machine learning model to work with individual words to determine the sentiment.
# Example: Consider the sentence "Tokenization is essential." Tokenization would break this down into the tokens "Tokenization", "is", and "essential".
# Lemmatization:
# 
# Definition: Lemmatization is a process in NLP where words are reduced to their base or root form. Unlike stemming, which crudely chops off the ends of words, lemmatization considers the context and converts the word to its meaningful base form, known as the lemma.
# Purpose: The goal is to reduce the inflectional forms of each word into a common base or root. It helps in standardizing words to their base form, which is essential for many text processing applications where the meaning of words is important, such as text classification, information retrieval, and natural language understanding.
# Example: For verbs, "am", "are", and "is" would all be lemmatized into "be". For nouns, "mice" would be lemmatized to "mouse".
# 
# In summary, while tokenization is about breaking text into smaller units for analysis, lemmatization is about reducing words to their base or dictionary form. Both are crucial preprocessing steps in many NLP tasks.

# In[16]:


get_ipython().system('pip install spacy')
get_ipython().system('python -m spacy download en_core_web_lg')
import spacy

nlp = spacy.load('en_core_web_lg')


# In[17]:


# Define the function for tokenization and lemmatization
def spacy_token(text):
    # Process the text using Spacy
    doc = nlp(text)
    # Extract the lemmatized tokens
    tokens = [token.lemma_ for token in doc]
    # Return the list of lemmatized tokens
    return tokens


# In[18]:


# Apply the spacy_token function to the tweets column
text_tokens = tweets_df.text.apply(spacy_token)
text_tokens


# Removing punctuations¶

# In[19]:


# Iterate over each row in the text_tokens DataFrame
for i in range(len(text_tokens)):
    # Remove punctuation from each token in the current row
    text_no_punct = [token for token in text_tokens.iloc[i] if token.isalnum()]
    # Update the current row in the text_tokens DataFrame with the modified tokens
    text_tokens.iloc[i] = text_no_punct

text_tokens


# Removing stopwords

# In[20]:


# Load stop words from spacy module
stopwords = spacy.lang.en.stop_words.STOP_WORDS


# In[21]:


# Iterate over each row in the text_tokens Series
for i in range(len(text_tokens)):
    # Convert each token to lowercase and remove stopwords
    tokens = [token.lower() for token in text_tokens.iloc[i] if token.lower() not in stopwords]
    # Join the processed tokens back into a string
    tokens_joined = " ".join(tokens)
    # Update the current row in the text_tokens DataFrame with the joined tokens
    text_tokens.iloc[i] = tokens_joined


# In[24]:


tweets_df.text = text_tokens
tweets_df


# Exploratory Data Analysis
# Word cloud
# To generate our wordcloud, we will join all the texts from the tweets. We will update our stopwords to include obvious words common in airports like flight, fly, plane, airport, airline e.t.c.
# 
# The word cloud generated has words such as "Thank," "time," "help," "delay," and "cancel" appearing often, which reveals the prevalent themes and sentiments in these tweets. The prominence of "Thank" suggests a high frequency of gratitude and appreciation expressed towards the airlines. The word "time" indicates a focus on flight schedules and punctuality. "Help" indicates a recurring theme of seeking assistance or support. The larger size of "delay" and "cancel" signifies that flight disruptions are commonly discussed.

# In[25]:


text = ''
for i in range(len(text_tokens)):
    text = text + " " + text_tokens.iloc[i]


# In[26]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud

fig = plt.figure(figsize=(14, 10))
cloud = WordCloud(background_color='black', colormap='Reds', min_font_size=7, stopwords=['flight', 'fly', 'plane', 'airport', 'airline', 'flightle']).generate(text)
plt.imshow(cloud)
plt.axis('off')
plt.tight_layout()
plt.show()



# Visualizing sample tweets¶
# We will visualize sample tweets with the words thank, time, delay, cancel, help, customer and service
# 
# Thank, Time, Delay, Cancel, Help
# A sample of 5 tweets with the words has tweets that sound positive and appreciative for thank, or complaining for time. delay, cancel, help and customer service

# In[29]:


import random

sample = []
for tweet in tweets_df.text.values:
    if 'thank' in tweet:
        sample.append(tweet)
random.seed(987)
random.sample(sample, 5)


# In[30]:


sample = []
for tweet in tweets_df.text.values:
    if 'time' in tweet:
        sample.append(tweet)
random.seed(987)
random.sample(sample, 5)


# In[31]:


sample = []
for tweet in tweets_df.text.values:
    if 'delay' in tweet:
        sample.append(tweet)
random.seed(987)
random.sample(sample, 5)


# In[33]:


sample = []
for tweet in tweets_df.text.values:
    if 'cancel' in tweet:
        sample.append(tweet)
random.seed(987)
random.sample(sample, 5)


# In[32]:


sample = []
for tweet in tweets_df.text.values:
    if 'help' in tweet:
        sample.append(tweet)
random.seed(987)
random.sample(sample, 5)


# In[34]:


sample = []
for tweet in tweets_df.text.values:
    if 'customer service' in tweet:
        sample.append(tweet)
random.seed(987)
random.sample(sample, 5)


# Airlines
# The bar chart shows the count of tweets directed to different airlines. United Airlines received the highest number of tweets with 3,817, suggesting a significant level of engagement and interaction on Twitter. US Airways and American Airlines followed closely with 2,905 and 2,754 tweets respectively. Southwest Airlines and JetBlue received 2,417 and 2,215 tweets respectively. tweets_df America had the lowest number of tweets with 504, suggesting relatively fewer mentions or interactions compared to other airlines.

# In[35]:


# Create a new figure with a specific size
fig = plt.figure(figsize=(8, 6))

# Generate a bar plot of the airline counts
tweets_df.airline_extracted.value_counts().plot(kind='bar', width=0.8)

# Set the plot labels
plt.xlabel('Airline')
plt.ylabel('Count')
plt.title('Airline addressed in the tweets')

# Adjust the layout of the plot to prevent overlapping
plt.tight_layout()

# Display the plot
plt.show()


# Sentiment Analysis using lexicon method¶

# In[36]:


get_ipython().system('pip install textblob')
from textblob import TextBlob

def lexicon_sentiment(df):
    """ Calculates the sentiment of text data in a DataFrame using the TextBlob library."""

    sentiment = []  # List to store the predicted sentiment labels
    sentiment_score = df.text.apply(lambda x: TextBlob(x).sentiment[0])  # Calculate sentiment scores using TextBlob

    # Iterate over each sentiment score and assign sentiment labels based on thresholds
    for score in sentiment_score:
        if score < -0.1:
            sentiment.append('negative')
        elif score > 0.1:
            sentiment.append('positive')
        else:
            sentiment.append('neutral')

    df['predicted_sentiment'] = sentiment  # Add the 'predicted_sentiment' column to the DataFrame

# Apply the function on tweets to determine sentiment
lexicon_sentiment(tweets_df)


# In[37]:


# Perform crosstabs
display(pd.crosstab(tweets_df['airline_sentiment'], tweets_df['predicted_sentiment'], normalize = 'index'))


# 

# In[39]:


get_ipython().system('pip install scikit-learn')
from sklearn.metrics import classification_report
print(classification_report(tweets_df['airline_sentiment'], tweets_df['predicted_sentiment']))


# In[40]:


accuracy = (np.sum(tweets_df['airline_sentiment'] == tweets_df['predicted_sentiment']))/len(tweets_df['airline_sentiment'])
print(accuracy)


# Conclusion of sentiment analysis using lexicon method: ?????
# 

# Preprocessing II¶
# Selecting data types

# In[41]:


# Define a list of column names representing numeric features
columns_num = tweets_df.select_dtypes(include = ['int','float']).columns
columns_cat = ['airline_extracted']

# Create y and convert it to numeric
y = tweets_df.airline_sentiment
y = y.replace({'negative':1, 'neutral':0, 'positive':2})
y


# In[ ]:


get_ipython().system('pip install scikit-learn')
from sklearn.model_selection import train_test_split

# Split to train and test dataset
X_train, X_test, y_train, y_test = train_test_split(tweets_df, y, random_state = 987, stratify = y)


# Standardize numeric columns

# In[43]:


from sklearn.preprocessing import MinMaxScaler

# Create a StandardScaler object
scaler = MinMaxScaler()

# Fit the scaler on the training data and transform it
X_train[columns_num] = scaler.fit_transform(X_train[columns_num])

# Transform the test data using the fitted scaler
X_test[columns_num] = scaler.transform(X_test[columns_num])
X_train[columns_num]


# Encode categorical columns

# In[44]:


from sklearn.preprocessing import OneHotEncoder

# Create an instance of the OneHotEncoder
encoder = OneHotEncoder()

# Fit the encoder on the categorical column in the training data and transform it
X_train_encoded = encoder.fit_transform(X_train[columns_cat])

# Transform the categorical column in the test data
X_test_encoded = encoder.transform(X_test[columns_cat])

# Drop the column from the dataset
X_test.drop(columns_cat, axis = 1)
X_train.drop(columns_cat, axis = 1)

# Convert encoded data to dataframe
X_train_airline = pd.DataFrame(X_train_encoded.toarray(), columns = encoder.get_feature_names_out(), index = X_train.index)
X_test_airline = pd.DataFrame(X_test_encoded.toarray(), columns = encoder.get_feature_names_out(), index = X_test.index)
X_train_airline


# Impute missing values in the tweets column

# In[45]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values='', strategy='constant', fill_value='missing')
X_train[['text']] = imputer.fit_transform(X_train[['text']])
X_test[['text']] = imputer.transform(X_test[['text']])
X_train['text']



# Sentiment Analysis using ML with Count Vectorizer¶
# Vectorization

# In[46]:


from sklearn.feature_extraction.text import CountVectorizer
#import pandas as pd

vectorizer = CountVectorizer(min_df=0.01, max_df=0.95, ngram_range=(1, 2))


# Assuming X_train and X_test are defined and have a column 'text'

# Vectorization
X_train_vect = vectorizer.fit_transform(X_train['text'])  # Removed double brackets
X_test_vect = vectorizer.transform(X_test['text'])  # Removed double brackets

# Converting to dataframe
X_train_vect_df = pd.DataFrame(X_train_vect.toarray(), columns=vectorizer.get_feature_names_out(), index=X_train.index)
X_test_vect_df = pd.DataFrame(X_test_vect.toarray(), columns=vectorizer.get_feature_names_out(), index=X_test.index)
X_train_vect_df
X_test_vect_df


# Merging all the dataframes¶

# In[47]:


X_train_final = pd.concat([X_train[columns_num],  X_train_vect_df, X_train_airline], axis =1)
X_test_final = pd.concat([X_test[columns_num],  X_test_vect_df, X_test_airline], axis =1)
X_train_final


# Model training and evaluation

# In[48]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


import xgboost as xgb

classifiers = [
    ('Logistic Regression', LogisticRegression()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier()),
    ('Support Vector Machine', SVC()),
    ('K-Nearest Neighbors', KNeighborsClassifier()),
    ('Gradient Boosting', GradientBoostingClassifier()),
    ('Naive Bayes', GaussianNB()),
    ('xgboost',xgb.XGBClassifier())
]


# In[49]:


#import numpy as np

# Iterate through the classifiers
# Check for NaN values


# Find rows with NaN values and display the first 15 of these rows
nan_rows = X_train[X_train.isna().any(axis=1)]

# Handle NaN values - option 1: fill with a value (e.g., 0 or mean)
X_train_final = X_train_final.fillna(0)
X_test_final = X_test_final.fillna(0)
# X_train_final = X_train_final.fillna(X_train_final.mean())

# Handle NaN values - option 2: drop rows with NaNs
# X_train_final = X_train_final.dropna()

for clf_name, clf in classifiers:
     # Fit the classifier on the training data
    clf.fit(X_train_final, y_train)

    # Make predictions on the test data
    y_pred = clf.predict(X_test_final)

    # Calculate the accuracy score
    accuracy = (np.sum(y_test == y_pred))/len(y_test)

    # Print the accuracy for each classifier
    print(f"{clf_name}: Accuracy = {accuracy}")


# Sentiment Analysis using ML with TFIDF Vectorization

# In[51]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

# TF-IDF Vectorization with adjusted min_df and max_df
tfidf = TfidfVectorizer(min_df=0.01, max_df=0.98)

X_train_vect = tfidf.fit_transform(X_train['text'])
X_test_vect = tfidf.transform(X_test['text'])

# Dimensionality Reduction using Truncated SVD
# Adjust n_components based on your dataset and requirements
svd = TruncatedSVD(n_components=110)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

X_train_vect_reduced = lsa.fit_transform(X_train_vect)
X_test_vect_reduced = lsa.transform(X_test_vect)

# Converting reduced data back to DataFrame for easier handling
#feature_names = [f'component_{i}' for i in range(X_train_vect_reduced.shape[1])]
X_train_vect_df = pd.DataFrame(X_train_vect_reduced, columns=tfidf.get_feature_names_out(), index=X_train.index)
X_test_vect_df = pd.DataFrame(X_test_vect_reduced, columns=tfidf.get_feature_names_out(), index=X_test.index)

# Display the transformed DataFrame
print(X_train_vect_df.head())
# Display the transformed DataFrame
print("Vectorized DataFrame:\n", X_train_vect_df.head())


# Merging all the dataframes

# In[52]:


X_train_final = pd.concat([X_train[columns_num],  X_train_vect_df, X_train_airline], axis =1)
X_test_final = pd.concat([X_test[columns_num],  X_test_vect_df, X_test_airline], axis =1)
X_train_final


# Model training and evaluation

# In[53]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


import xgboost as xgb

classifiers = [
    ('Logistic Regression', LogisticRegression()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier()),
    ('Support Vector Machine', SVC()),
    ('K-Nearest Neighbors', KNeighborsClassifier()),
    ('Gradient Boosting', GradientBoostingClassifier()),
    ('Naive Bayes', GaussianNB()),
    ('xgboost',xgb.XGBClassifier())
]


# In[54]:


import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb


# Find rows with NaN values and display the first 15 of these rows
nan_rows = X_train[X_train.isna().any(axis=1)]

# Handle NaN values - option 1: fill with a value (e.g., 0 or mean)
X_train_final = X_train_final.fillna(0)
X_test_final = X_test_final.fillna(0)

# X_train_final = X_train_final.fillna(X_train_final.mean())
# Handle NaN values - option 2: drop rows with NaNs
# X_train_final = X_train_final.dropna()

# Iterate through the classifiers
for clf_name, clf in classifiers:
    # Fit the classifier on the training data
    clf.fit(X_train_final, y_train)

    # Make predictions on the test data
    y_pred = clf.predict(X_test_final)

    # Calculate the accuracy score
    accuracy = (np.sum(y_test ==y_pred))/len(y_test)

    # Print the accuracy for each classifier
    print(f"{clf_name}: Accuracy = {accuracy}")

