# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import string

import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import streamlit as st

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Title and description
st.title('Sentiment Analyzer App')
st.write('Welcome to my sentiment analysis app!')

# Data Loading
kolom_name = ['tweet id', 'entity', 'sentiment', 'tweet content']
df = pd.read_csv("twitter_validation.csv", names=kolom_name, header=None)

# Data Cleaning
st.subheader('Data Cleaning')

# Before cleaning
st.write('**Before Cleaning:**')
st.write(df.isnull().sum())

# Drop NaN values
df = df.dropna()

# After cleaning
st.write('**After Cleaning:**')
st.write(df.isnull().sum())

# Remove Duplicates
st.subheader('Remove Duplicates')

# Before removing duplicates
st.write('**Before Removing Duplicates:**')
st.write(df.duplicated().sum())

# Drop duplicates
df = df.drop_duplicates()

# After removing duplicates
st.write('**After Removing Duplicates:**')
st.write(df.duplicated().sum())

# Data Analysis
st.subheader('Data Analysis')

# Outliers in tweet content length
df['tweet content length'] = df['tweet content'].map(lambda x: len(x.split(" ")))
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x="tweet content length", data=df, ax=ax)
ax.set_title('Distribution of Tweet Content Length')
st.pyplot(fig)

# Remove outliers
lower_quartile = np.percentile(df['tweet content length'], 25)
upper_quartile = np.percentile(df['tweet content length'], 75)
interquartile_range = upper_quartile - lower_quartile
lower_bound = lower_quartile - 1.5 * interquartile_range
upper_bound = upper_quartile + 1.5 * interquartile_range

remove_df = df[(df['tweet content length'] >= lower_bound) & (df['tweet content length'] <= upper_bound)]

# Visualize distribution after removing outliers
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
sns.boxplot(x="tweet content length", data=df, ax=ax[0])
ax[0].set_title('Before Removing Outliers')
sns.boxplot(x="tweet content length", data=remove_df, ax=ax[1], color='green')
ax[1].set_title('After Removing Outliers')
fig.tight_layout()
st.pyplot(fig)

# Sentiment Distribution
st.subheader('Sentiment Distribution')
sns.countplot(x="sentiment", data=remove_df, palette='bright', order=["Positive", "Negative", "Neutral", "Irrelevant"])
st.pyplot()

# Word Clouds
st.subheader('Word Clouds')

sentiment_list = ['Positive', 'Neutral', 'Negative', 'Irrelevant']
colormap_list = ['YlGn_r', 'Blues_r', 'Reds_r', 'copper_r']
stopwords_set = set(STOPWORDS)

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 9))

for sentiment, (row, col), colormap in zip(sentiment_list, [(0, 0), (0, 1), (1, 0), (1, 1)], colormap_list):
    text = " ".join(content for content in remove_df[remove_df['sentiment'] == sentiment]['tweet content'])
    wc = WordCloud(colormap=colormap, stopwords=stopwords_set, width=800, height=500).generate(text)
    ax[row, col].imshow(wc, interpolation='bilinear')
    ax[row, col].set_title(sentiment + " Wordcloud", fontsize=14)
    ax[row, col].axis('off')

fig.tight_layout()
st.pyplot(fig)

# Frequency of Entity
st.subheader('Frequency of Entity')

entity_frequency = remove_df.groupby(['sentiment', 'entity']).size().reset_index(name='frequency')
plt.figure(figsize=(20, 6))
sns.barplot(data=entity_frequency, x='entity', y='frequency', hue='sentiment')
plt.xticks(rotation=90)
plt.title('Frequency of Entity')
st.pyplot()

# Text Preprocessing Function
st.subheader('Text Preprocessing Function')

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def text_preprocessing(text):
    text_tokenize = word_tokenize(text)
    entity = text_tokenize[0]
    text_content = text_tokenize[1:]
    text_pos = pos_tag(text_content)
    remove_words = set(list(string.punctuation) + stopwords.words('english'))
    text_remove = [(word, pos) for (word, pos) in text_pos if word.lower() not in remove_words]
    word_lem = WordNetLemmatizer()
    text_lem = [(word_lem.lemmatize(word, pos=get_wordnet_pos(pos)) if get_wordnet_pos(pos) else word_lem.lemmatize(word), pos) for (word, pos) in text_remove]
    text_lem.append((entity,))
    return text_lem

example_text = "Overwatch Overwatch is a wonderful game, even after so many years."
st.write("Original sentence:", example_text)
st.write("After text preprocessing:", text_preprocessing(example_text))

# Data Splitting
st.subheader('Data Splitting')
x_train = df['entity'] + " " + df['tweet content']
y_train = df['sentiment']
x_test = df['entity'] + " " + df['tweet content']
y_test = df['sentiment']

# Text Preprocessing Pipeline
st.subheader('Text Preprocessing Pipeline')

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_preprocessing)),
    ('tfidf', TfidfTransformer())
])

x_train_processed = pipeline.fit_transform(x_train)
x_test_processed = pipeline.transform(x_test)

# Models
st.subheader('Models')
classifier_used = [
    LogisticRegression(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
]

classifier_accuracy = []

for classifier in classifier_used:
    fit = classifier.fit(x_train_processed, y_train)
    predict = fit.predict(x_test_processed)
    trainset_predict = fit.predict(x_train_processed)
    accuracy = accuracy_score(predict, y_test)
    trainset_accuracy = accuracy_score(trainset_predict, y_train)
    classifier_accuracy.append([classifier.__class__.__name__, accuracy, trainset_accuracy])

classifier_result = pd.DataFrame(classifier_accuracy, columns=["Classifier", "Accuracy", "Accuracy on Trainset"]).sort_values(by='Accuracy', ascending=False)
st.write(classifier_result)