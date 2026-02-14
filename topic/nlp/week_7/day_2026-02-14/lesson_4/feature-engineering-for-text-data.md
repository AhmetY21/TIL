---
title: "Feature Engineering for Text Data"
date: "2026-02-14"
week: 7
lesson: 4
slug: "feature-engineering-for-text-data"
---

# Topic: Feature Engineering for Text Data

## 1) Formal definition (what is it, and how can we use it?)

Feature engineering for text data is the process of transforming raw text into numerical features that can be used by machine learning algorithms. Since most machine learning models require numerical input, text must be converted into a numerical representation that captures its semantic meaning or statistical properties.  This involves extracting relevant characteristics from text data, selecting the most informative features, and transforming them into a suitable format for model training.

How can we use it?

*   **Improved Model Accuracy:** By selecting and engineering features that are most relevant to the task, we can significantly improve the accuracy and performance of our models.
*   **Reduced Model Complexity:** Feature selection can help reduce the dimensionality of the data, leading to simpler and faster models.
*   **Enhanced Interpretability:** Carefully engineered features can provide insights into the underlying patterns and relationships in the text data, making the model more interpretable.
*   **Task-Specific Optimization:** Feature engineering allows us to tailor the features to the specific requirements of the task at hand, whether it's sentiment analysis, topic classification, or machine translation.

Common feature engineering techniques include:

*   **Bag of Words (BoW):** Represents text as a collection of individual words, ignoring grammar and word order. The frequency of each word is often used as a feature.
*   **Term Frequency-Inverse Document Frequency (TF-IDF):**  Weights words based on their frequency in a document and their rarity across the entire corpus. This helps to downweight common words (like "the", "a", "is") that are less informative.
*   **N-grams:** Sequences of N consecutive words in a text. This captures some context information by considering word combinations.
*   **Word Embeddings (Word2Vec, GloVe, FastText):**  Represent words as dense vectors in a high-dimensional space, capturing semantic relationships between words. Words with similar meanings are located closer together in the vector space.
*   **Character-level Features:** Features based on individual characters or character sequences, such as character n-grams.  Useful for tasks like language identification or spelling correction.
*   **Syntactic Features:** Features derived from the grammatical structure of the text, such as part-of-speech (POS) tags, dependency parsing, and phrase structure.
*   **Sentiment Scores:** Numerical values indicating the sentiment (positive, negative, or neutral) expressed in the text.
*   **Readability Scores:** Metrics that quantify the readability of the text, such as the Flesch Reading Ease score.
*   **Metadata Features:** Features extracted from the metadata associated with the text, such as the author, publication date, or source.

## 2) Application scenario

**Scenario:** Sentiment Analysis of Customer Reviews

Imagine an e-commerce company wants to automatically analyze customer reviews to identify which products are receiving positive and negative feedback. The company collects thousands of reviews daily and needs a way to quickly gauge customer sentiment.

**Feature Engineering:**

1.  **Text Preprocessing:** Clean the text by removing irrelevant characters, HTML tags, and converting text to lowercase. Apply stemming or lemmatization to reduce words to their base form.
2.  **TF-IDF:** Use TF-IDF to create features that represent the importance of words in each review relative to the entire dataset of reviews.  This will highlight words that are particularly indicative of positive or negative sentiment.
3.  **Sentiment Lexicon Scores:**  Calculate sentiment scores for each review using a pre-built sentiment lexicon (e.g., VADER sentiment analyzer). This provides a numerical indication of the review's overall sentiment.
4.  **N-grams:** Include bigrams (sequences of two words) to capture phrases that express sentiment, such as "highly recommended" or "very disappointing".
5.  **Word Embeddings:** Use pre-trained word embeddings (e.g., GloVe or Word2Vec) to represent words and phrases as vectors. This allows the model to understand semantic relationships between words and capture nuances in sentiment.

**Model Training:**

A machine learning model (e.g., logistic regression, support vector machine, or a neural network) can then be trained on these engineered features to predict the sentiment of each review. The model can then be used to automatically classify new reviews as positive, negative, or neutral.

## 3) Python method (if possible)
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon') # Download the lexicon data

# Sample reviews
reviews = [
    "This product is amazing! I highly recommend it.",
    "The product was okay, but nothing special.",
    "I am extremely disappointed with this purchase. It's terrible!",
    "Great value for the price. I'm very happy with it.",
    "It broke after only one use. Complete waste of money."
]

# 1. TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2)) #Include unigrams and bigrams
tfidf_matrix = tfidf_vectorizer.fit_transform(reviews)
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out() #Correct method to get feature names
print("TF-IDF Feature Names:", tfidf_feature_names)
print("TF-IDF Matrix Shape:", tfidf_matrix.shape)


# 2. Sentiment Analysis with VADER
sid = SentimentIntensityAnalyzer()
sentiment_scores = [sid.polarity_scores(review)["compound"] for review in reviews] #Extract the compound score
print("Sentiment Scores:", sentiment_scores)


#Combining TF-IDF and Sentiment Feature (for demonstration purposes - often requires scaling)
#In reality this would be incorporated into a pipeline. This is a simplistic example.

import numpy as np
from scipy.sparse import hstack

#Convert sentiment scores to a sparse matrix
sentiment_matrix = np.array(sentiment_scores).reshape(-1, 1) #Needs to be an array not a list
sentiment_sparse = np.array(sentiment_matrix) #convert to numpy array to enable hstack

#Stack the TF-IDF and sentiment features
combined_features = hstack((tfidf_matrix, sentiment_sparse))

print("Combined Feature Shape:", combined_features.shape)
```

## 4) Follow-up question

How can we handle out-of-vocabulary (OOV) words when using pre-trained word embeddings, and what are the trade-offs of different approaches? For example, what are the advantages and disadvantages of using random initialization versus learning new embeddings for OOV words?