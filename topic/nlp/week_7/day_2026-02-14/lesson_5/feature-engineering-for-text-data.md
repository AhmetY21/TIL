---
title: "Feature Engineering for Text Data"
date: "2026-02-14"
week: 7
lesson: 5
slug: "feature-engineering-for-text-data"
---

# Topic: Feature Engineering for Text Data

## 1) Formal definition (what is it, and how can we use it?)

Feature engineering for text data is the process of transforming raw text into numerical features that machine learning models can understand and use. Raw text, in its unprocessed form, is not directly usable by most machine learning algorithms, which typically require numerical input. Feature engineering bridges this gap by extracting meaningful information from text and representing it as numerical data.

We can use feature engineering for text data in a variety of NLP tasks, including:

*   **Text classification:** Identifying the category or topic of a document (e.g., spam detection, sentiment analysis).
*   **Text clustering:** Grouping similar documents together.
*   **Information retrieval:** Finding relevant documents based on a user's query.
*   **Machine translation:** Converting text from one language to another.
*   **Question answering:** Providing answers to questions based on a given text.

Essentially, feature engineering helps to capture the semantic meaning, syntactic structure, and stylistic characteristics of the text in a way that is understandable by a machine learning model. Common approaches include:

*   **Bag-of-Words (BoW):** Represents text as the collection of its words, disregarding grammar and word order, and counting the frequency of each word.
*   **TF-IDF (Term Frequency-Inverse Document Frequency):** Weights words based on their frequency in a document and their inverse frequency across the entire corpus, giving higher weights to words that are important in a specific document but not common across all documents.
*   **N-grams:** Sequences of *n* consecutive words, which capture some information about word order.
*   **Word Embeddings:** Represent words as dense, low-dimensional vectors that capture semantic relationships between words (e.g., Word2Vec, GloVe, FastText).
*   **Character-level features:** Using character n-grams or character frequencies to extract features, useful for handling noisy text or dealing with different languages.
*   **Syntactic features:** Part-of-speech tagging, dependency parsing, and other syntactic analyses can be used to extract features about the grammatical structure of the text.

## 2) Application scenario

Consider a sentiment analysis task where we want to predict whether a movie review is positive or negative. Raw text alone cannot be fed into a machine learning model.  We need to transform the text into numerical features.

Using Bag-of-Words, we could create a vocabulary of all the unique words in our corpus of movie reviews. Then, for each review, we would create a vector representing the frequency of each word in the vocabulary within that review.  A review like "This movie was great and amazing!" would result in a vector where the counts for "great" and "amazing" would be 1, and the counts for all other words would be based on their presence in the review. This vector then becomes the input to a classifier.

Alternatively, using TF-IDF, we could assign higher weights to words like "amazing" and "terrible" (assuming they are rare across all movie reviews but common in highly positive or negative reviews, respectively) compared to common words like "the" or "and."

Word embeddings offer an even more sophisticated approach.  Instead of counting word occurrences, we can represent each word with a pre-trained vector capturing its semantic meaning. We could then average the word embeddings of all words in a review to obtain a single vector representing the overall sentiment of the review.

## 3) Python method (if possible)

Here's an example using scikit-learn to perform TF-IDF feature extraction:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]

# Create a TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Fit the vectorizer to the documents and transform them into a TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(documents)

# Print the TF-IDF matrix
print(tfidf_matrix.toarray())

# Print the vocabulary
print(vectorizer.vocabulary_)
```

This code snippet first defines a list of documents.  Then, it creates a `TfidfVectorizer` object from scikit-learn. The `fit_transform` method learns the vocabulary from the documents and transforms them into a TF-IDF matrix.  `tfidf_matrix.toarray()` converts the sparse matrix to a dense numpy array for easier viewing. `vectorizer.vocabulary_` provides a dictionary mapping each word to its index in the TF-IDF matrix.

For word embeddings, libraries like Gensim or Hugging Face Transformers are commonly used.

## 4) Follow-up question

How do you choose the *best* feature engineering technique for a particular text analysis task? What are some factors that influence this choice?