---
title: "Feature Engineering for Text Data"
date: "2026-03-04"
week: 10
lesson: 5
slug: "feature-engineering-for-text-data"
---

# Topic: Feature Engineering for Text Data

## 1) Formal definition (what is it, and how can we use it?)

Feature engineering for text data is the process of transforming raw text into numerical features that machine learning models can understand and utilize.  Raw text is inherently symbolic and needs to be converted into a format amenable to mathematical operations. These features can represent different aspects of the text, such as the frequency of words, the presence of specific patterns, or even semantic information.

We use feature engineering to:

*   **Improve Model Performance:** By extracting relevant information from the text, we provide the model with more meaningful input, leading to better accuracy and generalization.
*   **Reduce Dimensionality:** Text data is often high-dimensional (many unique words). Feature engineering can condense this information into a smaller, more manageable set of features.
*   **Capture Specific Linguistic Properties:** Engineered features can explicitly represent linguistic aspects like sentiment, topic, or part-of-speech, which might not be directly captured by simpler methods.
*   **Adapt to Specific Model Requirements:** Different machine learning models might benefit from different types of features. Feature engineering allows us to tailor the input data to the specific needs of the model.

Common feature engineering techniques for text include:

*   **Bag of Words (BoW):**  Represents text as the frequency of words within a document, ignoring grammar and word order.
*   **TF-IDF (Term Frequency-Inverse Document Frequency):** Weights words based on their frequency within a document and their inverse frequency across all documents, giving more importance to less common words.
*   **N-grams:** Sequences of *n* words (e.g., bigrams: "machine learning").  Captures some word order information.
*   **Word Embeddings (Word2Vec, GloVe, FastText):** Dense vector representations of words, capturing semantic relationships.
*   **Character-level features:** Counts of characters, specific characters, or character n-grams.
*   **Lexical Features:** Number of words, average word length, number of stop words, punctuation count.
*   **Sentiment Scores:**  Polarity and subjectivity scores derived from sentiment analysis tools.
*   **Part-of-Speech (POS) tagging:** Identifying the grammatical role of each word (noun, verb, adjective, etc.) and creating features based on the frequency of different POS tags.
*   **Topic Modeling (LDA, NMF):**  Assigning documents to different topics and using topic probabilities as features.

## 2) Application scenario

**Scenario:** Spam email detection.

**How feature engineering helps:**

Raw email text is unstructured and difficult to directly input into a classifier. By applying feature engineering, we can create a structured numerical representation that a machine learning model can use to distinguish between spam and legitimate emails.

**Example Features:**

*   **TF-IDF scores of words:**  Words like "discount," "free," and "urgent" may have high TF-IDF scores in spam emails compared to legitimate emails.
*   **Presence of certain keywords:**  A binary feature indicating whether an email contains keywords commonly found in spam, such as "Viagra," "lottery," or "inheritance."
*   **Number of exclamation marks:** Spam emails often use excessive exclamation marks to create a sense of urgency.
*   **Average sentence length:**  Spam emails might have shorter, more concise sentences.
*   **Ratio of capital letters:** Spam emails might use a higher proportion of capital letters in the subject line to grab attention.
*   **URL count:** Spam emails often contain numerous URLs, especially to suspicious domains.
*   **Sentiment score:**  Spam emails may use more persuasive or aggressive language.

By combining these features, we can create a robust representation of the email text that allows a machine learning model (e.g., a logistic regression or support vector machine) to accurately classify emails as spam or not spam.

## 3) Python method (if possible)

Here's an example using scikit-learn to implement TF-IDF and create features from a list of sentences:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample text data
corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Learn vocabulary and transform documents to feature vectors
tfidf_matrix = vectorizer.fit_transform(corpus)

# Print the TF-IDF matrix (sparse representation)
print(tfidf_matrix)

# Print the vocabulary (mapping of words to column indices)
print(vectorizer.vocabulary_)

# Convert to a dense array for easier viewing (optional)
tfidf_array = tfidf_matrix.toarray()
print(tfidf_array)

# Access feature names (words)
feature_names = vectorizer.get_feature_names_out()
print(feature_names)

# Access the TF-IDF score for a specific document and word
document_index = 0  # First document
word = "first"
word_index = vectorizer.vocabulary_[word]
tfidf_score = tfidf_array[document_index, word_index]
print(f"TF-IDF score for '{word}' in document {document_index+1}: {tfidf_score}")

```

This code demonstrates how to use `TfidfVectorizer` to convert text into a TF-IDF matrix, which can then be used as input features for a machine learning model.  The code also shows how to access the vocabulary, convert the matrix to a dense array (for easier human readability), retrieve the feature names (words), and access the TF-IDF score for a specific word in a specific document.

## 4) Follow-up question

How do I choose the *right* feature engineering techniques for a specific NLP task?  What are some common pitfalls to avoid when engineering features for text data, and how can I prevent them?