---
title: "Bag of Words (BoW) Model"
date: "2026-02-13"
week: 7
lesson: 2
slug: "bag-of-words-bow-model"
---

# Topic: Bag of Words (BoW) Model

## 1) Formal definition (what is it, and how can we use it?)

The Bag of Words (BoW) model is a text representation technique used in Natural Language Processing and Information Retrieval. It simplifies the text by disregarding grammar and word order, focusing only on the *frequency* of words within a document.

**What is it?**

*   It's a simplifying representation of text.
*   It transforms text into a set of individual words (tokens).
*   It counts the occurrences of each word in a document (or corpus).
*   It creates a vocabulary of all unique words across all documents.
*   It represents each document as a vector (or dictionary) where each element corresponds to the count of a particular word from the vocabulary in that document.

**How can we use it?**

BoW can be used for several NLP tasks, including:

*   **Text Classification:** Categorizing documents (e.g., spam/not spam).  The word frequencies can be used as features for a classifier like Naive Bayes or Logistic Regression.
*   **Sentiment Analysis:** Determining the emotional tone of text (e.g., positive, negative, neutral). The frequency of positive and negative words can be used as indicators.
*   **Information Retrieval:** Ranking documents based on their relevance to a search query. Documents with higher frequencies of the query terms are considered more relevant.
*   **Topic Modeling:**  Discovering underlying themes in a collection of documents. While more sophisticated topic modeling techniques exist (e.g., LDA), BoW can be a preprocessing step.

The key idea is that the presence and frequency of certain words can be indicative of the topic, sentiment, or category of the text, regardless of the word order.

## 2) Application scenario

Imagine we want to build a spam filter. We have a dataset of emails labeled as "spam" or "not spam" (ham).  We can use the Bag of Words model to create a feature vector for each email.

1.  **Preprocessing:** Remove punctuation, convert to lowercase.
2.  **Tokenization:** Split each email into individual words (tokens).
3.  **Vocabulary Creation:** Build a vocabulary of all unique words across all emails.
4.  **Feature Vector Creation:**  For each email, create a vector where each element corresponds to a word in the vocabulary, and the value is the number of times that word appears in the email.

For example:

Email 1 (Spam): "Free money click here urgent offer"
Email 2 (Ham): "Meeting tomorrow please confirm"

Vocabulary: `['free', 'money', 'click', 'here', 'urgent', 'offer', 'meeting', 'tomorrow', 'please', 'confirm']`

BoW representation:

Email 1: `[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]`
Email 2: `[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]`

Now, we can train a classifier (e.g., Naive Bayes) using these BoW feature vectors and the corresponding labels (spam/ham). The classifier learns which words are most indicative of spam and ham. When a new email arrives, we create its BoW representation and the classifier predicts whether it is spam or not.

## 3) Python method (if possible)

We can use scikit-learn's `CountVectorizer` to create a Bag of Words representation:

```python
from sklearn.feature_extraction.text import CountVectorizer

# Sample documents
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]

# Create a CountVectorizer object
vectorizer = CountVectorizer()

# Fit the vectorizer to the documents and transform them into a BoW representation
X = vectorizer.fit_transform(documents)

# Get the vocabulary
vocabulary = vectorizer.get_feature_names_out()
print("Vocabulary:", vocabulary)

# Get the BoW representation as a sparse matrix
print("BoW representation (sparse matrix):\n", X)

# Convert to a dense array (for easier viewing)
X_dense = X.toarray()
print("BoW representation (dense array):\n", X_dense)

# Print the shape of the BoW representation
print("Shape of BoW representation:", X_dense.shape)

# Access the count of a specific word in a specific document
# For example, count of the word "document" in the first document
document_index = 0
word_index = vocabulary.tolist().index("document") #Find index of word
count = X_dense[document_index, word_index]
print(f"Count of 'document' in document {document_index+1}: {count}")
```

This code first creates a `CountVectorizer` object.  Then, `fit_transform` learns the vocabulary from the documents and transforms each document into a BoW representation (a sparse matrix, which is efficient for large vocabularies).  The `get_feature_names_out()` method returns the vocabulary. Finally, the code converts the sparse matrix to a dense array for easier visualization.  The shape of the BoW representation indicates the number of documents (rows) and the size of the vocabulary (columns).
## 4) Follow-up question

The Bag of Words model doesn't consider word order and treats all words equally.  How can we improve upon the BoW model to capture more semantic information and context from the text?  Specifically, what are some other common techniques used in NLP for text representation that address the limitations of BoW?