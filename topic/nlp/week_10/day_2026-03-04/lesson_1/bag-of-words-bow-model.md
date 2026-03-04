---
title: "Bag of Words (BoW) Model"
date: "2026-03-04"
week: 10
lesson: 1
slug: "bag-of-words-bow-model"
---

# Topic: Bag of Words (BoW) Model

## 1) Formal definition (what is it, and how can we use it?)

The Bag of Words (BoW) model is a simplifying representation used in natural language processing and information retrieval. It represents text as an *unordered set of words*, disregarding grammar and even word order but keeping track of word counts.

**What is it?**

The core idea is to create a vocabulary of all unique words in the corpus (a collection of documents). Each document is then represented by a vector. The length of this vector is equal to the size of the vocabulary. Each element in the vector represents the *frequency* of the corresponding word in the document.

**How can we use it?**

BoW is primarily used for:

*   **Text Classification:**  Categorizing documents based on their content (e.g., spam/not spam, positive/negative sentiment).  The frequency counts of different words act as features that a machine learning classifier (e.g., Naive Bayes, Logistic Regression) can learn from.
*   **Information Retrieval:**  Finding documents relevant to a query. The query is also transformed into a BoW vector, and the similarity between the query vector and the document vectors is computed to rank the documents.  Techniques like cosine similarity are commonly used to measure this similarity.
*   **Topic Modeling:**  While not a direct application, the BoW representation can be a starting point for more sophisticated topic modeling techniques like Latent Dirichlet Allocation (LDA).
*   **Feature Extraction:** Converting textual data into numerical features suitable for machine learning models.

**Limitations:**

*   **Loss of Context:**  The most significant limitation is the disregard for word order and grammar.  "This is good, not bad" and "This is bad, not good" would have very similar BoW representations.
*   **Vocabulary Size:** The size of the vocabulary can become very large, especially with large corpora, leading to high-dimensional and sparse feature vectors.  This can increase computational cost and potentially hurt model performance.
*   **Equal Weighting:**  The model treats all words equally, which may not be desirable.  Common words like "the", "a", and "is" can dominate the frequency counts without contributing much meaning. This is usually addressed with techniques like TF-IDF (Term Frequency-Inverse Document Frequency).
*   **No Semantic Meaning:**  Words with similar meanings are treated as completely different.  For instance, "good" and "excellent" would be considered separate features.

## 2) Application scenario

**Scenario:** Sentiment analysis of customer reviews for an online product.

**Goal:** Automatically classify customer reviews as positive or negative based on the text of the reviews.

**How BoW is used:**

1.  **Data Collection:** Gather a set of customer reviews and label each review as either positive or negative.
2.  **Vocabulary Creation:** Create a vocabulary of all unique words across all the reviews.
3.  **Vectorization:**  For each review, create a BoW vector representing the frequency of each word in the vocabulary within that review.
4.  **Model Training:** Train a classification model (e.g., Naive Bayes, Logistic Regression) using the BoW vectors as features and the sentiment labels (positive/negative) as the target variable.
5.  **Prediction:**  For new, unseen reviews, create a BoW vector and use the trained model to predict the sentiment.

In this scenario, BoW provides a simple and efficient way to convert textual data into numerical data that a machine learning model can understand. While it ignores the nuances of language, it can often achieve reasonable accuracy, especially when combined with other techniques like stop word removal and TF-IDF.

## 3) Python method (if possible)

```python
from sklearn.feature_extraction.text import CountVectorizer

# Sample documents
documents = [
    "This is the first document.",
    "This is the second second document.",
    "And this is the third one.",
    "Is this the first document?"
]

# Create a CountVectorizer object
vectorizer = CountVectorizer()

# Fit the vectorizer to the documents (learn the vocabulary)
vectorizer.fit(documents)

# Transform the documents into BoW vectors
bow_vectors = vectorizer.transform(documents)

# Print the vocabulary
print("Vocabulary:", vectorizer.vocabulary_)

# Print the BoW vectors (sparse matrix format)
print("\nBoW Vectors (Sparse Matrix):")
print(bow_vectors)

# Convert the sparse matrix to a dense array (for easier viewing)
bow_vectors_dense = bow_vectors.toarray()
print("\nBoW Vectors (Dense Array):")
print(bow_vectors_dense)

# Get feature names (words in the vocabulary)
feature_names = vectorizer.get_feature_names_out()
print("\nFeature Names:", feature_names)

# You can then feed these BoW vectors into a machine learning model
```

**Explanation:**

1.  **`CountVectorizer`:**  The `CountVectorizer` class from scikit-learn is used to create the BoW representation.
2.  **`fit(documents)`:** This method learns the vocabulary from the input documents. It identifies all unique words and assigns an index to each word.
3.  **`transform(documents)`:** This method transforms the input documents into BoW vectors. Each vector represents the frequency of each word in the vocabulary within the corresponding document. The output is a sparse matrix, which is an efficient way to store matrices with many zero values.
4.  **`vocabulary_`:** This attribute stores the vocabulary as a dictionary, mapping each word to its index in the feature vector.
5.  **`toarray()`:** This method converts the sparse matrix to a dense NumPy array, which is easier to read and understand (but can consume more memory).
6.  **`get_feature_names_out()`:** This method returns an array of feature names corresponding to the columns in the BoW matrix.
7. The output dense array represents the frequency counts. For example:
    ```
    [[1 0 0 1 1 0 1 0 1]
     [1 0 1 1 1 0 1 2 0]
     [0 1 0 1 1 1 1 0 0]
     [1 0 0 1 1 0 1 0 1]]
    ```
This shows the frequency counts of the terms in the vocabulary, for each document, in the order of the terms in the vocabulary.

## 4) Follow-up question

How can the BoW model be improved to address the limitations of ignoring word importance and handle unseen words during the prediction phase (words that are not in the training vocabulary)? Explain the techniques that can be used for each of those issues separately.