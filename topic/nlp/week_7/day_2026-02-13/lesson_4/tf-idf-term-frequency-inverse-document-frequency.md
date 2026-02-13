---
title: "TF-IDF (Term Frequency-Inverse Document Frequency)"
date: "2026-02-13"
week: 7
lesson: 4
slug: "tf-idf-term-frequency-inverse-document-frequency"
---

# Topic: TF-IDF (Term Frequency-Inverse Document Frequency)

## 1) Formal definition (what is it, and how can we use it?)

TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical measure used to evaluate the importance of a word to a document in a collection of documents (corpus). It is often used in information retrieval and text mining as a weighting factor.  It reflects how important a word is to a document in a corpus.

TF-IDF has two main components:

*   **Term Frequency (TF):**  Measures how frequently a term occurs in a document. The basic intuition is that a term appearing more often in a document is more important to that document. It is calculated as:

    `TF(t, d) = (Number of times term t appears in document d) / (Total number of terms in document d)`

*   **Inverse Document Frequency (IDF):** Measures how rare a term is across the entire corpus. The idea is that terms that appear in many documents are less informative than terms that appear in only a few. It is calculated as:

    `IDF(t, D) = log(Total number of documents in the corpus / Number of documents containing term t)`

    Where `D` represents the whole corpus. The logarithm helps to dampen the effect of very common words. Some implementations add 1 to the denominator inside the logarithm to avoid division by zero.

The TF-IDF score is then calculated by multiplying the TF and IDF values:

`TF-IDF(t, d, D) = TF(t, d) * IDF(t, D)`

We can use TF-IDF to:

*   **Rank documents:**  Given a search query, TF-IDF can be used to rank documents based on their relevance to the query.  Documents with higher TF-IDF scores for the query terms are considered more relevant.
*   **Feature extraction:** TF-IDF scores can be used as features for machine learning models in text classification, clustering, and other NLP tasks.  Each document can be represented as a vector of TF-IDF scores, where each element in the vector corresponds to a term in the vocabulary.
*   **Keyword extraction:** Identify the most important keywords in a document. Terms with high TF-IDF scores are likely to be significant for that document.

## 2) Application scenario

Imagine you have a corpus of news articles about different topics like sports, politics, and technology. You want to build a search engine that allows users to search for articles based on keywords.

Using TF-IDF, you can:

1.  **Calculate TF-IDF scores for each term in each article.** For example, the term "basketball" might have a high TF-IDF score in sports articles and a low TF-IDF score in politics articles.
2.  **When a user searches for "basketball," your search engine can calculate the TF-IDF score of "basketball" in each article.**
3.  **Rank the articles based on their TF-IDF scores for the term "basketball."** Articles with higher scores are displayed first, as they are more likely to be relevant to the user's query.

Another scenario is text classification. Suppose you want to classify customer reviews as positive or negative. You can calculate TF-IDF scores for each word in each review and use these scores as features for a machine learning classifier (e.g., Naive Bayes, Support Vector Machine). This allows the model to learn which words are most indicative of positive or negative sentiment.

## 3) Python method (if possible)

Scikit-learn provides the `TfidfVectorizer` class to easily calculate TF-IDF scores.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Example documents
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

# Create a TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Fit and transform the documents
tfidf_matrix = vectorizer.fit_transform(documents)

# The resulting tfidf_matrix is a sparse matrix
# You can convert it to a dense array for easier inspection (but it can consume a lot of memory for large datasets)
tfidf_array = tfidf_matrix.toarray()

# Get the feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Print the TF-IDF matrix
print("Feature Names:", feature_names)
print("\nTF-IDF Matrix:")
for i, doc in enumerate(documents):
    print(f"Document {i+1}: {doc}")
    for j, term in enumerate(feature_names):
        print(f"  {term}: {tfidf_array[i][j]:.4f}", end="  ")
    print("\n")

# Access individual TF-IDF scores:
# Example: TF-IDF score of the word "document" in the first document.
try:
    document_index = 0
    word = "document"
    word_index = list(feature_names).index(word)
    tfidf_score = tfidf_array[document_index][word_index]
    print(f"TF-IDF score of '{word}' in document {document_index+1}: {tfidf_score:.4f}")
except ValueError:
    print(f"The word '{word}' is not in the vocabulary.")
```

**Explanation:**

1.  **`TfidfVectorizer()`:** Creates a TF-IDF vectorizer object.  It has several parameters to customize its behavior, such as `stop_words` (to remove common words like "the", "a", "is"), `ngram_range` (to consider phrases instead of single words), `max_df` and `min_df` (to filter words based on document frequency), and `norm` (to normalize the vectors).
2.  **`fit_transform(documents)`:**  This method first *fits* the vectorizer to the documents, learning the vocabulary (the set of unique words). Then, it *transforms* the documents into a TF-IDF matrix.  Each row represents a document, and each column represents a term in the vocabulary. The values in the matrix are the TF-IDF scores.
3.  **`get_feature_names_out()`:** Returns an array mapping feature index to feature name.
4.  The code then prints the TF-IDF matrix and demonstrates how to access the TF-IDF score of a specific word in a specific document.
5. **Handling `ValueError`**: The `try-except` block handles the scenario when the word is not present in the vocabulary learned by the vectorizer.

## 4) Follow-up question

How does TF-IDF perform with large datasets and very long documents? Are there any limitations or alternative approaches that are more suitable in such scenarios?