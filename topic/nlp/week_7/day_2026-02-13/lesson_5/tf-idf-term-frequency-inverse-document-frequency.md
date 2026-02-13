---
title: "TF-IDF (Term Frequency-Inverse Document Frequency)"
date: "2026-02-13"
week: 7
lesson: 5
slug: "tf-idf-term-frequency-inverse-document-frequency"
---

# Topic: TF-IDF (Term Frequency-Inverse Document Frequency)

## 1) Formal definition (what is it, and how can we use it?)

TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic that reflects how important a word is to a document in a collection or corpus. It is often used in information retrieval and text mining as a weighting factor in search engines, text summarization, and document classification.

Essentially, TF-IDF attempts to quantify the importance of a term within a document relative to the entire corpus. It works by:

*   **Term Frequency (TF):** Measures how frequently a term occurs in a given document. Higher TF values indicate that the term appears more often in the document.  A common formula for TF is:

    `TF(t, d) = (Number of times term t appears in document d) / (Total number of terms in document d)`

*   **Inverse Document Frequency (IDF):** Measures how important a term is across the entire corpus. It diminishes the weight of terms that occur very frequently in the corpus and increases the weight of terms that occur rarely. This helps to highlight terms that are more discriminative for specific documents. A common formula for IDF is:

    `IDF(t, D) = log (Total number of documents in corpus D / Number of documents in corpus D containing term t)`

    Where log is usually the base-10 logarithm.  We typically add 1 to the denominator inside the log to avoid division by zero (if a term never appears in any document) and sometimes also add 1 to the entire result to prevent IDF from being zero.

The TF-IDF score for a term *t* in a document *d* within corpus *D* is calculated by multiplying the term frequency (TF) and inverse document frequency (IDF):

`TF-IDF(t, d, D) = TF(t, d) * IDF(t, D)`

**How can we use it?**

*   **Information Retrieval (Search Engines):**  TF-IDF is a core component in many search engines.  When a user submits a query, the search engine can calculate the TF-IDF score of the query terms in each document in its index.  Documents with higher TF-IDF scores for the query terms are ranked higher in the search results.
*   **Text Summarization:** Identify important sentences in a document by calculating TF-IDF scores for the words in each sentence. Sentences with higher average TF-IDF scores may be included in a summary.
*   **Document Classification:**  Use TF-IDF scores to represent documents as vectors.  These vectors can then be used as input features for machine learning classifiers to categorize documents into different classes.
*   **Keyword Extraction:** Extract the most relevant keywords from a document by selecting terms with the highest TF-IDF scores.
*   **Recommendation Systems:**  Recommend documents to users based on the similarity of their TF-IDF vectors.

## 2) Application scenario

Imagine you have a collection of three documents:

*   Document 1: "The cat sat on the mat."
*   Document 2: "The dog sat on the rug."
*   Document 3: "The cat and dog played."

You want to find the most relevant document for the query "cat".

1.  **Calculate TF-IDF scores for the term "cat" in each document:**

    *   Document 1:
        *   TF("cat", Document 1) = 1/6 (1 "cat" out of 6 total words)
        *   IDF("cat", Corpus) = log(3/2) (3 total documents, 2 contain "cat") ≈ 0.176
        *   TF-IDF("cat", Document 1, Corpus) ≈ 0.029
    *   Document 2:
        *   TF("cat", Document 2) = 0/6 = 0
        *   TF-IDF("cat", Document 2, Corpus) = 0
    *   Document 3:
        *   TF("cat", Document 3) = 1/5 (1 "cat" out of 5 total words)
        *   IDF("cat", Corpus) = log(3/2) ≈ 0.176
        *   TF-IDF("cat", Document 3, Corpus) ≈ 0.035

2.  **Rank the documents based on TF-IDF scores:**

    Document 3 has the highest TF-IDF score (0.035), followed by Document 1 (0.029), and Document 2 (0). Therefore, Document 3 is considered the most relevant document for the query "cat".

In this simple example, TF-IDF helps to identify that Document 3, while also mentioning "dog", is more relevant to the single word query "cat" because "cat" has a greater relative frequency within that document as compared to Document 1. Document 2 which does not contain the word, is not considered.

## 3) Python method (if possible)

Scikit-learn provides a `TfidfVectorizer` class that simplifies the calculation of TF-IDF scores.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "The cat sat on the mat.",
    "The dog sat on the rug.",
    "The cat and dog played."
]

# Create a TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Fit and transform the documents
tfidf_matrix = vectorizer.fit_transform(documents)

# Get the feature names (terms)
feature_names = vectorizer.get_feature_names_out()

# Convert the TF-IDF matrix to a dense array
tfidf_array = tfidf_matrix.toarray()

# Print the TF-IDF scores for each term in each document
for i, doc in enumerate(documents):
    print(f"Document {i+1}: {doc}")
    for j, term in enumerate(feature_names):
        print(f"  {term}: {tfidf_array[i][j]:.4f}")  #Formatting output to 4 decimal places

# Example: Get the TF-IDF score for the term "cat" in the first document
term_index = feature_names.tolist().index("cat")
tfidf_score = tfidf_array[0][term_index]
print(f"\nTF-IDF score for 'cat' in Document 1: {tfidf_score:.4f}")
```

This code:

1.  **Imports `TfidfVectorizer`:** From the `sklearn.feature_extraction.text` module.
2.  **Defines the documents:** As a list of strings.
3.  **Creates a `TfidfVectorizer` object:**  This object will handle the TF-IDF calculation. By default, it performs lowercasing, removes punctuation, and tokenizes the text.
4.  **Fits and transforms the documents:**  `fit_transform()` learns the vocabulary from the documents and calculates the TF-IDF matrix. The result is a sparse matrix.
5.  **Gets the feature names:** `get_feature_names_out()` returns a list of the terms (words) that were extracted from the documents.
6.  **Converts the matrix to a dense array:**  The `toarray()` method converts the sparse TF-IDF matrix into a dense NumPy array. This allows for easier access to individual TF-IDF scores.
7.  **Prints the TF-IDF scores:**  Iterates through the documents and terms, printing the TF-IDF score for each term in each document.
8.  **Demonstrates how to retrieve a specific TF-IDF score:** Locates the index of "cat" and uses that to print the value for document 1.

## 4) Follow-up question

How do different normalization techniques (e.g., L1, L2) applied to the TF-IDF vectors affect the performance of a document classification model?  Why might one normalization technique be more suitable than another for a particular dataset or problem?