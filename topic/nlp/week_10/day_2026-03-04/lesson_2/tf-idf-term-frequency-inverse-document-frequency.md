---
title: "TF-IDF (Term Frequency-Inverse Document Frequency)"
date: "2026-03-04"
week: 10
lesson: 2
slug: "tf-idf-term-frequency-inverse-document-frequency"
---

# Topic: TF-IDF (Term Frequency-Inverse Document Frequency)

## 1) Formal definition (what is it, and how can we use it?)

TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. It is often used as a weighting factor in information retrieval and text mining. The TF-IDF value increases proportionally to the number of times a word appears in the document and is offset by the number of documents in the corpus that contain the word. This helps to adjust for the fact that some words appear more frequently in general.

In essence, TF-IDF aims to find words that are both frequent within a specific document (high Term Frequency) and rare across the entire corpus (high Inverse Document Frequency).

Here's a breakdown:

*   **Term Frequency (TF):**  Measures how frequently a term occurs in a document. Several ways to calculate it exist, a common one being:

    `TF(t, d) = (Number of times term t appears in document d) / (Total number of terms in document d)`

*   **Inverse Document Frequency (IDF):** Measures how rare a term is across the corpus. The fewer documents a term appears in, the higher its IDF score. A common formula is:

    `IDF(t, D) = log(Total number of documents in the corpus D / Number of documents containing term t)`

*   **TF-IDF:** The product of TF and IDF:

    `TF-IDF(t, d, D) = TF(t, d) * IDF(t, D)`

**How we can use it:**

*   **Keyword Extraction:** Identify the most important keywords in a document based on their TF-IDF scores.
*   **Document Similarity:** Calculate the similarity between documents by comparing their TF-IDF vectors.  Cosine similarity is often used on these vectors.
*   **Information Retrieval:** Rank search results based on how relevant the terms in the query are to the documents in the corpus (based on TF-IDF scores).
*   **Text Classification:** Use TF-IDF scores as features for training machine learning models to classify documents.

## 2) Application scenario

Imagine you have a collection of news articles about different topics (e.g., sports, politics, technology). You want to build a search engine that allows users to find articles relevant to their queries.

Using TF-IDF:

1.  **Indexing:** You would first compute the TF-IDF scores for each term in each article. This creates a TF-IDF matrix where rows represent documents (articles) and columns represent terms (words), and each cell contains the TF-IDF score for that term in that document.
2.  **Query Processing:** When a user enters a query (e.g., "election results"), you would calculate the TF-IDF scores for the terms in the query, treating the query itself as a small "document".
3.  **Ranking:** You would then compare the TF-IDF vector of the query to the TF-IDF vectors of each article in the corpus. The articles with the highest similarity scores (e.g., cosine similarity) to the query are ranked higher in the search results.

In this scenario, TF-IDF helps the search engine prioritize articles that contain the query terms frequently *and* that use those terms in a way that is relatively unique within the overall collection of articles. Articles about general topics with words shared across most documents will receive lower scores compared to articles specifically focused on "election results".

## 3) Python method (if possible)

Scikit-learn's `TfidfVectorizer` is a convenient way to compute TF-IDF in Python.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Example documents
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]

# Create a TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Fit and transform the documents
tfidf_matrix = vectorizer.fit_transform(documents)

# Get the vocabulary (terms)
feature_names = vectorizer.get_feature_names_out()

# Convert the TF-IDF matrix to a dense array for easier inspection
tfidf_dense = tfidf_matrix.todense()

# Print the TF-IDF values for each document and term
import pandas as pd
df = pd.DataFrame(tfidf_dense, columns=feature_names)
print(df)

#Example: get vector for a new document
new_document = ["This is a new document about the first document"]
new_tfidf = vectorizer.transform(new_document)
new_tfidf_dense = new_tfidf.todense()
new_df = pd.DataFrame(new_tfidf_dense, columns=feature_names)
print("\nNew Document TF-IDF:")
print(new_df)
```

**Explanation:**

1.  **`TfidfVectorizer()`:**  Creates a TF-IDF vectorizer object.  You can customize the vectorizer with parameters like `ngram_range` (to consider phrases), `stop_words` (to remove common words), `max_df` (to ignore terms that appear in too many documents), `min_df` (to ignore terms that appear in too few documents), etc.
2.  **`fit_transform(documents)`:**  Fits the vectorizer to the documents (learns the vocabulary) and transforms the documents into a TF-IDF matrix.  The `fit` part calculates the IDF values based on the document collection provided.  The `transform` part applies TF-IDF transformation.
3.  **`get_feature_names_out()`:** Returns a list of the terms (features) that correspond to the columns of the TF-IDF matrix.
4.  **`todense()`:**  Converts the sparse TF-IDF matrix to a dense NumPy array, which is easier to read and manipulate.
5.  **`vectorizer.transform(new_document)`:** transforms a new document according to the fitted vectorizer. It's important to `fit_transform` on the training/corpus set only and use `.transform` for test sets, unseen documents, and queries.

## 4) Follow-up question

How does TF-IDF handle words that are not in the vocabulary (the set of words encountered during the `fit` step)? What are some strategies to mitigate issues related to out-of-vocabulary words, and how do these strategies impact the performance and interpretability of the TF-IDF model?