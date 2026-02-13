---
title: "TF-IDF (Term Frequency-Inverse Document Frequency)"
date: "2026-02-13"
week: 7
lesson: 3
slug: "tf-idf-term-frequency-inverse-document-frequency"
---

# Topic: TF-IDF (Term Frequency-Inverse Document Frequency)

## 1) Formal definition (what is it, and how can we use it?)

TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. It's a widely used technique in information retrieval and text mining to determine the significance of words within a document compared to the entire corpus.  It measures the relevance of a word to a document relative to a corpus.

TF-IDF is calculated by multiplying two metrics: Term Frequency (TF) and Inverse Document Frequency (IDF).

*   **Term Frequency (TF):**  Measures how frequently a term appears in a document. A common approach is to simply count the number of times a term appears in a document, divided by the total number of words in that document.  Other variations exist, such as logarithmic scaling to reduce the impact of very frequent words.

    *   TF(t,d) = (Number of times term *t* appears in document *d*) / (Total number of words in document *d*)

*   **Inverse Document Frequency (IDF):** Measures how important a term is across the entire corpus. It aims to reduce the weight of common words (e.g., "the", "a", "is") that appear frequently in almost all documents, while increasing the weight of rare words that are more specific to particular documents. It's typically calculated as the logarithm of the total number of documents in the corpus divided by the number of documents containing the term.

    *   IDF(t, D) = log(Total number of documents in corpus *D* / Number of documents containing term *t*)

The TF-IDF score is then calculated as:

*   TF-IDF(t, d, D) = TF(t, d) * IDF(t, D)

**How can we use it?**

We use TF-IDF to:

*   **Rank search results:** Documents with higher TF-IDF scores for the query terms are considered more relevant.
*   **Feature extraction for text classification:**  TF-IDF scores can be used as features for machine learning models to classify documents.  Higher scores represent more relevant words for the class.
*   **Information retrieval:**  Identify the most important words in a document for indexing and retrieval purposes.
*   **Keyword extraction:**  Identify important keywords within a document.
*   **Text summarization:**  Identify sentences containing terms with high TF-IDF scores to form a summary.
*   **Document similarity:** TF-IDF vectors can be used to compare the similarity between documents. Documents with similar TF-IDF vectors are considered more similar.

## 2) Application scenario

Imagine you have a collection of articles about different sports: Soccer, Basketball, and Tennis.

*   **Scenario:** You want to build a search engine for these articles. When a user searches for "scoring strategies", you want to return the articles that are most relevant to that query.

*   **How TF-IDF helps:** TF-IDF can be used to weigh the importance of the terms "scoring" and "strategies" in each article relative to the entire collection.  An article about basketball that heavily uses the term "scoring" and "strategies" (relative to other articles) will have a higher TF-IDF score for those terms.  Conversely, articles that mention those terms only briefly might be less relevant.

*   **Expected outcome:** The search engine would return articles about Basketball (and perhaps Soccer) that discuss "scoring strategies" prominently, while articles about Tennis, or Basketball articles that don't focus on strategies, would be ranked lower.

## 3) Python method (if possible)
```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
    "This is a document about scoring in basketball.  Scoring is very important.",
    "Tennis is a sport that requires precise strokes and strategy.",
    "Soccer is a sport that requires scoring goals using different strategies."
]

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the documents
tfidf_matrix = vectorizer.fit_transform(documents)

# Get the feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Convert the TF-IDF matrix to a dense array (for easier viewing)
dense = tfidf_matrix.todense()
denselist = dense.tolist()

# Print the TF-IDF scores for each document and each term
import pandas as pd
df = pd.DataFrame(denselist, columns=feature_names)
print(df)

# Example: Get the TF-IDF score for the term "scoring" in the 5th document
scoring_index = feature_names.tolist().index("scoring")
print(f"\nTF-IDF score for 'scoring' in document 5: {denselist[4][scoring_index]}")

# To search for documents most relevant to the query "scoring strategies":
query = "scoring strategies"
query_vector = vectorizer.transform([query])
#We can then calculate cosine similarity between query_vector and each row of tfidf_matrix
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(query_vector, tfidf_matrix)
print("\nCosine similarities between query and documents:")
print(similarities)
```

## 4) Follow-up question

TF-IDF is a relatively simple and computationally efficient method. What are some of its limitations, and what are some alternative, more advanced techniques that address those limitations?