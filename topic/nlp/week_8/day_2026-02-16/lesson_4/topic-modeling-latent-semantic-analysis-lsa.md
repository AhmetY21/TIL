---
title: "Topic Modeling: Latent Semantic Analysis (LSA)"
date: "2026-02-16"
week: 8
lesson: 4
slug: "topic-modeling-latent-semantic-analysis-lsa"
---

# Topic: Topic Modeling: Latent Semantic Analysis (LSA)

## 1) Formal definition (what is it, and how can we use it?)

Latent Semantic Analysis (LSA), also known as Latent Semantic Indexing (LSI), is a technique in natural language processing (NLP) used for topic modeling and dimensionality reduction. It aims to uncover the latent (hidden) semantic structure within a collection of documents.

**What it is:** LSA is based on the assumption that there is an underlying semantic structure in the usage of words across documents, even if words don't explicitly appear together. It uses singular value decomposition (SVD) to reduce the dimensions of a term-document matrix.

**How it works:**

1.  **Term-Document Matrix:**  A matrix is created where rows represent terms (words) and columns represent documents.  Each cell (i, j) contains a value representing the frequency of term 'i' in document 'j' (e.g., term frequency-inverse document frequency, or TF-IDF).

2.  **Singular Value Decomposition (SVD):** The matrix is decomposed using SVD into three matrices:  U, Σ, and V<sup>T</sup>.
    *   **U (Term-Topic Matrix):**  Represents the relationship between terms and topics.  Each row corresponds to a term, and each column corresponds to a topic. The values indicate the importance of the term for that topic.
    *   **Σ (Singular Values):**  A diagonal matrix containing singular values. These values represent the "strength" or importance of each topic.
    *   **V<sup>T</sup> (Topic-Document Matrix):** Represents the relationship between topics and documents.  Each row corresponds to a topic, and each column corresponds to a document. The values indicate the relevance of the topic to that document.

3.  **Dimensionality Reduction:** The 'k' largest singular values (and corresponding columns/rows in U and V<sup>T</sup>) are kept, where 'k' is the desired number of topics. This truncation reduces the dimensionality of the matrices while retaining the most important information, effectively capturing the underlying semantic structure. The reduced matrices are U<sub>k</sub>, Σ<sub>k</sub>, and V<sub>k</sub><sup>T</sup>.

**How can we use it?:**

*   **Topic Modeling:**  By examining the terms with the highest values in the U matrix for a given topic, we can interpret the meaning of that topic.  Similarly, the V<sup>T</sup> matrix shows the relevance of each document to each topic.
*   **Document Similarity:** Documents can be compared based on their topic vectors (rows in V<sup>T</sup>).  Documents with similar topic vectors are considered semantically similar.
*   **Information Retrieval:**  LSA can be used to retrieve documents relevant to a query, even if the query doesn't contain the exact same words as the documents. The query can be represented as a pseudo-document and compared to the document vectors in V<sup>T</sup>.
*   **Dimensionality Reduction:** The reduced matrices can be used for downstream tasks like clustering or classification, reducing computational complexity and potentially improving performance.

## 2) Application scenario

**Scenario:** Suppose we have a collection of news articles about various topics like politics, sports, technology, and finance. We want to automatically group these articles into meaningful topics.

**How LSA can be applied:**

1.  **Data Preprocessing:** Clean the text data (e.g., remove stop words, punctuation, and perform stemming or lemmatization).
2.  **Term-Document Matrix Creation:** Build a term-document matrix, where rows represent words and columns represent news articles.  TF-IDF weighting is commonly used.
3.  **LSA Application:** Apply SVD to the term-document matrix and reduce the dimensionality to a smaller number of topics (e.g., 10-20).
4.  **Topic Interpretation:**  Examine the top words associated with each topic (from the U matrix). For example, a topic might have words like "election," "candidate," "vote," and "campaign," indicating that it represents the "Politics" topic.
5.  **Document Assignment:**  Assign each news article to the most relevant topic based on its vector in the V<sup>T</sup> matrix.
6.  **Insights:** By analyzing the topics and the articles assigned to them, we can gain insights into the main themes covered in the news articles. We can also identify relationships between different topics or articles.

## 3) Python method (if possible)

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Sample documents
documents = [
    "The cat sat on the mat.",
    "The dog chased the cat.",
    "The bird flew in the sky.",
    "The airplane took off from the airport.",
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning is a type of machine learning."
]

# 1. Create the Term-Document Matrix using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english') # Remove common English stop words
X = vectorizer.fit_transform(documents)

# 2. Apply Truncated SVD (LSA)
n_components = 2  # Number of topics to extract
svd = TruncatedSVD(n_components=n_components, random_state=42)
svd.fit(X)

# 3. Get the topic-term matrix (U matrix)
terms = vectorizer.get_feature_names_out()
for i, comp in enumerate(svd.components_):
    terms_zip = zip(terms, comp)
    sorted_terms = sorted(terms_zip, key=lambda x: x[1], reverse=True)[:5] # Top 5 terms per topic
    print(f"Topic {i}:")
    for term, value in sorted_terms:
        print(f"{term}: {value:.3f}")
    print("\n")

# 4. Get the document-topic matrix (V^T matrix)
document_topic_matrix = svd.transform(X)
print("Document-Topic Matrix:")
print(document_topic_matrix)

# Example: Check topic distribution of first document
print(f"\nTopic distribution for document 1: {document_topic_matrix[0]}")
```

**Explanation:**

*   **TfidfVectorizer:**  Used to create the term-document matrix with TF-IDF weighting.  `stop_words='english'` removes common English words that don't carry much meaning.
*   **TruncatedSVD:**  This is the scikit-learn implementation of SVD used in LSA. We specify the number of components (topics) we want to extract using `n_components`. `random_state` ensures reproducibility.  Truncated SVD is preferred over regular SVD for large sparse matrices, which are common in text data.
*   **svd.components\_:**  This attribute holds the U<sup>T</sup> matrix (transpose of the U matrix). Each row represents a topic, and the values indicate the importance of each term for that topic.
*   **svd.transform(X):**  Transforms the term-document matrix X into the document-topic matrix (V<sup>T</sup> matrix).

## 4) Follow-up question

How does LSA handle polysemy (words with multiple meanings) and synonymy (different words with similar meanings), and what are its limitations in addressing these challenges compared to more advanced topic modeling techniques like LDA (Latent Dirichlet Allocation)?