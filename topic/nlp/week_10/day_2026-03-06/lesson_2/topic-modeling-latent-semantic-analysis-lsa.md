---
title: "Topic Modeling: Latent Semantic Analysis (LSA)"
date: "2026-03-06"
week: 10
lesson: 2
slug: "topic-modeling-latent-semantic-analysis-lsa"
---

# Topic: Topic Modeling: Latent Semantic Analysis (LSA)

## 1) Formal definition (what is it, and how can we use it?)

Latent Semantic Analysis (LSA), also known as Latent Semantic Indexing (LSI), is a technique in Natural Language Processing (NLP) for discovering underlying semantic relationships between words and documents. It's a type of topic modeling method. At its core, LSA relies on singular value decomposition (SVD) to reduce the dimensionality of a term-document matrix.

**What it is:**

*   LSA starts with a term-document matrix where rows represent terms (words) and columns represent documents. Each cell contains a weight reflecting the importance of the term in that document (e.g., Term Frequency-Inverse Document Frequency - TF-IDF).
*   SVD is applied to this matrix. SVD decomposes the original matrix into three matrices: U, Σ, and V<sup>T</sup>.
    *   U (term-topic matrix): Represents the relationship between terms and topics. Each row corresponds to a term, and each column corresponds to a topic.  The values in this matrix represent the "loading" or importance of each term for each topic.
    *   Σ (singular value matrix): A diagonal matrix containing singular values, which represent the strength or importance of each topic.  Topics are ordered by their singular values (highest to lowest), so the first few topics are the most important.
    *   V<sup>T</sup> (document-topic matrix): Represents the relationship between documents and topics. Each row corresponds to a document, and each column corresponds to a topic. The values indicate how relevant each topic is to each document.
*   Dimensionality reduction is achieved by keeping only the top *k* singular values (and corresponding columns in U and rows in V<sup>T</sup>).  This *k* represents the number of topics we want to discover.

**How can we use it?**

*   **Topic Discovery:** LSA helps identify latent topics present within a collection of documents.  By examining the terms with the highest weights in the term-topic matrix (U), we can interpret what each topic represents.
*   **Document Similarity:**  Documents can be compared based on their topic representation in the document-topic matrix (V<sup>T</sup>). Documents with similar topic vectors are considered more similar.
*   **Information Retrieval:**  LSA can be used to improve information retrieval by matching queries to documents based on semantic similarity rather than just keyword matching.  The query can be represented as a pseudo-document and compared to the document-topic vectors.
*   **Text Summarization:** Identifying the most important topics in a document or set of documents can inform the creation of concise summaries.
*   **Cross-Lingual Retrieval:** With parallel corpora, LSA can be used to map terms and documents across different languages into a shared semantic space.

## 2) Application scenario

**Scenario:** Analyzing customer feedback on a product.

A company collects thousands of customer reviews for its new smartphone. They want to understand the main topics or concerns that customers are discussing without manually reading every review.

**LSA Application:**

1.  **Data Preparation:** The reviews are preprocessed (tokenization, stop word removal, stemming/lemmatization).
2.  **Term-Document Matrix Creation:** A term-document matrix is created where rows are terms and columns are customer reviews. TF-IDF weighting is applied to assign importance scores.
3.  **SVD Application:** SVD is performed on the term-document matrix.
4.  **Dimensionality Reduction:** The number of topics (*k*) is chosen (e.g., 10 topics). The top 10 singular values and corresponding columns/rows from U and V<sup>T</sup> are kept.
5.  **Topic Interpretation:** The top terms for each of the 10 topics are examined.  For example:
    *   Topic 1: "battery", "life", "charge", "power" (might represent battery performance)
    *   Topic 2: "screen", "display", "bright", "resolution" (might represent screen quality)
    *   Topic 3: "camera", "photo", "quality", "image" (might represent camera performance)
6.  **Analysis & Action:** The company can then analyze the V<sup>T</sup> matrix to see which reviews are most related to which topics. They can identify common problems (e.g., battery draining too fast) and address them through product updates or improved customer support.

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
    "The dog barked loudly.",
    "The cat purred softly."
]

# 1. Create TF-IDF matrix
vectorizer = TfidfVectorizer(stop_words='english')  # Remove common words
X = vectorizer.fit_transform(documents)

# 2. Apply Truncated SVD (LSA)
n_components = 2  # Number of topics
svd = TruncatedSVD(n_components=n_components, algorithm='arpack', random_state=42) # Set random_state for reproducibility
svd.fit(X)

# 3. Get topic terms
terms = vectorizer.get_feature_names_out()

for i, comp in enumerate(svd.components_):
    terms_with_score = sorted(zip(terms, comp), key=lambda x: x[1], reverse=True)
    print(f"Topic {i + 1}:")
    for term, score in terms_with_score[:5]:  # Print top 5 terms per topic
        print(f"{term}: {score:.3f}")

# 4. Get document-topic matrix
document_topic_matrix = svd.transform(X)
print("\nDocument-Topic Matrix:")
print(document_topic_matrix)
```

**Explanation:**

1.  **Import Libraries:** Imports necessary libraries (NumPy, scikit-learn).
2.  **Sample Documents:**  Defines a list of example documents.
3.  **TF-IDF Vectorization:**  Uses `TfidfVectorizer` to create a term-document matrix, assigning TF-IDF weights to terms.  `stop_words='english'` removes common English stop words.
4.  **Truncated SVD:** Uses `TruncatedSVD` (scikit-learn's implementation of SVD for sparse matrices). `n_components` specifies the number of topics. The `random_state` parameter is set for reproducibility. The `algorithm` parameter is set to 'arpack' since the default 'randomized' algorithm can be problematic for very small datasets like this one (it can sometimes fail).
5.  **Topic Term Extraction:**  Iterates through the components (topics) of the SVD result (`svd.components_`). For each topic, it extracts the terms with the highest weights and prints the top 5 terms, along with their scores, providing insights into the topic's meaning.
6.  **Document-Topic Matrix:** Calculates the document-topic matrix (`svd.transform(X)`) which represents each document in the topic space. The rows represent the documents, and the columns represent the topics.  The values indicate the strength of the association between the document and the topic.

## 4) Follow-up question

How does LSA compare to other topic modeling techniques like Latent Dirichlet Allocation (LDA), and what are the advantages and disadvantages of using LSA in different scenarios?