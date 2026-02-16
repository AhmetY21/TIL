---
title: "Text Similarity Measures (Cosine, Jaccard)"
date: "2026-02-16"
week: 8
lesson: 6
slug: "text-similarity-measures-cosine-jaccard"
---

# Topic: Text Similarity Measures (Cosine, Jaccard)

## 1) Formal definition (what is it, and how can we use it?)

Text similarity measures aim to quantify how similar two pieces of text are. This is typically achieved by comparing the representations of the text, not the raw string characters.  Cosine and Jaccard similarity are two popular methods.

*   **Cosine Similarity:** This measures the cosine of the angle between two non-zero vectors in a multi-dimensional space. In the context of text, these vectors typically represent the term frequency-inverse document frequency (TF-IDF) or word embeddings of the documents. The cosine similarity ranges from -1 to 1, where 1 indicates identical documents (same orientation), 0 indicates orthogonality (no similarity), and -1 indicates exactly opposite documents. It focuses on the orientation (direction) of the vectors rather than their magnitude (length).

    *   **How to use it:** Convert documents into vectors (e.g., using TF-IDF). Calculate the cosine similarity between each pair of document vectors. The resulting score indicates the similarity.

*   **Jaccard Similarity (Jaccard Index):** This measures the similarity between two sets. In the context of text, the sets typically represent the unique words or tokens present in the documents.  It's defined as the size of the intersection divided by the size of the union of the sets. The Jaccard similarity ranges from 0 to 1, where 1 indicates identical sets and 0 indicates disjoint sets (no common elements).

    *   **How to use it:** Convert documents into sets of unique words/tokens. Calculate the Jaccard index between each pair of document sets. The resulting score indicates the similarity.

## 2) Application scenario

Here are application scenarios for each measure:

*   **Cosine Similarity:**
    *   **Document Clustering:** Grouping similar documents together based on topic.
    *   **Information Retrieval:** Finding documents relevant to a user's query (search engines).
    *   **Recommendation Systems:** Recommending items (e.g., products, movies) to users based on their past behavior and the similarity of item descriptions.
    *   **Plagiarism Detection:** Identifying documents that are similar to existing sources.

*   **Jaccard Similarity:**
    *   **Duplicate Detection:** Identifying duplicate web pages or entries in a database.
    *   **Collaborative Filtering:** Finding users with similar tastes or interests based on the items they've interacted with (e.g., rated highly).
    *   **Market Basket Analysis:** Discovering associations between items purchased together.
    *   **Bioinformatics:** Comparing sets of genes or proteins.

## 3) Python method (if possible)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data (run once)
# nltk.download('punkt')

def cosine_similarity_example(doc1, doc2):
    """Calculates cosine similarity between two documents using TF-IDF."""
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([doc1, doc2])
    return cosine_similarity(vectors[0], vectors[1])[0][0]

def jaccard_similarity_example(doc1, doc2):
    """Calculates Jaccard similarity between two documents."""
    # Tokenize the documents
    tokens1 = set(word_tokenize(doc1.lower())) # Convert to lowercase and tokenize
    tokens2 = set(word_tokenize(doc2.lower()))

    # Calculate intersection and union
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)

    # Calculate Jaccard similarity
    return len(intersection) / len(union) if len(union) > 0 else 0.0


# Example usage:
doc1 = "This is a sample document about natural language processing."
doc2 = "This is another example document focusing on NLP."

cosine_sim = cosine_similarity_example(doc1, doc2)
jaccard_sim = jaccard_similarity_example(doc1, doc2)


print(f"Cosine Similarity: {cosine_sim}")
print(f"Jaccard Similarity: {jaccard_sim}")
```

## 4) Follow-up question

What are the limitations of Cosine and Jaccard similarity, and how can these limitations be addressed with other text similarity measures (e.g., Word Mover's Distance)?