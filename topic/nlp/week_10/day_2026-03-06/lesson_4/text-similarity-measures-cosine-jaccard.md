---
title: "Text Similarity Measures (Cosine, Jaccard)"
date: "2026-03-06"
week: 10
lesson: 4
slug: "text-similarity-measures-cosine-jaccard"
---

# Topic: Text Similarity Measures (Cosine, Jaccard)

## 1) Formal definition (what is it, and how can we use it?)

Text similarity measures quantify the degree to which two pieces of text are alike. They are used to determine how similar the content or meaning of two texts is based on their textual representation. Cosine similarity and Jaccard similarity are two popular methods.

*   **Cosine Similarity:**
    *   **What it is:** Cosine similarity measures the angle between two vectors representing the texts in a multi-dimensional space. These vectors are typically term frequency-inverse document frequency (TF-IDF) vectors or word embeddings. A smaller angle indicates higher similarity. Specifically, it's the cosine of the angle between them.
    *   **How to use it:** It provides a score between -1 and 1, where 1 means the vectors are perfectly aligned (identical), 0 means they are orthogonal (no similarity), and -1 means they are diametrically opposed (opposite).
    *   **Formula:**  `cosine_similarity(A, B) = (A . B) / (||A|| * ||B||)`, where A and B are the vector representations of the texts, A . B is the dot product of A and B, and ||A|| and ||B|| are the magnitudes (Euclidean norms) of A and B.  It focuses on the orientation rather than the magnitude of the vectors.

*   **Jaccard Similarity (also known as Jaccard Index):**
    *   **What it is:** Jaccard similarity measures the similarity between two sets (in this case, sets of words or tokens in the texts). It is defined as the size of the intersection divided by the size of the union of the sets.
    *   **How to use it:** It provides a score between 0 and 1, where 1 means the sets are identical, and 0 means they have no common elements.
    *   **Formula:** `Jaccard(A, B) = |A ∩ B| / |A ∪ B|`, where A and B are the sets of words/tokens in the texts, |A ∩ B| is the cardinality (size) of the intersection of A and B, and |A ∪ B| is the cardinality of the union of A and B.

## 2) Application scenario

Here are examples where text similarity measures can be used:

*   **Document Clustering:** Grouping similar documents together based on content.  Cosine similarity often works well for this.
*   **Information Retrieval:** Ranking search results by relevance to a query.  Cosine similarity between the query and documents is frequently used.
*   **Plagiarism Detection:** Identifying instances where text has been copied from another source. Jaccard similarity can be useful for checking overlap in wording.
*   **Question Answering:** Finding the most similar question in a knowledge base to a given question to retrieve the corresponding answer.
*   **Spam Detection:** Identifying near-duplicate emails or messages.
*   **Product Recommendation:** Recommending similar products to users based on their product descriptions.
*   **Paraphrase Detection:** Determine if two sentences or passages convey the same meaning, even if they use different wording.
*   **Duplicate detection:** Finding duplicate questions on Stack Overflow for moderation.
*   **Summarization:** Extracting the most representative sentences from a document by measuring similarity between candidate sentences and the original document.

## 3) Python method (if possible)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data (only needed once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
import string

def preprocess_text(text):
    """Tokenize, remove stopwords, and punctuation."""
    tokens = word_tokenize(text.lower())
    table = str.maketrans('', '', string.punctuation) #remove punctuation
    stripped = [w.translate(table) for w in tokens]

    stop_words = set(stopwords.words('english'))
    words = [word for word in stripped if word.isalpha() and word not in stop_words] #isalpha to remove numbers
    return " ".join(words)


def cosine_similarity_example(text1, text2):
    """Calculates cosine similarity using TF-IDF vectorization."""

    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)

    corpus = [text1, text2]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity_score


def jaccard_similarity_example(text1, text2):
    """Calculates Jaccard similarity using sets of words."""

    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)

    set1 = set(text1.split())
    set2 = set(text2.split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if len(union) == 0: #Handle the case of empty sets which would cause a division by zero error
      return 0.0
    return len(intersection) / len(union)


# Example usage
text1 = "This is the first document."
text2 = "This document is the second document."
text3 = "Totally unrelated document."

print(f"Cosine similarity between text1 and text2: {cosine_similarity_example(text1, text2)}")
print(f"Jaccard similarity between text1 and text2: {jaccard_similarity_example(text1, text2)}")

print(f"Cosine similarity between text1 and text3: {cosine_similarity_example(text1, text3)}")
print(f"Jaccard similarity between text1 and text3: {jaccard_similarity_example(text1, text3)}")
```

## 4) Follow-up question

How do different text preprocessing techniques (e.g., stemming, lemmatization, stop word removal, handling synonyms) affect the results of cosine similarity and Jaccard similarity, and in what scenarios would each technique be most beneficial? Why does one need to preprocess the text first?