Topic: Text Similarity Measures (Cosine, Jaccard)

1- **Provide formal definition, what is it and how can we use it?**

Text similarity measures quantify the degree of similarity between two pieces of text. Cosine and Jaccard similarity are two common approaches:

*   **Cosine Similarity:**
    *   **Definition:** Cosine similarity measures the angle between two non-zero vectors in a multi-dimensional space. In the context of text, these vectors typically represent term frequencies (or TF-IDF values) of words in the documents. The cosine of 0° is 1, indicating perfect similarity, while the cosine of 90° is 0, indicating orthogonality (no similarity).
    *   **Formula:**  `cosine_similarity(A, B) = (A . B) / (||A|| * ||B||)` where:
        *   `A . B` is the dot product of vectors A and B.
        *   `||A||` and `||B||` are the magnitudes (Euclidean norms) of vectors A and B.
    *   **Usage:**  It's often used when the length of the documents is a factor. Cosine similarity focuses on the orientation (angle) of the vectors, not their magnitude. This is crucial in text analysis, as longer documents might have higher term frequencies, even if they're not more semantically similar. It normalizes for document length.

*   **Jaccard Similarity:**
    *   **Definition:** Jaccard similarity measures the size of the intersection of two sets divided by the size of their union. In text analysis, the sets represent the unique words (or n-grams) present in the documents.
    *   **Formula:** `jaccard_similarity(A, B) = |A ∩ B| / |A ∪ B|` where:
        *   `|A ∩ B|` is the number of elements in the intersection of sets A and B.
        *   `|A ∪ B|` is the number of elements in the union of sets A and B.
    *   **Usage:**  It is suitable when you want to focus on the presence or absence of terms, rather than their frequency. It is less sensitive to document length than approaches like raw term frequency. It is most effective for short documents and when the presence or absence of words is more important than their frequency.

2- **Provide an application scenario**

*   **Scenario:** *Document Clustering/Topic Detection*. Imagine you have a large collection of news articles and you want to automatically group them into clusters based on their content.

    *   **How it helps:** You can use Cosine or Jaccard similarity to measure the similarity between each pair of articles.  Articles with high similarity scores are grouped together, forming clusters around common topics. For example, articles about the "economy" would cluster together, separate from articles about "sports" or "politics." Cosine Similarity might be preferred if articles have varying lengths, as it normalizes for that. Jaccard Similarity might be preferred if you care more about the presence of particular keywords in each article.
    *   *Specifically Cosine*: If you are looking to build a search engine, you want to return results that are the closest match to a user's query. In this case, the cosine similarity can be computed between the query and the document in a vector space, and return the highest scoring documents.
    *   *Specifically Jaccard*: If you are looking for duplicate entries in a database, a Jaccard similarity can quickly compare the sets of words in the documents, and anything exceeding a threshold can be flagged.

3- **Provide a method to apply in python**

python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    filtered_text = [w for w in word_tokens if not w in stop_words and w.isalnum()]
    return " ".join(filtered_text)


def calculate_cosine_similarity(doc1, doc2):
    """Calculates cosine similarity between two documents."""
    doc1 = preprocess_text(doc1)
    doc2 = preprocess_text(doc2)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([doc1, doc2])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return similarity


def calculate_jaccard_similarity(doc1, doc2):
    """Calculates Jaccard similarity between two documents."""
    doc1 = preprocess_text(doc1)
    doc2 = preprocess_text(doc2)

    set1 = set(doc1.split())
    set2 = set(doc2.split())

    if not set1 and not set2: # Handle empty sets to avoid division by zero
        return 1.0  # or 0.0 depending on desired behavior.  Empty sets are defined as identical
    elif not set1 or not set2:
        return 0.0

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    similarity = float(intersection) / union
    return similarity


# Example usage:
document1 = "The quick brown fox jumps over the lazy dog."
document2 = "A quick brown dog leaps over the lazy fox."

cosine_sim = calculate_cosine_similarity(document1, document2)
jaccard_sim = calculate_jaccard_similarity(document1, document2)

print(f"Cosine Similarity: {cosine_sim}")
print(f"Jaccard Similarity: {jaccard_sim}")


4- **Provide a follow up question about that topic**

How do more advanced text representations like word embeddings (e.g., Word2Vec, GloVe, BERT embeddings) affect the performance of cosine similarity in capturing semantic relationships compared to using TF-IDF vectors? What are the tradeoffs associated with using these more complex methods in terms of computational cost and memory requirements?