---
title: "GloVe (Global Vectors for Word Representation)"
date: "2026-02-17"
week: 8
lesson: 4
slug: "glove-global-vectors-for-word-representation"
---

# Topic: GloVe (Global Vectors for Word Representation)

## 1) Formal definition (what is it, and how can we use it?)

GloVe (Global Vectors for Word Representation) is an unsupervised learning algorithm for obtaining vector representations of words. Unlike count-based methods like Latent Semantic Analysis (LSA) which are statistically optimal but perform poorly on word analogy tasks, and predictive methods like word2vec which perform well on word analogy tasks but may not utilize global corpus statistics as efficiently, GloVe aims to leverage both local context and global statistics of word co-occurrence to learn word vectors.

The core idea of GloVe is based on the ratio of co-occurrence probabilities.  Instead of directly modeling the probabilities, GloVe models the log-bilinear model based on the ratios of co-occurrence probabilities.

Let:

*   *X* be the word-word co-occurrence matrix, where *X<sub>ij</sub>* represents the number of times word *j* appears in the context of word *i*.
*   *X<sub>i</sub>* be the number of times any word appears in the context of word *i*, i.e.,  *X<sub>i</sub> = Σ<sub>k</sub> X<sub>ik</sub>*.
*   *P<sub>ij</sub> = P(j | i) = X<sub>ij</sub> / X<sub>i</sub>* be the probability of word *j* appearing in the context of word *i*.
*   *v<sub>i</sub>* and *v<sub>j</sub>* be the word vectors for words *i* and *j*, respectively.
*   *b<sub>i</sub>* and *b<sub>j</sub>* be the biases associated with words *i* and *j*, respectively.

GloVe attempts to learn word vectors such that:

`v<sub>i</sub><sup>T</sup> v<sub>j</sub> + b<sub>i</sub> + b<sub>j</sub> ≈ log(X<sub>ij</sub>)`

The overall objective function that GloVe minimizes is a weighted least squares regression model:

`J = Σ<sub>i,j</sub> f(X<sub>ij</sub>) (v<sub>i</sub><sup>T</sup> v<sub>j</sub> + b<sub>i</sub> + b<sub>j</sub> - log(X<sub>ij</sub>))<sup>2</sup>`

where *f(X<sub>ij</sub>)* is a weighting function that helps to avoid rare co-occurrences dominating the learning process. A common weighting function is:

`f(x) = (x/x<sub>max</sub>)<sup>α</sup>  if x < x<sub>max</sub>,  else 1`

where *α* is typically around 0.75 and *x<sub>max</sub>* is usually set to 100.

How can we use it? After training the GloVe model, we obtain word vectors that capture semantic and syntactic relationships between words. These word vectors can then be used as:

*   **Input features for downstream NLP tasks:** such as text classification, named entity recognition, sentiment analysis, and machine translation.
*   **To perform word similarity analysis:**  by calculating the cosine similarity between the vectors of different words.
*   **To solve word analogy problems:** (e.g., "king - man + woman = queen").

## 2) Application scenario

Consider a scenario where you want to build a sentiment analysis model for movie reviews. You can use GloVe embeddings as input features for your model. Here's how:

1.  **Pre-process your text data:** Tokenize the movie reviews into individual words.
2.  **Load pre-trained GloVe embeddings:** Download pre-trained GloVe vectors (e.g., trained on Wikipedia or Common Crawl) of a suitable dimension (e.g., 100d, 200d, 300d).
3.  **Represent words as vectors:** For each word in your vocabulary, look up its corresponding vector from the GloVe embeddings. If a word is not in the pre-trained vocabulary, you can use a random vector or a zero vector.
4.  **Create sentence/review embeddings:**  Average the word vectors in each movie review to create a single vector representation for the entire review. More sophisticated methods, like using a weighted average or a recurrent neural network (RNN) with GloVe embeddings, can also be employed.
5.  **Train your sentiment analysis model:** Use the sentence/review embeddings as input features to train a classification model (e.g., logistic regression, support vector machine, neural network) to predict the sentiment (positive, negative, neutral) of each review.

Using GloVe embeddings provides a good starting point and often improves the performance of the sentiment analysis model compared to using one-hot encoding or other simpler word representations.

## 3) Python method (if possible)

```python
import numpy as np

# Assuming you have downloaded pre-trained GloVe embeddings (e.g., glove.6B.100d.txt)
# and stored it in a file called 'glove.6B.100d.txt'

def load_glove_embeddings(file_path):
    """Loads GloVe embeddings from a text file.

    Args:
        file_path (str): Path to the GloVe embeddings file.

    Returns:
        dict: A dictionary mapping words to their corresponding vectors.
    """
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

def sentence_to_embedding(sentence, embeddings, embedding_dim=100):
    """Converts a sentence to an embedding vector by averaging the GloVe vectors of its words.

    Args:
        sentence (str): The input sentence.
        embeddings (dict): A dictionary mapping words to their corresponding vectors.
        embedding_dim (int): The dimensionality of the GloVe embeddings.

    Returns:
        numpy.ndarray: The embedding vector for the sentence. Returns a zero vector if no words in the sentence have embeddings.
    """
    words = sentence.lower().split()
    word_vectors = []
    for word in words:
        if word in embeddings:
            word_vectors.append(embeddings[word])

    if not word_vectors:
        return np.zeros(embedding_dim)  # Return zero vector if no words have embeddings

    return np.mean(word_vectors, axis=0)

# Example usage:
if __name__ == '__main__':
    glove_file = 'glove.6B.100d.txt'  # Replace with your GloVe file path
    glove_embeddings = load_glove_embeddings(glove_file)

    sentence1 = "This is a great movie."
    sentence2 = "This movie is terrible."

    embedding1 = sentence_to_embedding(sentence1, glove_embeddings)
    embedding2 = sentence_to_embedding(sentence2, glove_embeddings)

    print(f"Embedding for '{sentence1}': {embedding1[:5]}...") # Print first 5 elements
    print(f"Embedding for '{sentence2}': {embedding2[:5]}...") # Print first 5 elements

    # Example using cosine similarity to compare sentences:
    from numpy.linalg import norm

    def cosine_similarity(v1, v2):
      return np.dot(v1, v2) / (norm(v1) * norm(v2))

    similarity = cosine_similarity(embedding1, embedding2)
    print(f"Cosine similarity between sentences: {similarity}")
```

## 4) Follow-up question

How does the choice of the weighting function *f(X<sub>ij</sub>)* in the GloVe objective function affect the resulting word embeddings?  Are there other weighting functions that could be used, and what are their potential advantages or disadvantages compared to the commonly used one?