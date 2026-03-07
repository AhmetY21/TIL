---
title: "Semantic Analogies with Word Vectors"
date: "2026-03-07"
week: 10
lesson: 6
slug: "semantic-analogies-with-word-vectors"
---

# Topic: Semantic Analogies with Word Vectors

## 1) Formal definition (what is it, and how can we use it?)

Semantic analogies with word vectors refer to the ability to use word embeddings to solve analogies of the form "a is to b as c is to d".  We represent words as vectors in a high-dimensional space where semantically similar words are closer together.  The core idea is that the relationship between `a` and `b` can be captured as a vector difference (`b` - `a`). We hypothesize that the same relationship should hold between `c` and `d`, so we aim to find a word vector `d` such that `d` is approximately equal to `c + (b - a)`.

More formally:

Given words `a`, `b`, and `c`, find the word `d` such that:

`d ≈ c + (b - a)`

We measure the similarity between `d` and candidate words using cosine similarity (or other distance metrics) between their vector representations. The word vector closest to the calculated vector `c + (b - a)` is then predicted to be the answer to the analogy.

We can use this technique to answer questions like:

*   "king is to queen as man is to woman" (gender analogy)
*   "Paris is to France as Berlin is to Germany" (capital analogy)
*   "walk is to walking as eat is to eating" (verb tense analogy)

Essentially, it allows us to explore and quantify semantic relationships between words based on their vector representations learned from large text corpora. The quality of the analogy solving depends heavily on the quality of the word embeddings themselves.

## 2) Application scenario

A practical application of semantic analogies is in language understanding and information retrieval systems. Imagine a system that needs to understand relationships between concepts in a document or query.  For example:

*   **Query Expansion:** If a user searches for "car repair," a system could use analogy to expand the query to include related terms. Knowing "car is to mechanic as computer is to software engineer" allows the system to understand the user might also be interested in "computer repair" or "software engineering" related services.

*   **Machine Translation:**  Analogy can assist in finding appropriate translations by identifying parallel relationships in different languages. If "gato is to gato_s as perro is to ?", the system, understanding the plural form relationship can determine that "perro_s" would be the aporopriate translation of "perro" to plural form.

*   **Question Answering:**  In question answering systems, analogies can help in reasoning about the relationships between entities in the question and the knowledge base.  For instance, if the system knows the analogy "London is to England as Washington is to United States", and the question is "What is the capital of England?", it can use this knowledge to infer that 'capital' is the relation and apply it to the United States.

*   **Sentiment Analysis:**  Analogies can aid in capturing nuanced sentiment relationships. For example, if "good is to great as bad is to terrible," understanding the intensification relationship between sentiment words can improve the ability to accurately classify sentiment, especially in the case of sarcasm or implied opinions.

## 3) Python method (if possible)
```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Assume you have word embeddings loaded into a dictionary like this:
# word_embeddings = {"king": [0.1, 0.2, 0.3], "queen": [0.4, 0.5, 0.6], ...}
# For demonstration purposes, let's create some dummy embeddings:

word_embeddings = {
    "king": np.array([1, 1, 1]),
    "queen": np.array([2, 2, 2]),
    "man": np.array([3, 3, 3]),
    "woman": np.array([4, 4, 4]),
    "paris": np.array([5, 5, 5]),
    "france": np.array([6, 6, 6]),
    "berlin": np.array([7, 7, 7]),
    "germany": np.array([8, 8, 8]),
    "walk": np.array([9, 9, 9]),
    "walking": np.array([10, 10, 10]),
    "eat": np.array([11, 11, 11]),
    "eating": np.array([12, 12, 12])
}


def analogy(word_a, word_b, word_c, word_embeddings):
    """
    Finds the word d such that a is to b as c is to d.

    Args:
        word_a: The first word in the analogy (e.g., "king").
        word_b: The second word in the analogy (e.g., "queen").
        word_c: The third word in the analogy (e.g., "man").
        word_embeddings: A dictionary mapping words to their vector embeddings.

    Returns:
        The predicted word d (string) and its cosine similarity score.  Returns None, None if any input word is not found.
    """

    if word_a not in word_embeddings or word_b not in word_embeddings or word_c not in word_embeddings:
        print("One or more input words not found in embeddings.")
        return None, None

    e_a, e_b, e_c = word_embeddings[word_a], word_embeddings[word_b], word_embeddings[word_c]

    e_d_predicted = e_c + (e_b - e_a)

    best_word = None
    best_similarity = -1.0

    for word, embedding in word_embeddings.items():
        if word in [word_a, word_b, word_c]: # Exclude input words
            continue
        similarity = cosine_similarity(e_d_predicted.reshape(1, -1), embedding.reshape(1, -1))[0][0]
        if similarity > best_similarity:
            best_similarity = similarity
            best_word = word

    return best_word, best_similarity


# Example usage:
predicted_word, similarity = analogy("king", "queen", "man", word_embeddings)
print(f"king is to queen as man is to {predicted_word} (similarity: {similarity})") # Output: king is to queen as man is to woman (similarity: 0.9999999999999998)

predicted_word, similarity = analogy("paris", "france", "berlin", word_embeddings)
print(f"paris is to france as berlin is to {predicted_word} (similarity: {similarity})")

predicted_word, similarity = analogy("walk", "walking", "eat", word_embeddings)
print(f"walk is to walking as eat is to {predicted_word} (similarity: {similarity})")

```

## 4) Follow-up question

How does the quality of the word embeddings (e.g., trained with different algorithms like Word2Vec, GloVe, or FastText, and different corpus sizes) impact the performance of semantic analogy solving? And, are there any specific pre-processing steps that can be taken on the text corpus before training word embeddings to improve the accuracy of solving these analogies?  For example, does stemming or lemmatization help, or hurt?