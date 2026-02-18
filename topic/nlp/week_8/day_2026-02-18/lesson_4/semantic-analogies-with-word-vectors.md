---
title: "Semantic Analogies with Word Vectors"
date: "2026-02-18"
week: 8
lesson: 4
slug: "semantic-analogies-with-word-vectors"
---

# Topic: Semantic Analogies with Word Vectors

## 1) Formal definition (what is it, and how can we use it?)

Semantic analogies with word vectors leverage the ability of word embeddings (like Word2Vec, GloVe, or fastText) to capture semantic relationships between words. The core idea is to express an analogy of the form "A is to B as C is to D" using vector arithmetic.

Formally, given words A, B, and C, we want to find a word D that satisfies the analogy "A:B :: C:D".  This is achieved by searching for a word vector *v_D* that is close to the vector *v_B - v_A + v_C*.

In other words, we assume that the relationship between A and B can be represented as a vector difference *v_B - v_A*. We then apply this same relationship to word C to find the analogous word D. We find D by adding this relationship to C's vector representation:  *v_D ≈ v_C + (v_B - v_A)*. The word whose vector is closest to the calculated *v_D* in the embedding space is then predicted as the solution.

More formally:

*   We are given word vectors  v_A, v_B, and v_C.
*   We want to find v_D such that: `v_D ≈ v_C + (v_B - v_A)`
*   We find the word D whose vector embedding is closest (using cosine similarity or other distance metrics) to the calculated vector `v_C + (v_B - v_A)`.

**How can we use it?**

*   **Reasoning:** Test the ability of a word embedding model to capture relational knowledge (e.g., "king - man + woman = queen").
*   **Knowledge Discovery:** Discover novel relationships between entities based on existing knowledge encoded in the word embeddings.
*   **Completing Analogies:** Given "A is to B as C is to ?", predict the most suitable word to fill the blank.
*   **Word Sense Disambiguation:** Choose the appropriate sense of a word based on its context within an analogy.

## 2) Application scenario

**Scenario:**  Language learning and teaching.

Imagine you are building a language learning application that helps users understand grammatical relationships. You want to demonstrate the concept of adjective comparisons (e.g., "tall", "taller", "tallest").

Using semantic analogies, you could present the following analogy:

"Good is to better as bad is to what?"

The application would use word vectors to calculate: `v_better - v_good + v_bad` and then find the word vector closest to the result. If the word embedding is well-trained, it should return "worse" as the answer, visually demonstrating the parallel grammatical structure. This makes the abstract concept of adjective comparison more tangible for the learner.

Another example is with country and capital cities: "Germany is to Berlin as France is to?"
## 3) Python method (if possible)

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def analogy(word1, word2, word3, word_vectors):
    """
    Solves the analogy "word1 is to word2 as word3 is to ?"

    Args:
        word1: The first word (string).
        word2: The second word (string).
        word3: The third word (string).
        word_vectors: A dictionary where keys are words (strings) and values are their corresponding vector representations (numpy arrays).

    Returns:
        The word (string) that best completes the analogy.
    """

    word1_vec = word_vectors[word1]
    word2_vec = word_vectors[word2]
    word3_vec = word_vectors[word3]

    # Calculate the target vector
    target_vec = word2_vec - word1_vec + word3_vec

    # Find the word with the closest vector
    best_word = None
    best_similarity = -1  # Initialize with a low value

    for word, vec in word_vectors.items():
        if word not in [word1, word2, word3]:  # Exclude the input words
            similarity = cosine_similarity([target_vec], [vec])[0][0] # cosine_similarity returns a matrix, so we get the scalar out

            if similarity > best_similarity:
                best_similarity = similarity
                best_word = word

    return best_word


# Example usage (assuming you have a dictionary of word vectors)
# In a real application, you'd load pre-trained word vectors.
# Here's a toy example:
word_vectors = {
    "king": np.array([0.9, 0.1, 0.3, 0.5]),
    "man": np.array([0.8, 0.2, 0.4, 0.6]),
    "woman": np.array([0.7, 0.3, 0.5, 0.7]),
    "queen": np.array([0.6, 0.4, 0.6, 0.8]),
    "germany": np.array([0.1, 0.9, 0.2, 0.8]),
    "berlin": np.array([0.2, 0.8, 0.3, 0.7]),
    "france": np.array([0.3, 0.7, 0.4, 0.6]),
    "paris": np.array([0.4, 0.6, 0.5, 0.5])

}

result = analogy("man", "king", "woman", word_vectors)
print(f"man:king :: woman:{result}") # Expected: queen

result = analogy("germany", "berlin", "france", word_vectors)
print(f"germany:berlin :: france:{result}") # Expected: paris
```

## 4) Follow-up question

How does the choice of word embedding model (Word2Vec, GloVe, fastText) influence the accuracy of solving semantic analogies, and why? What are some limitations of using this approach to solve analogies?