---
title: "Word2Vec: Negative Sampling and Optimization"
date: "2026-02-17"
week: 8
lesson: 3
slug: "word2vec-negative-sampling-and-optimization"
---

# Topic: Word2Vec: Negative Sampling and Optimization

## 1) Formal definition (what is it, and how can we use it?)

Word2Vec is a group of techniques to learn word embeddings. These embeddings are dense, low-dimensional vector representations of words that capture semantic relationships between them. Negative sampling is a crucial optimization technique used within the Skip-gram and Continuous Bag-of-Words (CBOW) models of Word2Vec.

**The problem without negative sampling:** The standard Skip-gram and CBOW models attempt to maximize the probability of predicting context words given a target word (Skip-gram) or predicting a target word given its context words (CBOW).  This involves iterating through the entire vocabulary for each training sample and calculating probabilities using softmax, which is computationally expensive, especially with large vocabularies.  Updating the model parameters after each word encountered also slows training.

**What negative sampling does:** Negative sampling addresses this inefficiency by reformulating the task as a binary classification problem. Instead of updating weights for every word in the vocabulary, it only updates weights for:

*   **The "positive" sample:** The actual context-target word pair from the training data.
*   **A few "negative" samples:** Randomly selected words from the vocabulary that are *not* in the context of the target word.  These negative samples represent words that *should not* be predicted.

The model learns to distinguish between these positive and negative samples.  In essence, it's trained to assign high probabilities to the positive sample and low probabilities to the negative samples.

**How we can use it:**

1.  **Faster Training:**  Dramatically reduces the computational cost, especially for large vocabularies, enabling the training of Word2Vec models on massive datasets.
2.  **Word Embeddings:**  The resulting word embeddings capture semantic relationships. Similar words will have similar vector representations.  These embeddings can be used in various downstream NLP tasks.
3.  **Downstream Tasks:** Common applications of these word embeddings include:
    *   **Text classification:** Using word embeddings as features to classify documents.
    *   **Sentiment analysis:** Determining the sentiment expressed in a piece of text.
    *   **Machine translation:** Mapping words between different languages based on their embeddings.
    *   **Information retrieval:** Finding documents relevant to a given query based on the semantic similarity of words.
    *   **Recommender systems:**  Representing users and items as embeddings to predict user preferences.

**Optimization:**

Beyond negative sampling, other optimization techniques are used to enhance Word2Vec performance. These include:

*   **Subsampling of Frequent Words:**  Words like "the," "a," and "is" occur very frequently and provide less information. Subsampling reduces the frequency of these words during training, which can improve the quality of embeddings for less frequent words.  A word *w* is discarded with probability `P(w) = 1 - sqrt(t/f(w))`, where `f(w)` is the frequency of the word and `t` is a chosen threshold (typically around 1e-5).
*   **Hierarchical Softmax:**  Another alternative to negative sampling, hierarchical softmax uses a binary tree to represent the vocabulary. The probability of a word is calculated by traversing the tree, which reduces the computational complexity from O(V) to O(log V), where V is the vocabulary size. While hierarchical softmax can be effective, negative sampling is often preferred in practice.

## 2) Application scenario

**Scenario:** Building a movie recommendation system.

**Problem:** We want to recommend movies to users based on their past viewing history.

**Word2Vec Solution:**

1.  **Data Preparation:** Treat each user's viewing history as a "sentence" of movies.
2.  **Word2Vec Training:** Train a Word2Vec model (either CBOW or Skip-gram with negative sampling) on this dataset of user viewing histories, treating movies as "words".
3.  **Movie Embeddings:** The resulting word embeddings represent each movie as a vector. Movies with similar audiences or themes will have similar vector representations.
4.  **Recommendation:** To recommend movies to a user, find the movies they have already watched, average their embeddings, and then find other movies with embeddings that are close to the average user embedding.  Distance can be calculated using cosine similarity.

This allows the system to recommend movies that are similar to those the user has enjoyed in the past, even if those movies don't share actors, genres, or keywords directly.

## 3) Python method (if possible)
```python
from gensim.models import Word2Vec
import nltk
from nltk.corpus import brown

# Download the Brown corpus if you haven't already
try:
    brown.words()
except LookupError:
    nltk.download('brown')

# Load the Brown corpus
sentences = brown.sents()

# Train Word2Vec model with negative sampling
model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4, sg=1, negative=10) # sg=1 for Skip-gram, negative=10 for 10 negative samples

# Access word vectors
vector = model.wv['king']
print(vector)

# Find similar words
similar_words = model.wv.most_similar('king', topn=5)
print(similar_words)

# Save and load the model
model.save("word2vec.model")
loaded_model = Word2Vec.load("word2vec.model")
```

**Explanation:**

*   **`gensim.models.Word2Vec`**:  This is the main class for training Word2Vec models in Python.
*   **`sentences`**: The training data. It should be a list of lists, where each inner list represents a sentence (a list of words).  The Brown corpus is used as an example.
*   **`vector_size`**: The dimensionality of the word vectors.
*   **`window`**: The maximum distance between the current and predicted word within a sentence.
*   **`min_count`**: Ignores all words with total frequency lower than this.
*   **`workers`**: Number of worker threads to train the model (parallel processing).
*   **`sg`**:  Training algorithm: 1 for Skip-gram; otherwise, CBOW.
*   **`negative`**: Specifies how many "noise words" should be drawn (usually between 5-20).  If set to 0, no negative sampling is used.
*   **`model.wv['king']`**: Accesses the vector representation of the word "king".
*   **`model.wv.most_similar('king', topn=5)`**: Finds the 5 words that are most similar to "king" based on cosine similarity.
*   **`model.save("word2vec.model")`**: Saves the trained model to disk.
*   **`loaded_model = Word2Vec.load("word2vec.model")`**: Loads a saved model from disk.

## 4) Follow-up question

How does the choice of the number of negative samples affect the quality and training time of the Word2Vec embeddings? What are the trade-offs?