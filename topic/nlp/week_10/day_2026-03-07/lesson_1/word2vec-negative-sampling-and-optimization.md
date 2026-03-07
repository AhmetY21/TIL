---
title: "Word2Vec: Negative Sampling and Optimization"
date: "2026-03-07"
week: 10
lesson: 1
slug: "word2vec-negative-sampling-and-optimization"
---

# Topic: Word2Vec: Negative Sampling and Optimization

## 1) Formal definition (what is it, and how can we use it?)

**Word2Vec** is a popular technique for learning word embeddings, which are dense vector representations of words in a vocabulary. These embeddings capture semantic relationships between words, allowing us to perform tasks like finding similar words, analogy completion (e.g., "king is to queen as man is to woman"), and improving the performance of downstream NLP models.

**Negative Sampling** is an optimization technique used within Word2Vec to make the training process more efficient. Standard Word2Vec algorithms (like Skip-gram and CBOW) require calculating the probability of a word appearing in a context based on the entire vocabulary, which can be computationally expensive, especially for large vocabularies.  Negative sampling simplifies this by converting the multi-class classification problem into a binary classification problem.

Here's how it works within the **Skip-gram** architecture (the most common example):

*   **Skip-gram Goal:** Given a target word, predict the surrounding context words.

*   **The Issue:**  Originally, Skip-gram would try to maximize the probability of observing the actual context words given the target word, which requires normalizing probabilities across the entire vocabulary.

*   **Negative Sampling Approach:** Instead of trying to predict *all* context words, negative sampling trains the model to discriminate between *actual* context words (positive samples) and randomly chosen "noise" words (negative samples).

    *   For each target-context word pair (positive sample), we randomly select a small number of words from the vocabulary that *did not* appear in the context of the target word (negative samples). The number of negative samples is a hyperparameter, usually between 5 and 20.

    *   The model then learns to predict '1' for the positive samples (actual context words) and '0' for the negative samples. This transforms the problem into a series of binary classification tasks.

**Optimization:**  Negative sampling enables the use of simpler and more efficient optimization algorithms (like stochastic gradient descent - SGD) because we are only dealing with a small set of positive and negative examples during each iteration, rather than the entire vocabulary. This drastically reduces the computational cost of training, especially with large vocabularies.

**How can we use it?**

*   **Generate Word Embeddings:** Train a Word2Vec model (using negative sampling for efficiency) on a large corpus of text. The resulting word vectors can then be used as features in other NLP tasks.
*   **Semantic Similarity:** Calculate the cosine similarity between word vectors to find words that are semantically related.
*   **Analogy Completion:** Solve analogy problems like "A is to B as C is to D" by using vector arithmetic (e.g., vector(B) - vector(A) + vector(C) should be close to vector(D)).
*   **Input to other models:** Use pre-trained word embeddings as input to more complex models, like LSTMs or Transformers, to improve performance on tasks like sentiment analysis, text classification, and machine translation.

## 2) Application scenario

Let's say you are building a sentiment analysis model for customer reviews of electronic products. Your vocabulary is quite large (e.g., 100,000 words).

Without negative sampling, training a Word2Vec model to generate word embeddings from your review data would be very slow because each update requires calculations over the entire vocabulary. This makes training on a large corpus of product reviews extremely time consuming.

By using negative sampling, you can significantly speed up the training process. For each review, you can create positive examples of target-context word pairs and then randomly sample a small number of negative words (e.g., 10 words) that don't appear in the immediate context. This reduces the computational burden and allows you to train the Word2Vec model much faster, enabling you to get better word embeddings and, consequently, a better sentiment analysis model.  You can then use these embeddings as input to a more complex classification model like a neural network for improved sentiment classification accuracy.
## 3) Python method (if possible)

```python
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# Sample corpus (list of sentences, where each sentence is a list of words)
corpus = [
    "This is the first sentence in the corpus.",
    "This is the second sentence.",
    "And another sentence.  This one is third in the series.",
    "Is this the end of the corpus?",
]

# Preprocess the corpus (tokenize and lowercase)
tokenized_corpus = [simple_preprocess(sentence) for sentence in corpus]

# Train Word2Vec model with negative sampling
model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4, sg=1, negative=5)

# Parameters explanation:
# - sentences:  The corpus to train on (list of lists of words)
# - vector_size: Dimensionality of the word vectors (embeddings)
# - window: Maximum distance between the current and predicted word within a sentence.
# - min_count: Ignores all words with total frequency lower than this.
# - workers: Use these many worker threads to train the model (=faster training with multicore machines).
# - sg: Training algorithm: 1 for skip-gram; otherwise CBOW.
# - negative: If > 0, negative sampling will be used, the int for negative specifies how many "noise words" should be drawn (usually between 5-20).
# Access word vectors
word_vectors = model.wv

# Get the vector for a specific word
vector_of_word = word_vectors['sentence']
print(f"Vector for 'sentence': {vector_of_word}")

# Find similar words
similar_words = model.wv.most_similar('sentence', topn=3)
print(f"Words similar to 'sentence': {similar_words}")

# Save the model
model.save("word2vec.model")

# Load the model
loaded_model = Word2Vec.load("word2vec.model")
```

## 4) Follow-up question

How does the number of negative samples affect the quality of the resulting word embeddings and the training time?  What are some strategies for choosing the optimal number of negative samples?  Also, does the size of the corpus influence the optimal number of negative samples?