---
title: "GloVe (Global Vectors for Word Representation)"
date: "2026-03-07"
week: 10
lesson: 2
slug: "glove-global-vectors-for-word-representation"
---

# Topic: GloVe (Global Vectors for Word Representation)

## 1) Formal definition (what is it, and how can we use it?)

GloVe (Global Vectors for Word Representation) is an unsupervised learning algorithm for obtaining vector representations of words. Unlike word2vec, which is a "local context" method (trained on predicting a word given its neighbors), GloVe is a "global" method that leverages global word-word co-occurrence statistics from a corpus.

Specifically, GloVe aims to learn word vectors such that their dot product equals the logarithm of the words' probability of co-occurrence.  Let:

*   *X<sub>ij</sub>* be the number of times word *j* occurs in the context of word *i*.
*   *X<sub>i</sub>* = ∑<sub>k</sub> *X<sub>ik</sub>* be the number of times any word appears in the context of word *i*.
*   *P<sub>ij</sub>* = *P(j | i)* = *X<sub>ij</sub> / X<sub>i</sub>* be the probability of word *j* appearing in the context of word *i*.

The core idea is that the ratio of co-occurrence probabilities for two words *i* and *j* relative to a "probe" word *k* should reveal something meaningful about the relationship between words *i* and *j*.  The GloVe model is based on the following cost function:

J = ∑<sub>i,j=1</sub><sup>V</sup> *f(X<sub>ij</sub>) (w<sub>i</sub><sup>T</sup> w<sub>j</sub> + b<sub>i</sub> + b<sub>j</sub> - log X<sub>ij</sub>)<sup>2</sup>*

where:

*   *V* is the size of the vocabulary.
*   *w<sub>i</sub>* and *w<sub>j</sub>* are the word vectors for words *i* and *j*.
*   *b<sub>i</sub>* and *b<sub>j</sub>* are bias terms for words *i* and *j*.
*   *f(X<sub>ij</sub>)* is a weighting function that helps to avoid overfitting to frequent word pairs and underfitting to rare word pairs.  A common form for *f(x)* is:

    *f(x)* = (x/x<sub>max</sub>)<sup>α</sup> if x < x<sub>max</sub>, else 1

    where *x<sub>max</sub>* and *α* are hyperparameters.

**How can we use it?**

GloVe vectors, once trained, represent words as dense vectors in a high-dimensional space. These vectors can be used in various NLP tasks, including:

*   **Word similarity:** Measuring the cosine similarity between word vectors can indicate how semantically similar two words are.
*   **Analogy tasks:** Solving analogies like "king - man + woman = ?" by performing vector arithmetic (v("king") - v("man") + v("woman")).  The closest word to the resulting vector is often "queen".
*   **Downstream tasks:** GloVe vectors can be used as input features for machine learning models in tasks like text classification, sentiment analysis, named entity recognition, and machine translation.
*   **Visualization:** Reducing the dimensionality of GloVe vectors (e.g., using t-SNE or PCA) allows for visualizing the semantic relationships between words in a 2D or 3D space.

## 2) Application scenario

Imagine you're building a sentiment analysis system for movie reviews. You want to classify reviews as either positive or negative. You could use GloVe word embeddings to represent each word in the review as a vector. You can then aggregate these word vectors (e.g., by averaging or summing) to create a representation of the entire review. This aggregated vector can then be used as input to a classifier like a logistic regression model or a neural network. By using GloVe embeddings, the system can leverage pre-trained semantic knowledge about words, improving its ability to understand the sentiment expressed in the reviews, even for words it hasn't seen frequently during training. For instance, it might understand "fantastic" and "amazing" are semantically close and convey positive sentiment.

## 3) Python method (if possible)
While you can implement GloVe from scratch, using existing libraries is much more practical. Here's how you can use pre-trained GloVe embeddings in Python using the `gensim` library:

```python
import gensim.downloader as api
import numpy as np

# Load pre-trained GloVe vectors (e.g., glove-wiki-gigaword-50)
glove_model = api.load("glove-wiki-gigaword-50")  # 50-dimensional vectors

# Get the vector for a word
word_vector = glove_model["king"]
print(f"Vector for 'king': {word_vector}")

# Calculate cosine similarity between two words
similarity = glove_model.similarity("king", "queen")
print(f"Similarity between 'king' and 'queen': {similarity}")

# Perform an analogy (king - man + woman = ?)
result = glove_model.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
print(f"Analogy result: {result}")

# Example: Using GloVe for sentence embedding (averaging word vectors)
def sentence_vector(sentence, model):
    words = sentence.split()
    vectors = [model[word] for word in words if word in model]  # Handle out-of-vocabulary words
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)  # Return a zero vector if no words are in the model

sentence1 = "The weather is great today."
sentence2 = "It is raining cats and dogs."

vector1 = sentence_vector(sentence1, glove_model)
vector2 = sentence_vector(sentence2, glove_model)

print(f"Sentence vector 1: {vector1}")
print(f"Sentence vector 2: {vector2}")

# Calculate cosine similarity between sentences
from sklearn.metrics.pairwise import cosine_similarity
sentence_similarity = cosine_similarity([vector1], [vector2])[0][0]
print(f"Similarity between sentences: {sentence_similarity}")


# Note: Training your own GloVe model from scratch requires more code and typically involves libraries like scikit-learn or writing custom code. Libraries like `glove-python` or `textacy` are used for training.

```

## 4) Follow-up question

How do the hyperparameters of the GloVe model, especially `x_max` and `α` in the weighting function *f(x)*, affect the quality and characteristics of the resulting word embeddings?  Specifically, what are the trade-offs involved in choosing different values for these parameters, and how might one select appropriate values for a given dataset and task?