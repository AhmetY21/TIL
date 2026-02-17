---
title: "Introduction to Word2Vec (CBOW vs Skip-gram)"
date: "2026-02-17"
week: 8
lesson: 2
slug: "introduction-to-word2vec-cbow-vs-skip-gram"
---

# Topic: Introduction to Word2Vec (CBOW vs Skip-gram)

## 1) Formal definition (what is it, and how can we use it?)

Word2Vec is a group of related models used to produce word embeddings. Word embeddings are vector representations of words that capture semantic meaning. Unlike traditional one-hot encoding, word embeddings represent words in a continuous, high-dimensional space where semantically similar words are located closer to each other.  This allows algorithms to understand relationships between words, such as synonymy and analogy.

There are two main architectures within Word2Vec:

*   **Continuous Bag of Words (CBOW):** CBOW predicts a target word given the context words around it. The model takes the context words as input and tries to predict the missing word in the center of the window. Essentially, it averages the embeddings of the context words to predict the target word.

*   **Skip-gram:** Skip-gram is the opposite of CBOW. It predicts the surrounding context words given a target word. It takes a word as input and tries to predict the words that surround it. This often performs better than CBOW, especially with small datasets and rare words, because it provides more training samples for each word.

Both models are shallow neural networks trained using either Hierarchical Softmax or Negative Sampling to make the training process more efficient. They can be used to:

*   **Improve NLP tasks:** Provide meaningful word representations for downstream tasks like sentiment analysis, machine translation, and text classification.
*   **Word Similarity:** Determine the similarity between words by calculating the cosine similarity between their corresponding vectors.
*   **Analogy Reasoning:** Solve analogy questions (e.g., "king is to man as queen is to ?").

## 2) Application scenario

Consider a search engine.  Instead of simply matching keywords, Word2Vec allows the search engine to understand the *meaning* of the user's query.

*   **Scenario:** A user searches for "big vehicle".
*   **Without Word2Vec:** The search engine might only return results containing the exact phrase "big vehicle".
*   **With Word2Vec:**  The search engine can leverage word embeddings to understand that "big" is semantically similar to "large," and "vehicle" is similar to "car" or "truck."  It can then return results containing phrases like "large car" or "huge truck," even if those exact words weren't present in the original search query.

This leads to more relevant and comprehensive search results, improving the user experience. Word2Vec can also be used for recommendation systems, sentiment analysis (to identify positive, negative, or neutral sentiment in text), and other various NLP applications.

## 3) Python method (if possible)

We can use the `gensim` library in Python to train Word2Vec models.  Here's an example:

```python
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# Sample corpus (list of sentences, where each sentence is a list of words)
corpus = [
    "I love natural language processing",
    "Word embeddings are cool",
    "Natural language is fascinating",
    "Processing language is fun"
]

# Preprocess the text
tokenized_corpus = [simple_preprocess(doc) for doc in corpus]

# Train a CBOW Word2Vec model
model_cbow = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, sg=0) # sg=0 for CBOW

# Train a Skip-gram Word2Vec model
model_skipgram = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, sg=1) # sg=1 for Skip-gram


# Access the vector for a word
vector_cbow = model_cbow.wv['language']
vector_skipgram = model_skipgram.wv['language']

# Find the most similar words
similar_words_cbow = model_cbow.wv.most_similar('language', topn=3)
similar_words_skipgram = model_skipgram.wv.most_similar('language', topn=3)


print("CBOW Vector for 'language':", vector_cbow)
print("Skip-gram Vector for 'language':", vector_skipgram)
print("CBOW Similar words to 'language':", similar_words_cbow)
print("Skip-gram Similar words to 'language':", similar_words_skipgram)
```

**Explanation:**

*   `Word2Vec(sentences, vector_size, window, min_count, sg)`: Initializes the Word2Vec model.
    *   `sentences`: The training corpus (a list of tokenized sentences).
    *   `vector_size`:  The dimensionality of the word vectors (e.g., 100).
    *   `window`: The maximum distance between the current and predicted word within a sentence.
    *   `min_count`: Ignores all words with total frequency lower than this.
    *   `sg`:  Training algorithm: 1 for Skip-gram; otherwise CBOW.
*   `model.wv['word']`:  Accesses the vector representation of the word.
*   `model.wv.most_similar('word', topn)`: Returns the `topn` most similar words to the given word.

## 4) Follow-up question

What are some of the limitations of Word2Vec, and what are some alternative word embedding techniques that address these limitations? For example, how does GloVe compare to Word2Vec, and what are the pros and cons of each?