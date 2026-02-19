---
title: "Limitations of Static Word Embeddings"
date: "2026-02-19"
week: 8
lesson: 1
slug: "limitations-of-static-word-embeddings"
---

# Topic: Limitations of Static Word Embeddings

## 1) Formal definition (what is it, and how can we use it?)

Static word embeddings are pre-trained vector representations of words where each word is mapped to a single, fixed vector in a high-dimensional space. These vectors are typically learned from large corpora of text using algorithms like Word2Vec (skip-gram or CBOW), GloVe, or FastText. The core idea is that words appearing in similar contexts will have similar vector representations, capturing semantic relationships.

How can we use them?

*   **Semantic similarity:**  By calculating cosine similarity or other distance metrics between word vectors, we can estimate the semantic similarity between words. Words like "king" and "queen" will have higher similarity than "king" and "apple."
*   **Analogical reasoning:**  Word embeddings can be used to solve analogy problems like "man is to king as woman is to queen".  We can find the vector closest to *vector("king") - vector("man") + vector("woman")*.
*   **Input features for NLP tasks:** Word embeddings can be used as input features for various downstream NLP tasks like text classification, sentiment analysis, machine translation, and named entity recognition. They provide a dense, low-dimensional representation of words compared to sparse one-hot encodings.  They allow models to generalize better to unseen words based on semantic similarity.
*   **Word Clustering:** Word embeddings can be clustered to group similar words, which can then be used for tasks such as creating thesauruses or topic modeling.

However, a fundamental limitation is that these embeddings are *static*. A word has the *same* vector representation regardless of the context in which it appears. This means that the word "bank" will have the same vector whether it refers to a financial institution or the bank of a river. This is a major drawback because word meanings are often context-dependent.

## 2) Application scenario

Imagine you are building a sentiment analysis model for customer reviews. You have reviews for a new restaurant. Consider these two sentences:

*   "The food was great, but the service was *slow*." (Negative connotation of "slow")
*   "The internet connection was *slow*, but they offered to compensate for it." (Less negative, possibly neutral connotation of "slow").

Using static word embeddings, the word "slow" will have the same vector representation in both sentences. This means the sentiment analysis model will struggle to differentiate between the nuances in meaning conveyed by the context. The model might incorrectly classify the second sentence as more negative than it actually is because of the negative association of "slow" in its pre-trained embedding.

Another scenario is handling polysemous words like "bank". A question answering system using static embeddings might provide incorrect answers if the question involves differentiating between the river bank and the financial institution bank, as the same embedding would be used for both.

## 3) Python method (if possible)

Using the `gensim` library, you can load a pre-trained Word2Vec model and access word vectors. This demonstrates how static embeddings work, but also highlights the static nature.

```python
import gensim.downloader as api
from gensim.models import KeyedVectors

# Download a pre-trained Word2Vec model (small size for demonstration)
try:
    word_vectors = api.load("glove-twitter-25")  # can also be word2vec-google-news-300
except:
    print("Download failed. Please check your internet connection")
    exit()

# Get the vector representation of the word "bank"
vector_bank = word_vectors["bank"]
print("Vector representation of 'bank':", vector_bank[:10])  # Print the first 10 dimensions

# Check the similarity between "bank" and "river" and "bank" and "finance"
similarity_river = word_vectors.similarity("bank", "river")
similarity_finance = word_vectors.similarity("bank", "finance")

print("Similarity between 'bank' and 'river':", similarity_river)
print("Similarity between 'bank' and 'finance':", similarity_finance)

#Demonstrating that the vector for 'bank' is the same regardless of context
vector_bank_context1 = word_vectors["bank"]
vector_bank_context2 = word_vectors["bank"]

print("Are the bank vectors the same in both context? : ", (vector_bank_context1 == vector_bank_context2).all())


```

This code snippet demonstrates that the vector for "bank" remains the same regardless of the context.  The similarity scores show the model captures *some* semantic relationships, but it cannot distinguish between different meanings of "bank" based on context. The fact that `vector_bank_context1` and `vector_bank_context2` are identical confirms the static nature.

## 4) Follow-up question

How do *contextualized* word embeddings (like those produced by BERT, RoBERTa, or ELMo) address the limitations of static word embeddings, and what are their own associated limitations?