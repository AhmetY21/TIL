---
title: "Distributional Semantics and Word Vectors"
date: "2026-02-17"
week: 8
lesson: 1
slug: "distributional-semantics-and-word-vectors"
---

# Topic: Distributional Semantics and Word Vectors

## 1) Formal definition (what is it, and how can we use it?)

**Distributional Semantics** is a theory of meaning that proposes that words which occur in similar contexts tend to have similar meanings. In other words, "you shall know a word by the company it keeps" (J.R. Firth). This contrasts with other approaches that might focus on defining words through logical relationships or using ontologies. The *distribution* refers to the observed occurrences of a word in relation to other words within a corpus.

**Word Vectors (also known as Word Embeddings)** are numerical representations of words in a high-dimensional space, capturing semantic and syntactic relationships learned from large text corpora based on distributional semantics. Each dimension of the vector corresponds to a specific feature or aspect of the word's meaning, though these dimensions are often not directly interpretable.

**How can we use it?**

*   **Semantic Similarity:** We can calculate the similarity (e.g., cosine similarity) between word vectors to determine how closely related two words are in meaning. This is used for tasks like identifying synonyms, finding related terms, and clustering words into semantic categories.
*   **Word Analogy:** We can perform vector arithmetic to solve analogies like "king - man + woman = queen". This demonstrates that word vectors capture relational information.
*   **Improved NLP Task Performance:** Word vectors are often used as features in various NLP tasks such as text classification, machine translation, question answering, and sentiment analysis, significantly improving their accuracy compared to using one-hot encoded representations.
*   **Information Retrieval:** Word vectors can be used to enhance search results by retrieving documents that contain semantically similar words to the query, even if the exact words are not present.

## 2) Application scenario

Imagine you're building a chatbot for an e-commerce website selling electronics. A user types, "I need a good camera for travel photography."  Without distributional semantics, the chatbot might struggle if the product descriptions don't explicitly use the phrase "travel photography". However, with pre-trained word vectors:

1.  The chatbot converts the user query into word vectors.
2.  It calculates the cosine similarity between the word vector for "travel" and the word vectors in the product descriptions.
3.  It identifies products whose descriptions contain words that are semantically similar to "travel," such as "adventure," "vacation," "outdoors," etc.
4.  It also calculates the cosine similarity between the word vector for "camera" and word vectors in the product descriptions.
5. By combining the similarity scores (e.g., by averaging them), the chatbot can identify cameras that are best suited for travel photography based on the semantic meaning of the query, even if the product descriptions use different wording.

This allows the chatbot to provide more relevant and helpful product recommendations to the user.

## 3) Python method (if possible)

We can use the `gensim` library in Python to work with pre-trained word vectors (e.g., Word2Vec, GloVe, FastText). Here's an example using a pre-trained Word2Vec model:

```python
import gensim.downloader as api
from gensim.models import KeyedVectors

# Load a pre-trained Word2Vec model (this downloads the model if it's not already present)
wv = api.load('word2vec-google-news-300')

# Calculate the cosine similarity between two words
similarity = wv.similarity('king', 'queen')
print(f"Similarity between 'king' and 'queen': {similarity}")

# Find the most similar words to a given word
similar_words = wv.most_similar('travel', topn=5)
print(f"Words most similar to 'travel': {similar_words}")

# Solve an analogy: king - man + woman = ?
result = wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print(f"Analogy result: {result}")

# Check if a word is in the vocabulary
if 'camera' in wv:
    camera_vector = wv['camera']
    print(f"Vector for 'camera': {camera_vector[:10]}...")  # Print the first 10 dimensions
else:
    print("'camera' not found in the vocabulary.")
```

This code snippet demonstrates loading a pre-trained model, calculating word similarity, finding similar words, solving analogies, and accessing a word vector. Remember to install `gensim` first: `pip install gensim`.  The `word2vec-google-news-300` model is large (around 1.6GB), so downloading it may take some time.  Alternatively, you can use smaller models like `glove-wiki-gigaword-100`.

## 4) Follow-up question

How does the choice of corpus used to train word vectors affect their performance on downstream NLP tasks? Specifically, what are the potential benefits and drawbacks of using a very general corpus (e.g., Wikipedia) versus a more specialized corpus (e.g., biomedical literature) for tasks in a specific domain (e.g., clinical text analysis)?