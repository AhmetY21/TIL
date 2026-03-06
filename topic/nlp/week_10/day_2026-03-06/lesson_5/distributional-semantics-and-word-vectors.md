---
title: "Distributional Semantics and Word Vectors"
date: "2026-03-06"
week: 10
lesson: 5
slug: "distributional-semantics-and-word-vectors"
---

# Topic: Distributional Semantics and Word Vectors

## 1) Formal definition (what is it, and how can we use it?)

Distributional semantics is a theory and methodology in natural language processing (NLP) that argues that the meaning of a word can be inferred from the contexts in which it appears. The core idea is encapsulated in the famous quote attributed to J.R. Firth: "You shall know a word by the company it keeps."

Formally, distributional semantics represents words as vectors in a high-dimensional space, where each dimension corresponds to a context. These context dimensions can be defined in various ways, such as:

*   **Word co-occurrences:** How often a word appears near other words.
*   **Document co-occurrences:** Whether a word appears in the same documents as other words.
*   **Syntactic dependencies:** The grammatical relationships a word has with other words in a sentence.

Word vectors (also called word embeddings) are these numerical representations derived from distributional semantics. They capture semantic relationships between words based on their usage. Words with similar meanings are expected to have vectors that are close to each other in the vector space, as measured by distance metrics like cosine similarity.

**How can we use it?**

*   **Semantic Similarity:**  Determine how similar two words are in meaning.
*   **Word Analogy:**  Solve analogy questions like "man is to king as woman is to ?", by performing vector arithmetic (e.g., `king - man + woman` should yield a vector close to `queen`).
*   **Text Classification:**  Use word vectors as features to train machine learning models for tasks like sentiment analysis or topic categorization.
*   **Information Retrieval:** Improve search relevance by matching queries to documents based on semantic similarity rather than just keyword matching.
*   **Machine Translation:**  Map words in one language to corresponding words in another language based on their vector representations.
*   **Question Answering:**  Understand the semantic relationships between the question and the answer.
*   **Neural Networks:** Word embeddings are commonly used as the initial input layer for neural networks in NLP tasks. They allow the network to learn richer representations than just using one-hot encoded vectors.

## 2) Application scenario

Consider a search engine. Instead of just matching keywords directly, we can use distributional semantics to improve search results. Suppose a user searches for "large dog."  Without distributional semantics, the search engine might only return documents containing the exact phrase "large dog."

With distributional semantics, we can:

1.  Retrieve word vectors for "large" and "dog."
2.  Find other words with vectors similar to "large" (e.g., "big," "huge," "giant") and "dog" (e.g., "canine," "puppy," "hound").
3.  Expand the search query to include these related words. Now the search engine might also retrieve documents containing phrases like "big canine" or "giant hound," leading to more relevant results that capture the user's intended meaning, even if the exact phrase "large dog" isn't present.

Another application scenario is in recommender systems. Imagine recommending books to a user.  If the user enjoyed a book with many occurrences of words like "adventure," "fantasy," and "magic," the system can recommend other books containing words with similar vector representations, even if those other books are written by different authors and have entirely different plots.

## 3) Python method (if possible)

Here's how you can use the `gensim` library in Python to create word embeddings using Word2Vec:

```python
from gensim.models import Word2Vec
from nltk.corpus import brown
import nltk
nltk.download('brown') # Download the Brown Corpus if you haven't already

# Load the Brown Corpus (a common text corpus for NLP)
sentences = brown.sents()

# Train a Word2Vec model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4)
# vector_size: dimensionality of the word vectors
# window: maximum distance between the current and predicted word within a sentence
# min_count: ignores all words with total frequency lower than this
# workers: use this many worker threads to train the model (=faster training with multicore machines)

# Save the model (optional)
model.save("brown_word2vec.model")

# Load a pre-trained model (optional)
# model = Word2Vec.load("brown_word2vec.model")

# Get the vector for a word
vector = model.wv['computer']
print(f"Vector for 'computer': {vector[:10]}...")  # Print first 10 elements

# Find the most similar words to a word
similar_words = model.wv.most_similar('computer', topn=5)
print(f"Words similar to 'computer': {similar_words}")

# Perform a word analogy
result = model.wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print(f"woman is to king as man is to: {result}")
```

**Explanation:**

1.  **Import Libraries:**  We import `Word2Vec` from `gensim.models` and `brown` from `nltk.corpus` (and nltk to download brown corpus).
2.  **Load Data:** We load sentences from the Brown Corpus.
3.  **Train the Model:** We create a `Word2Vec` model instance and train it on the sentences. The `vector_size` parameter controls the dimensionality of the word vectors. `window` sets the size of the context window.  `min_count` ignores infrequent words.
4.  **Access Vectors:** We can access the word vector for a specific word using `model.wv['word']`.
5.  **Find Similar Words:**  The `most_similar()` method returns a list of the most similar words to a given word, based on cosine similarity.
6.  **Word Analogy:** We can perform word analogies using `most_similar()` with `positive` and `negative` lists.

**Important Notes:**

*   This example uses the Brown Corpus, which is relatively small. For better word embeddings, train on much larger datasets (e.g., Wikipedia, Common Crawl).
*   Word2Vec is just one algorithm for creating word embeddings. Other popular algorithms include GloVe and FastText.
*   Pre-trained word embeddings (trained on massive datasets) are often available for download. Using pre-trained embeddings can save significant training time and often yields better results, especially when working with smaller datasets. Common sources for pre-trained embeddings include the Stanford NLP group (GloVe) and Facebook (FastText).

## 4) Follow-up question

How do different word embedding algorithms (Word2Vec, GloVe, FastText) compare in terms of their underlying mechanisms, strengths, and weaknesses, and when might one algorithm be preferred over another?