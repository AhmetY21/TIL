---
title: "Limitations of Static Word Embeddings"
date: "2026-02-19"
week: 8
lesson: 2
slug: "limitations-of-static-word-embeddings"
---

# Topic: Limitations of Static Word Embeddings

## 1) Formal definition (what is it, and how can we use it?)

Static word embeddings are a type of word representation where each word is mapped to a single, fixed vector. This vector is learned from a large corpus of text using techniques like Word2Vec (Skip-gram, CBOW), GloVe, or FastText. The underlying assumption is that words appearing in similar contexts will have similar vector representations. Once trained, the embedding for a particular word remains constant, regardless of the specific context in which it appears.

**What is it?** A static word embedding is a fixed vector representation for each word in a vocabulary. These vectors are usually high-dimensional (e.g., 100-300 dimensions) and capture semantic and syntactic relationships between words based on their co-occurrence statistics in the training data.

**How can we use it?** Static word embeddings can be used as input features for various NLP tasks such as:

*   **Text Classification:** Representing documents as the average or weighted sum of their word embeddings.
*   **Sentiment Analysis:** Providing semantic context for sentiment classification models.
*   **Machine Translation:** Helping align words and phrases between different languages.
*   **Information Retrieval:** Calculating semantic similarity between queries and documents.
*   **Named Entity Recognition:** Identifying and classifying named entities based on surrounding words.
*   **Word Similarity and Analogy Tasks:** Evaluating the quality of the embeddings by checking if they can capture word relationships (e.g., "king - man + woman" should be close to "queen").

The core idea is that by replacing discrete words with dense, continuous vectors, we can leverage geometric operations (e.g., cosine similarity) to measure semantic relatedness and improve the performance of various NLP models.

## 2) Application scenario

Consider the word "bank". "Bank" can refer to a financial institution or the edge of a river. A static word embedding will represent "bank" with a single vector that attempts to capture both meanings. This can lead to ambiguity and inaccurate representations in downstream tasks.

**Scenario:** Imagine you are building a sentiment analysis system for financial news articles. If the word "bank" appears in a sentence like "The bank is facing financial difficulties," the system needs to understand that "bank" refers to a financial institution. However, if the word "bank" appears in a sentence like "The river bank was eroding," the system shouldn't associate negative sentiment with the *physical* bank. A static word embedding struggles to distinguish between these two senses, potentially leading to incorrect sentiment predictions. The system will treat both usages as semantically equivalent, averaging out the meanings and diluting the signal. This means that if the word embedding associated with "bank" has a slight negative sentiment learned from instances about financial issues, that negative sentiment could incorrectly get applied to the sentence about the riverbank eroding.

Other scenarios where static embeddings are problematic include handling:

*   **Rare Words:** Words that occur infrequently in the training data often have poorly learned embeddings.
*   **Out-of-Vocabulary Words (OOV):** Words that are not present in the training data cannot be represented at all.
*   **Evolving Language:** The meaning of words can change over time. Static embeddings cannot adapt to these changes.

## 3) Python method (if possible)

```python
import gensim.downloader as api
from gensim.models import KeyedVectors
import numpy as np

# Load pre-trained word embeddings (e.g., word2vec-google-news-300)
try:
    word_vectors = api.load("word2vec-google-news-300") # or glove-wiki-gigaword-100
except:
    print("Failed to download embeddings. Please check your internet connection.")
    word_vectors = None

if word_vectors:
    # Example demonstrating the limitation of a static embedding for "bank"
    bank_financial = word_vectors["bank"]
    bank_river = word_vectors["bank"] # Same vector!

    # Let's assume we have a simplified sentiment lexicon:
    sentiment_words = {"good": 0.8, "bad": -0.7, "difficulties": -0.5, "eroding": -0.3}

    # Sentence 1: "The bank is facing financial difficulties."
    sentence1_words = ["the", "bank", "is", "facing", "financial", "difficulties"]
    sentence1_embedding = np.mean([word_vectors[word] if word in word_vectors else np.zeros(300) for word in sentence1_words], axis=0)
    sentence1_sentiment = sum([sentiment_words.get(word, 0) for word in sentence1_words]) #Simple lexicon score

    # Sentence 2: "The river bank was eroding."
    sentence2_words = ["the", "river", "bank", "was", "eroding"]
    sentence2_embedding = np.mean([word_vectors[word] if word in word_vectors else np.zeros(300) for word in sentence2_words], axis=0)
    sentence2_sentiment = sum([sentiment_words.get(word, 0) for word in sentence2_words]) #Simple lexicon score

    #Due to static embeddings, sentence1_embedding and sentence2_embedding contribute the exact same vector for the word bank
    # even though the sentiment should be negative for sentence 1 but potentially neutral for sentence 2.
    print(f"Sentence 1 Sentiment Score (lexicon): {sentence1_sentiment}") # -0.5
    print(f"Sentence 2 Sentiment Score (lexicon): {sentence2_sentiment}") # -0.3 - This is less negative than reality.

    #The embedding for bank is the *same* for both sentences, even if context is different.
    print(f"Cosine similarity between the embedding of 'bank' with 'money' in embedding space: {word_vectors.similarity('bank','money')}")
    print(f"Cosine similarity between the embedding of 'bank' with 'river' in embedding space: {word_vectors.similarity('bank','river')}")

```

**Explanation:**

1.  **Loading Pre-trained Embeddings:** The code uses `gensim` to load pre-trained Word2Vec embeddings. These embeddings are trained on a large corpus of text and capture semantic relationships between words.
2.  **Demonstrating the Limitation:** It shows that regardless of context, the word "bank" gets the same vector representation.
3.  **Sentiment Analysis Example:** A basic sentiment analysis using sentiment lexicon and averaging embedding vector. The code shows the same vector for "bank" is used for both sentences, leading to an imprecise sentiment analysis score.
4.  **Similarity Check:** The code also shows cosine similarity to words like 'money' and 'river' to show the lack of context consideration.

## 4) Follow-up question

How do contextualized word embeddings (e.g., BERT, ELMo) address the limitations of static word embeddings, and what are the trade-offs involved in using contextualized embeddings instead of static embeddings? Explain with an example of how BERT would solve the "bank" problem above.