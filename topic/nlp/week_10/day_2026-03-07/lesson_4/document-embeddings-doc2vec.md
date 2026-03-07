---
title: "Document Embeddings (Doc2Vec)"
date: "2026-03-07"
week: 10
lesson: 4
slug: "document-embeddings-doc2vec"
---

# Topic: Document Embeddings (Doc2Vec)

## 1) Formal definition (what is it, and how can we use it?)

Document embeddings, also known as Doc2Vec or Paragraph Vectors, are numerical vector representations of entire documents. Unlike word embeddings which represent individual words, document embeddings capture the semantic meaning of a larger chunk of text, such as a paragraph, document, or article.  They are designed to address the limitations of aggregating word embeddings to represent documents, as simple averaging or summing can lose contextual information and word order significance.

Doc2Vec, proposed by Mikolov et al. (2014), extends the Word2Vec architecture to create these document-level representations. It learns distributed representations of documents by considering the context of words within the document. The core idea is to treat each document as a special "token" alongside the words and learn a vector representation for it.

There are two main variants of Doc2Vec:

*   **Distributed Memory Model of Paragraph Vectors (PV-DM):** Similar to the Continuous Bag-of-Words (CBOW) model in Word2Vec, PV-DM predicts the next word given the document vector and the surrounding words. The document vector acts as a "memory" of the document's context. In effect, it's learning how much each document vector contributes to the context of each word.

*   **Distributed Bag of Words version of Paragraph Vector (PV-DBOW):** Analogous to the Skip-gram model in Word2Vec, PV-DBOW predicts words from the document ID. It essentially aims to predict the words present in the document, given the document vector. This method tends to be faster than PV-DM.

**How can we use them?**

Document embeddings can be used for a wide range of NLP tasks, including:

*   **Document Similarity:** Calculate the similarity between documents by comparing their corresponding vectors (e.g., using cosine similarity). This is useful for recommendation systems, plagiarism detection, and finding related articles.
*   **Document Classification:** Train a classifier using the document embeddings as features. This is beneficial for tasks such as sentiment analysis, topic categorization, and spam detection.
*   **Information Retrieval:**  Embed user queries and documents into the same vector space.  Retrieve documents whose embeddings are closest to the query embedding.
*   **Clustering:** Group documents into clusters based on the similarity of their embeddings. This can be used for topic discovery and identifying different themes within a corpus.
*   **Search Engine Optimization:** Improve search relevance by understanding the semantic meaning of webpages using document embeddings.

## 2) Application scenario

Imagine you are building a customer support system for a large e-commerce company. You have a vast database of customer reviews, questions, and support tickets. A key challenge is to quickly identify similar tickets to a new incoming ticket, allowing agents to easily find relevant solutions and resolve issues faster.

Using Doc2Vec, you can:

1.  Train a Doc2Vec model on the existing database of customer support tickets.
2.  Embed each ticket into a high-dimensional vector space.
3.  When a new ticket arrives, embed it using the same Doc2Vec model.
4.  Calculate the cosine similarity between the new ticket's embedding and the embeddings of all existing tickets.
5.  Identify the tickets with the highest similarity scores. These tickets represent the most relevant and potentially helpful solutions for the new incoming ticket.

This application showcases how Doc2Vec can be used to improve customer support efficiency by enabling rapid identification of similar issues and relevant solutions.

## 3) Python method (if possible)

The `gensim` library in Python provides a convenient way to implement Doc2Vec. Here's an example of training a Doc2Vec model using the PV-DM algorithm:

```python
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt') # Download punkt tokenizer if you don't have it

# Sample data (list of documents)
data = [
    "I love this product. It's amazing!",
    "This is a terrible service. I am very disappointed.",
    "The quality is good, but the price is a bit high.",
    "I am satisfied with the purchase. It arrived on time.",
    "The product is faulty. I want a refund."
]

# Tokenize and tag the documents
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]

# Instantiate and train the Doc2Vec model (PV-DM)
model = Doc2Vec(tagged_data, vector_size=20, window=2, min_count=1, workers=4, dm=1) # dm=1 indicates PV-DM
# Alternatively for PV-DBOW use: model = Doc2Vec(tagged_data, vector_size=20, window=2, min_count=1, workers=4, dm=0)

# Train the model for multiple epochs
for epoch in range(100):
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

# Save the model (optional)
model.save("doc2vec_model.model")

# Load a trained model
# model = Doc2Vec.load("doc2vec_model.model")

# Infer the vector for a new document
new_doc = "I hate this product."
new_vector = model.infer_vector(word_tokenize(new_doc.lower()))

# Print the vector
print(new_vector)

# Find similar documents
similar_doc = model.dv.most_similar([new_vector], topn=3)
print(similar_doc)
```

**Explanation:**

1.  **Import necessary libraries:** `gensim` for Doc2Vec, `nltk` for tokenization.
2.  **Sample Data:**  A list of example documents is created.
3.  **Tokenize and Tag:** Each document is tokenized (split into words) using `word_tokenize` and tagged using `TaggedDocument`.  The `tags` are unique identifiers for each document, essential for Doc2Vec.
4.  **Instantiate the Model:** A `Doc2Vec` model is initialized with parameters like `vector_size` (dimensionality of the document vectors), `window` (context window size), `min_count` (minimum frequency of a word), `workers` (number of cores), and `dm` (set to 1 for PV-DM and 0 for PV-DBOW).
5.  **Train the Model:** The model is trained using the `train` method, iterating over the tagged documents for multiple epochs.
6.  **Save and Load Model (Optional):** This lets you reuse the trained model later without re-training.
7.  **Infer Vector:**  `model.infer_vector()` generates a vector representation for a new, unseen document.
8.  **Find Similar Documents:**  `model.dv.most_similar()` finds the documents in the training data most similar to the inferred vector of the new document. It returns a list of tuples, where each tuple contains the document ID and its similarity score.

## 4) Follow-up question

How can the performance of a Doc2Vec model be evaluated, and what techniques can be used to improve its accuracy in downstream tasks like document classification?