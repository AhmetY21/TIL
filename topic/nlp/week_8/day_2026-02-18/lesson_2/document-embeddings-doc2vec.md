---
title: "Document Embeddings (Doc2Vec)"
date: "2026-02-18"
week: 8
lesson: 2
slug: "document-embeddings-doc2vec"
---

# Topic: Document Embeddings (Doc2Vec)

## 1) Formal definition (what is it, and how can we use it?)

Document embeddings (also known as Doc2Vec or Paragraph Vector) are vector representations of entire documents. Unlike word embeddings which represent individual words, doc embeddings capture the semantic meaning of a document as a whole. Doc2Vec is an unsupervised learning technique that learns fixed-length feature representations from variable-length pieces of texts, such as sentences, paragraphs, or entire documents.

There are two main architectures for Doc2Vec:

*   **Distributed Memory Model of Paragraph Vectors (PV-DM):**  Similar to Word2Vec's Continuous Bag-of-Words (CBOW) model.  PV-DM predicts a target word given the context words and a document ID. The document ID is treated as another word and contributes to the prediction. The model learns to associate a vector with each document ID, representing the document's meaning. In essence, the document vector acts as a memory that remembers what the document is missing.

*   **Distributed Bag of Words version of Paragraph Vector (PV-DBOW):** Similar to Word2Vec's Skip-gram model. PV-DBOW predicts words in a document based solely on the document ID. The model is trained to predict words randomly sampled from the document given the document vector.  This model typically works better for larger datasets.

We can use document embeddings in several ways:

*   **Document Similarity:**  Calculate the cosine similarity between the document embeddings to find documents that are semantically similar.
*   **Document Classification:**  Use the document embeddings as features in a classification model to predict the category or label of a document.
*   **Information Retrieval:**  Embed a search query and compare it to embeddings of documents in a corpus to retrieve relevant documents.
*   **Clustering:**  Group documents based on the similarity of their document embeddings.
*   **Recommendation Systems:** Recommend documents based on their similarity to documents a user has interacted with.

## 2) Application scenario

Imagine you're building a system to categorize customer support tickets. You have a large dataset of past tickets, each containing a text description of the customer's issue.  Using Doc2Vec, you can:

1.  Train a Doc2Vec model on your historical ticket data. This will create a document embedding for each ticket.
2.  For a new, incoming ticket, you can infer its document embedding using the trained Doc2Vec model.
3.  Compare the new ticket's embedding to the embeddings of past tickets.
4.  Based on the similarity, you can automatically categorize the new ticket (e.g., "Billing Issue", "Technical Support", "Order Inquiry"). This can save time and improve the efficiency of your support team.  Furthermore, you could route the ticket to an appropriate support agent based on historical patterns of handling similar ticket types.

Another scenario:

You are building a news aggregation website. You want to cluster news articles based on their topic, even if they don't share many of the same keywords. By using Doc2Vec to create document embeddings, you can cluster articles based on semantic similarity rather than just keyword overlap. This can lead to more relevant and coherent news clusters.

## 3) Python method (if possible)

We can use the `gensim` library in Python to implement Doc2Vec.

```python
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

# Sample documents
documents = [
    "This is the first document about NLP.",
    "The second document discusses machine learning techniques.",
    "Document number three focuses on deep learning models.",
    "This document explores natural language processing applications."
]

# Tokenize the documents and tag them
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(documents)]

# Initialize and train the Doc2Vec model (PV-DM)
model_dm = Doc2Vec(tagged_data, vector_size=20, window=2, min_count=1, epochs=100, dm=1) # dm=1 for PV-DM

# Initialize and train the Doc2Vec model (PV-DBOW)
model_dbow = Doc2Vec(tagged_data, vector_size=20, window=2, min_count=1, epochs=100, dm=0) # dm=0 for PV-DBOW

# Infer vector for a new document
new_doc = "New document about text analysis."
new_vector_dm = model_dm.infer_vector(word_tokenize(new_doc.lower()))
new_vector_dbow = model_dbow.infer_vector(word_tokenize(new_doc.lower()))

# Print the vector
print("PV-DM Vector:", new_vector_dm)
print("PV-DBOW Vector:", new_vector_dbow)

# Find similar documents based on PV-DM
similar_doc_dm = model_dm.dv.most_similar([new_vector_dm], topn=len(documents))
print("PV-DM Similar Documents:", similar_doc_dm)

# Find similar documents based on PV-DBOW
similar_doc_dbow = model_dbow.dv.most_similar([new_vector_dbow], topn=len(documents))
print("PV-DBOW Similar Documents:", similar_doc_dbow)

# Save and load the model
model_dm.save("doc2vec_model_dm.model")
loaded_model_dm = Doc2Vec.load("doc2vec_model_dm.model")

model_dbow.save("doc2vec_model_dbow.model")
loaded_model_dbow = Doc2Vec.load("doc2vec_model_dbow.model")
```

**Explanation:**

1.  **Import Libraries:** Import necessary libraries: `Doc2Vec`, `TaggedDocument` from `gensim`, `word_tokenize` from `nltk`, and `nltk`. The nltk package needs to download the punkt tokenizer models.
2.  **Prepare Data:**  Create a list of `TaggedDocument` objects. Each `TaggedDocument` consists of a list of tokens (words) and a list of tags (document IDs).  Here, we use the index of the document as the tag.  The text is converted to lowercase and tokenized.
3.  **Initialize and Train:** Initialize the `Doc2Vec` model.
    *   `vector_size`: Dimensionality of the document vectors.
    *   `window`: Maximum distance between the current and predicted word within a sentence.
    *   `min_count`: Ignores all words with total frequency lower than this.
    *   `epochs`: Number of iterations (passes) over the training corpus.
    *   `dm`: Defines the training algorithm. If `dm=1`, distributed memory model 'PV-DM' is used. Otherwise, distributed bag of words 'PV-DBOW' is employed.
4.  **Infer Vector:** Use `infer_vector()` to get the document embedding for a new (unseen) document. This performs several iterations of training on the new document to fine-tune its embedding.
5.  **Find Similar Documents:**  Use `model.dv.most_similar()` to find the documents most similar to the new document, based on the cosine similarity of their embeddings. `dv` is the Docvecs array which stores the learned document embeddings.
6.  **Save and Load:**  The `save()` and `load()` methods allow you to persist and retrieve the trained model.

## 4) Follow-up question

How do you choose the best value for the `vector_size` hyperparameter in Doc2Vec?  Are there any rules of thumb or specific strategies for optimization?