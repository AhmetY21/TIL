---
title: "FastText: Handling Out-of-Vocabulary Words"
date: "2026-03-07"
week: 10
lesson: 3
slug: "fasttext-handling-out-of-vocabulary-words"
---

# Topic: FastText: Handling Out-of-Vocabulary Words

## 1) Formal definition (what is it, and how can we use it?)

FastText is a word embedding technique developed by Facebook AI Research. While conceptually similar to Word2Vec, a key advantage of FastText lies in its ability to handle Out-of-Vocabulary (OOV) words.  OOV words are words that were not seen during the training of the model.

FastText achieves this by breaking down words into smaller subword units, specifically character n-grams. For example, consider the word "apple". With n=3, FastText would represent it as: "<ap", "app", "ppl", "ple", "le>". The angle brackets indicate the beginning and end of the word. This allows FastText to represent unseen words by utilizing the embeddings of its constituent n-grams.

When encountering an OOV word, FastText computes its vector representation by summing the embeddings of its n-grams. This approach allows the model to:

*   **Represent OOV words:**  Even if a word isn't in the vocabulary, it can still be represented by combining the vectors of its subword units.
*   **Handle morphological variants:**  Words with similar morphology (e.g., "running" and "runner") will have similar n-grams and therefore similar vector representations, even if one wasn't seen during training.
*   **Improve performance on rare words:** Because frequent words are represented both as single words and as a collection of n-grams, the impact of rare words on the overall embedding space is lessened.

We can use FastText for various NLP tasks like:

*   **Word Similarity:** Calculate the similarity between words, including OOV words, based on their vector representations.
*   **Text Classification:** Use FastText embeddings as input features to train text classifiers.
*   **Information Retrieval:**  Find documents relevant to a query, even if the query contains OOV words.
*   **Language Modeling:**  Predict the probability of a word sequence, even if it contains OOV words.

## 2) Application scenario

Consider a sentiment analysis task where the training data consists of reviews of electronic products. The training data may not contain words like "smartphone," "bluetooth," or specific brand names.  If a new review contains the sentence "This new iGadget smartphone has amazing bluetooth capabilities!", a traditional Word2Vec model might struggle with the words "iGadget," "smartphone," and "bluetooth," assigning them random or zero vectors, because these words were likely unseen during training.

FastText, on the other hand, can handle these OOV words:

*   "iGadget" might be represented as n-grams like "<ig", "iga", "gad", "dge", "get", "et>".
*   "smartphone" would be represented by its own n-grams. Since "smart" is a relatively common word, the n-grams containing "smart" would have meaningful embeddings, contributing to a more accurate representation of the word.
*   "bluetooth" also would be represented by its n-grams.

By combining the embeddings of these n-grams, FastText can generate reasonable vector representations for these OOV words, enabling the sentiment analysis model to better understand the sentence and make a more accurate prediction about the sentiment expressed in the review.

## 3) Python method (if possible)

```python
import fasttext

# Train a FastText model (skipgram or cbow)
# Replace 'data.txt' with the path to your training data file
model = fasttext.train_unsupervised('data.txt', model='skipgram') # or 'cbow'

# Get the vector representation of a word, including OOV words
word_vector = model.get_word_vector('unknownword')

print(word_vector.shape) # Returns a numpy array representing the word vector

# Save the model
model.save_model("model_filename.bin")

# Load the model
loaded_model = fasttext.load_model("model_filename.bin")

# Example with sentences and OOV words
sentence = "This is a new and unknown word to test the model."
words = sentence.split()

for word in words:
  vector = loaded_model.get_word_vector(word)
  print(f"Vector for '{word}': {vector[:5]}...") # print only first 5 elements for brevity
```

*   **`fasttext.train_unsupervised('data.txt', model='skipgram')`:** This function trains a FastText model on a text file (`data.txt`).  The `model` parameter specifies the training algorithm, which can be either 'skipgram' or 'cbow'.
*   **`model.get_word_vector('unknownword')`:**  This function returns the vector representation of the word 'unknownword'.  If the word is OOV, FastText will calculate its vector based on the n-grams of the word.
*   **`model.save_model("model_filename.bin")`:** This saves the trained model to a binary file.
*   **`fasttext.load_model("model_filename.bin")`:**  This loads a previously trained FastText model.

## 4) Follow-up question

How does the choice of the n-gram size (the 'n' in n-grams) affect the performance of FastText, particularly when dealing with different languages (e.g., English vs. highly inflected languages like Finnish or Turkish)? What are the trade-offs to consider when selecting the optimal n-gram size?