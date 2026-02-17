---
title: "FastText: Handling Out-of-Vocabulary Words"
date: "2026-02-17"
week: 8
lesson: 5
slug: "fasttext-handling-out-of-vocabulary-words"
---

# Topic: FastText: Handling Out-of-Vocabulary Words

## 1) Formal definition (what is it, and how can we use it?)

FastText is a word embedding technique developed by Facebook AI Research. While sharing similarities with Word2Vec (specifically the Skip-gram and CBOW models), FastText's key innovation is its ability to handle out-of-vocabulary (OOV) words. This is achieved by representing each word as the sum of character n-grams.

Specifically, instead of learning a single vector representation for each word in the vocabulary, FastText decomposes words into character n-grams. For example, consider the word "eating" and suppose we're using n=3 (trigrams).  FastText would represent "eating" as the collection of these trigrams:

`<ea`, `eat`, `ati`, `tin`, `ing`, `ng>` and the special character `<` and `>` denoting the start and end of the word (i.e., `<eating>`). It also includes the whole word `<eating>` as a feature.

When encountering an OOV word, FastText can still generate a word embedding by summing the embeddings of its constituent n-grams.  Even if the entire word has never been seen before, some of its n-grams likely have.  This allows FastText to produce a reasonable vector representation for unseen words, making it significantly more robust than methods like Word2Vec, which simply assign a random vector or an "UNK" (unknown) token to OOV words.

We can use FastText to:

*   **Generate word embeddings:** Like Word2Vec, it produces vector representations of words that capture semantic relationships.
*   **Handle OOV words:** This is its primary advantage, allowing it to process text with unseen words effectively.
*   **Text classification:** The learned word embeddings can be used as features for various text classification tasks.
*   **Word similarity tasks:** Determining how similar words are to each other based on their embeddings.
*   **Language identification:**  Using character n-grams can be helpful for identifying the language of a text, even with limited data.

## 2) Application scenario

Consider a situation where you are building a sentiment analysis model for a social media platform. This platform is constantly evolving, and new slang terms and abbreviations are being created all the time. A standard Word2Vec model would struggle with these new, unseen words, potentially leading to poor performance.

Using FastText in this scenario would be beneficial. Even if a new slang term like "yeet" appears, FastText can generate a reasonable embedding for it based on its constituent character n-grams. This allows the sentiment analysis model to handle new vocabulary more gracefully and maintain accuracy even when faced with unseen words.

Another scenario is in multilingual text processing.  Even if your training data doesn't contain words from a specific language, if that language shares similar characters and n-gram patterns with languages in your training data, FastText can still generate somewhat meaningful embeddings.

## 3) Python method (if possible)

You can use the `fasttext` library in Python to train and use FastText models. Here's a code example demonstrating how to train a model and get the vector for an OOV word:

```python
import fasttext

# Sample training data (replace with your actual data file)
with open("training_data.txt", "w") as f:
    f.write("This is a sample sentence.\n")
    f.write("Another sentence for training.\n")
    f.write("FastText is awesome!\n")

# Train a FastText model (skipgram architecture)
model = fasttext.train_unsupervised('training_data.txt', model='skipgram')

# Get the vector for a known word
vector_known = model.get_word_vector("awesome")
print("Vector for 'awesome':", vector_known[:5]) # print first 5 dimensions

# Get the vector for an out-of-vocabulary word
vector_oov = model.get_word_vector("unseenword")
print("Vector for 'unseenword':", vector_oov[:5]) # print first 5 dimensions

# Save the model
model.save_model("fasttext_model.bin")

# Load the model (optional)
loaded_model = fasttext.load_model("fasttext_model.bin")

# Clean up training data
import os
os.remove("training_data.txt")
os.remove("fasttext_model.bin")

```

**Explanation:**

1.  **Import `fasttext`:** Imports the necessary library.
2.  **Training Data:** Creates a simple training data file ( `training_data.txt`). **Important:** Replace this with your actual training data. FastText expects a text file where each line is a sentence.
3.  **`fasttext.train_unsupervised()`:** Trains a FastText model in unsupervised mode.
    *   `'training_data.txt'` specifies the path to the training data.
    *   `model='skipgram'` selects the Skip-gram architecture (alternatively, you can use `'cbow'`).  Other parameters like `lr` (learning rate), `dim` (embedding dimension), `epoch` (number of training epochs), and `minCount` (minimum word count) can also be specified.
4.  **`model.get_word_vector(word)`:**  Retrieves the word vector for the given word.  The example shows how to get vectors for both a known word ("awesome") and an OOV word ("unseenword").
5.  **`model.save_model(filename)`:** Saves the trained model to a file.
6.  **`fasttext.load_model(filename)`:** Loads a previously saved model.
7.  **Cleanup:** Delete the training file and the trained model to avoid clutter.

## 4) Follow-up question

How does the choice of the n-gram size (the 'n' in n-grams) affect the performance of FastText, particularly regarding handling OOV words and computational efficiency? What are the trade-offs involved in selecting different n-gram sizes?