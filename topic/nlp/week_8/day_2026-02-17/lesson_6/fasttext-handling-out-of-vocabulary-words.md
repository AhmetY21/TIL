---
title: "FastText: Handling Out-of-Vocabulary Words"
date: "2026-02-17"
week: 8
lesson: 6
slug: "fasttext-handling-out-of-vocabulary-words"
---

# Topic: FastText: Handling Out-of-Vocabulary Words

## 1) Formal definition (what is it, and how can we use it?)

FastText is a word embedding and text classification library developed by Facebook AI Research. Its key advantage, particularly relevant here, is its ability to handle Out-of-Vocabulary (OOV) words. Unlike models like Word2Vec that assign a vector only to known words from the training vocabulary, FastText represents each word as an aggregate of character n-grams.

Here's how it works:

1. **Character n-grams:** FastText breaks down each word into a sequence of character n-grams. For example, the word "apple" might be represented by the n-grams "<ap", "app", "ppl", "ple", "le>".  The angle brackets typically mark the beginning and end of the word.

2. **Embedding each n-gram:**  Each of these n-grams is then associated with its own embedding vector, similar to how Word2Vec assigns vectors to whole words.

3. **Word representation:** The vector representation of a word is the sum (or average) of the embedding vectors of all its constituent n-grams.

4. **Handling OOV words:** When encountering an OOV word, FastText doesn't assign a zero vector. Instead, it breaks down the OOV word into its constituent n-grams, retrieves their embedding vectors, and combines them to create a vector representation for the unknown word. This allows FastText to generate reasonable embeddings for words it has never seen before, based on the similarity of their character-level structure to known words.

We can use this to improve performance in scenarios where the vocabulary is limited or rapidly changing or when handling noisy text containing typos. By leveraging subword information, FastText provides more robust and generalizable word representations. It reduces the impact of rare words by associating them with similar subword features found in other words.

## 2) Application scenario

Consider a customer support chatbot trained on a specific product vocabulary. Now, imagine a user types a question containing a misspelled word, or a new slang term related to the product. Traditional word embedding models might fail to understand the user's intent because the misspelled word or slang term is an OOV word.

FastText, however, can handle this scenario more effectively. It can break down the misspelled word or slang term into character n-grams and use the learned embeddings of these n-grams to create a meaningful representation for the unknown word. This allows the chatbot to still understand the user's question and provide a relevant answer.

Specifically, scenarios where FastText excels include:

*   **Social Media Analysis:** Social media often contains slang, misspellings, and evolving vocabulary.
*   **E-commerce Product Search:** Users might misspell product names or use informal language.
*   **Low-Resource Languages:** Languages with limited training data can benefit from FastText's ability to generalize from subword information.
*   **Bioinformatics:** Biological sequences can be treated as strings of characters, and FastText can be used to find similarities between them.

## 3) Python method (if possible)

```python
import fasttext

# Example demonstrating loading a pre-trained model and getting the vector for an OOV word

# Download a pre-trained model (e.g., from https://fasttext.cc/docs/en/pretrained-vectors.html)
# Replace "path/to/model.bin" with the actual path to the downloaded model.
# For this example, let's assume you downloaded a model and named it "model.bin" in the same directory.
model_path = "model.bin"  # Or the full path if not in the current directory.

try:
    model = fasttext.load_model(model_path)
except ValueError as e:
    print(f"Error loading the model: {e}. Make sure the model file exists and is a valid FastText model.")
    print("You may need to download a pre-trained model from https://fasttext.cc/docs/en/pretrained-vectors.html and place it in the current directory.")
    exit()  # Exit the script if the model fails to load

# Example OOV word
oov_word = "unseenword"

# Get the vector representation for the OOV word
vector = model.get_word_vector(oov_word)

# Print the vector (it will be a numpy array)
print(f"Vector for '{oov_word}':")
print(vector)

# You can also get the nearest neighbors to the OOV word:
try:
    neighbors = model.get_nearest_neighbors(oov_word)
    print(f"\nNearest neighbors to '{oov_word}':")
    print(neighbors)
except ValueError as e:
    print(f"Error getting nearest neighbors: {e}. This can happen if the word embedding model doesn't support nearest neighbor functionality or if there are other model-specific constraints.")
```

**Explanation:**

1.  **`fasttext.load_model(model_path)`:** Loads a pre-trained FastText model.  Crucially, you need to *have* a pre-trained model. You can download these from the official FastText website. The code includes a `try-except` block to handle potential `ValueError` exceptions when loading the model (e.g., if the file doesn't exist or is corrupted). An error message gives helpful instructions on where to find pretrained models.
2.  **`model.get_word_vector(oov_word)`:** Retrieves the vector representation for the given word (even if it's an OOV word).
3.  **`model.get_nearest_neighbors(oov_word)`:** (Optional) Retrieves the nearest neighbors to the given word based on cosine similarity in the embedding space.  A `try-except` block is used here to handle cases where the loaded model might not support this method (depending on how it was trained). The docstrings for the fasttext library can explain which models support which methods.

**Important:** This code assumes you have a pre-trained FastText model available at the specified `model_path`.  Training a FastText model from scratch using Python is also possible, but this example focuses on the OOV word handling aspect with a pre-trained model. Training would involve using `fasttext.train_unsupervised()` or `fasttext.train_supervised()`.
Also, pre-trained models are very large (GBs). Download one with a language appropriate for your task.

## 4) Follow-up question

How does the choice of the n-gram size (e.g., 3-grams, 5-grams) affect the performance of FastText, especially when handling OOV words in morphologically rich languages (like Turkish or Finnish)? How do we determine the optimal n-gram size for a specific task and language?