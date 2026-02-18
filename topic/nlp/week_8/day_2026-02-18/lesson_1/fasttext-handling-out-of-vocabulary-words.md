---
title: "FastText: Handling Out-of-Vocabulary Words"
date: "2026-02-18"
week: 8
lesson: 1
slug: "fasttext-handling-out-of-vocabulary-words"
---

# Topic: FastText: Handling Out-of-Vocabulary Words

## 1) Formal definition (what is it, and how can we use it?)

FastText is a word embedding and text classification library developed by Facebook AI. One of its key advantages, particularly compared to models like Word2Vec, is its ability to handle **out-of-vocabulary (OOV)** words, i.e., words that were not seen during the training phase.

Instead of treating words as indivisible units, FastText represents each word as a bag of **character n-grams**. An n-gram is a contiguous sequence of *n* characters within a word.  For instance, for the word "where" and n=3, we would have the n-grams "<wh", "whe", "her", "ere", "re>" where "<" and ">" denote the beginning and end of the word respectively.

How does this help with OOV words?

1.  **Vector Representation:** FastText learns vector representations for each of these n-grams during training.  The vector representation of a word is then the sum (or average) of the vector representations of its constituent n-grams.

2.  **Handling OOV Words:**  When encountering an OOV word, FastText breaks it down into its character n-grams. Since many of these n-grams are likely to have been seen during training (even if the entire word hasn't), FastText can compute a vector representation for the OOV word based on the learned vectors of its n-grams. This allows FastText to provide a reasonable embedding even for words it has never encountered before.

We can use FastText to generate word embeddings for a vocabulary, perform text classification tasks, and handle situations where new, unseen words are likely to appear in the input text. This is crucial for real-world applications dealing with dynamically evolving language, misspelled words, or specialized terminology.

## 2) Application scenario

Consider a customer support chatbot trained on a dataset of common queries and responses. Now, imagine a user types a query containing a new slang term or a misspelled word. Traditional word embedding models like Word2Vec might fail to understand the user's intent because they cannot generate a vector representation for the OOV word.

In this scenario, FastText shines. Even if the chatbot hasn't seen the exact word "n00b" (a slang term for "newbie") before, FastText can break it down into n-grams like "n00", "00b", "ob". If those n-grams were present in the training data (within other words, perhaps), FastText can approximate a vector representation for "n00b", enabling the chatbot to understand the query, potentially infer that it means "newbie", and provide a relevant response.

This ability to handle OOV words makes FastText highly suitable for applications such as:

*   **Chatbots:** Understanding user queries with slang, misspellings, or new terms.
*   **Search Engines:**  Handling misspelled search queries.
*   **Social Media Analysis:**  Processing user-generated content containing informal language and neologisms.
*   **Document Classification:**  Classifying documents that may contain domain-specific vocabulary or abbreviations.

## 3) Python method (if possible)

Here's an example of how to use FastText in Python using the `fasttext` library:

```python
import fasttext

# Create a dummy training file
with open("train.txt", "w") as f:
  f.write("This is a sentence. This is another sentence.  This is a longer sentence.\n")
  f.write("The cat sat on the mat.\n")
  f.write("The dog is barking loudly.\n")

# Train a FastText model
model = fasttext.train_unsupervised('train.txt', model='cbow') # or 'skipgram'

# Get the vector representation of a known word
vector = model.get_word_vector("cat")
print(f"Vector for 'cat': {vector[:5]}...") #Print first 5 elements for brevity

# Get the vector representation of an OOV word
oov_vector = model.get_word_vector("unknownword")
print(f"Vector for 'unknownword': {oov_vector[:5]}...") #Print first 5 elements for brevity

# Save the model
model.save_model("model_filename.bin")

# Load a pre-trained model
loaded_model = fasttext.load_model("model_filename.bin")

# You can also use pre-trained word vectors from FastText.  For example, using the English model:
# import fasttext.util
# ft = fasttext.load_model('cc.en.300.bin') #Download the appropriate language from: https://fasttext.cc/docs/en/pretrained-vectors.html
# vector_example = ft.get_word_vector("example")
# print(f"Vector for 'example' from pre-trained model: {vector_example[:5]}...")
```

**Explanation:**

1.  **`fasttext.train_unsupervised()`:** This function trains a FastText model on a text file.  The `model` parameter specifies the architecture ("cbow" for Continuous Bag of Words, or "skipgram" for Skip-gram). The 'train.txt' file must contain the training data.
2.  **`model.get_word_vector(word)`:** This method returns the vector representation of a given word. Even if the word is not in the vocabulary, FastText will attempt to create a vector based on its n-grams.
3.  **`model.save_model()` and `fasttext.load_model()`:**  These functions save and load a trained FastText model to/from a file.
4. **Loading pre-trained models:** Demonstrates loading a pre-trained model for a language.  You'll need to download the appropriate `.bin` file from the FastText website. These pre-trained models are very powerful and can be used directly without further training on your data (though fine-tuning can improve performance).

**Note:** You will need to install the `fasttext` library: `pip install fasttext`.

## 4) Follow-up question

How does the choice of the n-gram size (the 'n' in n-grams) affect the performance of FastText, especially in terms of handling OOV words and computational efficiency?  Are there any guidelines or methods for selecting an appropriate n-gram size for a specific task and dataset?