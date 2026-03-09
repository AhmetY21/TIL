---
title: "Text Generation with RNNs"
date: "2026-03-09"
week: 11
lesson: 6
slug: "text-generation-with-rnns"
---

# Topic: Text Generation with RNNs

## 1) Formal definition (what is it, and how can we use it?)

Text generation with Recurrent Neural Networks (RNNs) is the process of training an RNN model on a corpus of text data and then using that model to generate new, potentially coherent, text sequences.

**What is it?**

Essentially, an RNN is trained to predict the next character or word in a sequence given the preceding sequence.  The RNN maintains a hidden state that represents the model's "memory" of the sequence so far.  This hidden state is updated as each character/word is processed. By repeatedly predicting the next element and feeding it back into the model as input, we can generate longer and longer sequences of text.  Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRUs) are commonly used variants of RNNs that address the vanishing gradient problem and are better at capturing long-range dependencies in text.

**How can we use it?**

*   **Creative Writing:** Generating poems, stories, scripts, or song lyrics.
*   **Machine Translation:** While not the primary architecture used in modern machine translation (Transformers are), RNNs can be used to generate target language text from encoded source language text.
*   **Chatbots:** Generating responses in conversational AI systems.
*   **Code Generation:** Generating code snippets based on prompts or specifications.
*   **Data Augmentation:** Creating synthetic data for training other NLP models.
*   **Language Modeling:** Building models that can estimate the probability of a sequence of words, which is useful in various NLP tasks.

## 2) Application scenario

**Application Scenario: Generating Shakespearean Sonnets**

Imagine we want to create a model that can write Shakespearean sonnets.  We would collect a large corpus of Shakespeare's sonnets (and perhaps other plays). We'd then train an RNN (e.g., an LSTM) on this corpus.  The model would learn the patterns of language, rhyming schemes, and meter characteristic of Shakespeare's writing.

After training, we can feed the model a starting character or sequence of characters. The model will predict the next character based on its training. We can then feed this predicted character back into the model and repeat the process.

By controlling parameters such as the "temperature" (which influences the randomness of the predictions), we can influence the style and creativity of the generated sonnet.  A lower temperature leads to more predictable and conservative text, while a higher temperature leads to more surprising and potentially nonsensical text.

The output might be a sonnet that (hopefully!) resembles Shakespeare's work in terms of language and structure, although it's unlikely to be truly original or profound.

## 3) Python method (if possible)

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample text data (replace with a larger corpus)
text = """
The quick brown fox jumps over the lazy dog.
The lazy dog sleeps soundly.
A quick brown fox.
"""

# 1. Tokenize the text
tokenizer = Tokenizer(num_words=None, char_level=True, oov_token="<UNK>") # character-level tokenizer
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1 # +1 for the <UNK> token and padding

# 2. Create sequences
input_sequences = []
for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# 3. Pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

# 4. Create predictors and label
X = input_sequences[:,:-1]
y = input_sequences[:,-1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words) # one-hot encode the output

# 5. Build the model
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(LSTM(150))
model.add(Dense(total_words, activation='softmax'))

# 6. Compile the model
optimizer = Adam(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# 7. Train the model
model.fit(X, y, epochs=100, verbose=0) # Increase epochs for better results with larger datasets

# 8. Generate text
def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += output_word
    return seed_text

# Example usage:
seed_text = "T"
next_words = 5
generated_text = generate_text(seed_text, next_words, model, max_sequence_len)
print(generated_text)
```

**Explanation:**

1.  **Data Preparation:** The code tokenizes the text, creates sequences of characters, pads these sequences to a uniform length, and prepares the input (X) and output (y) data. The output is one-hot encoded.
2.  **Model Building:** An RNN model is built using an Embedding layer (to convert characters to dense vectors), an LSTM layer (to capture sequence dependencies), and a Dense layer (to predict the next character).
3.  **Training:** The model is trained on the prepared data using categorical cross-entropy loss and the Adam optimizer.
4.  **Text Generation:**  The `generate_text` function takes a seed text, predicts the next characters iteratively, and appends them to the seed text to generate a longer sequence.  The `temperature` parameter (not implemented here for simplicity, but important) controls the randomness of the predictions.
5.  **Important Notes:**
    *   This is a simplified example with a small dataset. For better results, you'll need a much larger dataset and more training epochs.
    *   Character-level tokenization is used here for simplicity.  Word-level tokenization is also possible but requires more memory and computational resources for training on large datasets.
    *   Experiment with different architectures, hyperparameters, and optimization techniques to improve the performance of the model.
    *   Add `temperature` parameter to control the randomness of the generated text.

## 4) Follow-up question

How can attention mechanisms, commonly used in Transformers, be incorporated into RNN-based text generation models to improve their performance, especially for long sequences? Can you describe a potential architecture combining RNNs and attention for text generation?