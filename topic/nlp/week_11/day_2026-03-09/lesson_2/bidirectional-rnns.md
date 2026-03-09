---
title: "Bidirectional RNNs"
date: "2026-03-09"
week: 11
lesson: 2
slug: "bidirectional-rnns"
---

# Topic: Bidirectional RNNs

## 1) Formal definition (what is it, and how can we use it?)

Bidirectional Recurrent Neural Networks (BRNNs) are a type of recurrent neural network that processes input sequences in *both* forward and backward directions. Unlike traditional RNNs which only process input from the beginning to the end of the sequence, BRNNs leverage information from both past and future contexts to make predictions at each point in the sequence.

Formally, a BRNN consists of two separate RNNs:

*   **Forward RNN:** Processes the input sequence *x = (x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>T</sub>)* from *x<sub>1</sub>* to *x<sub>T</sub>*, generating a sequence of forward hidden states *h<sup>f</sup> = (h<sup>f</sup><sub>1</sub>, h<sup>f</sup><sub>2</sub>, ..., h<sup>f</sup><sub>T</sub>)*.
*   **Backward RNN:** Processes the input sequence *x* in reverse order, from *x<sub>T</sub>* to *x<sub>1</sub>*, generating a sequence of backward hidden states *h<sup>b</sup> = (h<sup>b</sup><sub>1</sub>, h<sup>b</sup><sub>2</sub>, ..., h<sup>b</sup><sub>T</sub>)*.

At each time step *t*, the hidden states from both the forward and backward RNNs, *h<sup>f</sup><sub>t</sub>* and *h<sup>b</sup><sub>t</sub>*, are combined (e.g., concatenated or summed) to produce a final representation *h<sub>t</sub>*. This combined representation *h<sub>t</sub>* is then used for prediction.

**Use Cases:** BRNNs are particularly useful when the prediction at a certain time step depends on information from both before and after that time step in the sequence.  This is common in many NLP tasks. They are used to:

*   **Improve Contextual Understanding:** Provides a more complete context for each element in the sequence.
*   **Better Accuracy:** Often leads to higher accuracy compared to unidirectional RNNs.
*   **Sequence Labeling:** Suitable for tasks like part-of-speech tagging or named entity recognition.
*   **Machine Translation:** While transformer architectures have become dominant, BRNNs were historically used, especially in encoder-decoder models.
*   **Sentiment Analysis:** Useful for analyzing sentiment within a larger context of a document.

## 2) Application scenario

Let's consider the application of part-of-speech (POS) tagging. The goal is to assign a POS tag (e.g., noun, verb, adjective) to each word in a sentence.

**Example:**

Sentence: "The quick brown fox jumps over the lazy dog."

A unidirectional RNN would process the sentence from "The" to "dog".  When trying to tag the word "fox", it would only have information about "The quick brown". However, knowing what comes *after* "fox" ("jumps") is also helpful. For instance, if it saw "fox jumped", it'd reinforce that "fox" is likely a noun in this context.

A bidirectional RNN, on the other hand, processes the sentence in both directions. The forward RNN processes from "The" to "dog", and the backward RNN processes from "dog" to "The". This allows the BRNN to capture both the preceding and subsequent context for each word. When tagging "fox", the BRNN has information about "The quick brown" (from the forward RNN) and "jumps over the lazy dog" (from the backward RNN). This combined context allows for a more accurate POS tag assignment.

Therefore, using a BRNN in POS tagging allows the model to consider the entire sentence context, leading to improved accuracy in determining the correct POS tags for each word.

## 3) Python method (if possible)

Using TensorFlow/Keras:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense

# Parameters
vocab_size = 10000  # Example vocabulary size
embedding_dim = 128
lstm_units = 64
num_classes = 5  # Example number of POS tags

# Model Definition
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=None),  # input_length can be specified if sequences are padded to a fixed length
    Bidirectional(LSTM(lstm_units, return_sequences=True)),  # return_sequences=True for sequence labeling
    Dense(num_classes, activation='softmax') # Softmax for classification over POS tags
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Example Data (Replace with your actual data)
import numpy as np

# Assume your data is already tokenized and converted to integers
# x_train: (num_samples, sequence_length) - Integer sequences representing text
# y_train: (num_samples, sequence_length, num_classes) - One-hot encoded POS tags
x_train = np.random.randint(0, vocab_size, size=(100, 20))  # Example: 100 sequences, each of length 20
y_train = np.random.randint(0, num_classes, size=(100, 20))
y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)  # One-hot encode labels

# Train the model (Replace with your actual training loop)
model.fit(x_train, y_train, epochs=2, batch_size=32)
```

Explanation:

*   **Embedding Layer:** Maps integer word indices to dense vector representations.
*   **Bidirectional(LSTM(...)):**  This is the core of the BRNN.  `Bidirectional` wraps an LSTM layer and applies it in both directions. `return_sequences=True` is crucial for sequence labeling tasks like POS tagging, where we need an output for each time step in the input sequence. If you want to output just one vector for the entire input sequence (e.g., for sentiment analysis), you would omit  `return_sequences=True`.
*   **Dense Layer:**  Maps the LSTM output to the number of classes (POS tags in this case) and applies a softmax activation for probability distribution.
*   **Compilation:** Configures the learning process with an optimizer, loss function (categorical cross-entropy for multi-class classification), and metrics.
*   **Training:** The model is trained using the `.fit()` method. Remember to prepare your data (tokenization, padding if needed, and one-hot encoding for labels) before training.

## 4) Follow-up question

How do attention mechanisms improve upon the standard bidirectional RNN architecture, and in what specific scenarios would attention be preferred?