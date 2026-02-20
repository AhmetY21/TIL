---
title: "Bidirectional RNNs"
date: "2026-02-20"
week: 8
lesson: 3
slug: "bidirectional-rnns"
---

# Topic: Bidirectional RNNs

## 1) Formal definition (what is it, and how can we use it?)

A Bidirectional Recurrent Neural Network (BRNN) is a type of recurrent neural network that processes input sequences in both directions – forward and backward – before making predictions.  Unlike a standard RNN, which processes information sequentially from the beginning to the end, a BRNN uses two separate RNNs: one processing the input sequence in its original order (forward RNN) and the other processing the input sequence in reverse order (backward RNN).

Formally:

*   **Input:** A sequence of vectors *x = (x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>T</sub>)*.
*   **Forward RNN:** Processes the input from *x<sub>1</sub>* to *x<sub>T</sub>*.  At each time step *t*, it computes a hidden state *h<sub>t</sub><sup>f</sup>* based on the current input *x<sub>t</sub>* and the previous hidden state *h<sub>t-1</sub><sup>f</sup>*.
*   **Backward RNN:** Processes the input from *x<sub>T</sub>* to *x<sub>1</sub>*. At each time step *t*, it computes a hidden state *h<sub>t</sub><sup>b</sup>* based on the current input *x<sub>t</sub>* and the previous hidden state *h<sub>t+1</sub><sup>b</sup>*.  Note that while the *input* to the backward RNN might be in reverse order, the time index *t* still increases from 1 to T.
*   **Output:** At each time step *t*, the outputs of the forward and backward RNNs, *h<sub>t</sub><sup>f</sup>* and *h<sub>t</sub><sup>b</sup>*, are combined (e.g., by concatenation, summation, or averaging) to produce the final output *y<sub>t</sub>*.  Mathematically: *y<sub>t</sub> = f(h<sub>t</sub><sup>f</sup>, h<sub>t</sub><sup>b</sup>)*, where *f* is a combination function.

We can use BRNNs to:

*   **Understand context better:** By considering information from both past and future inputs, BRNNs provide a more comprehensive understanding of the context for each element in the sequence.  This is particularly useful when the meaning of an element depends on what comes before and after it.
*   **Improve prediction accuracy:** BRNNs often achieve higher accuracy than standard RNNs in tasks where context is crucial, such as sequence labeling, machine translation, and speech recognition.
*   **Handle long-range dependencies:**  While BRNNs are still subject to the vanishing gradient problem to some extent, the bidirectional processing can help capture long-range dependencies more effectively than unidirectional RNNs.

## 2) Application scenario

A common application scenario for BRNNs is **Part-of-Speech (POS) tagging**. In POS tagging, the goal is to assign a grammatical tag (e.g., noun, verb, adjective) to each word in a sentence.  The correct POS tag for a word often depends on the words surrounding it.

For example, consider the sentence "The cat sat on the mat."  A BRNN can analyze the sentence from left to right (forward RNN) and from right to left (backward RNN). The forward RNN might see "The cat" and start to build an understanding that "cat" is likely a noun.  The backward RNN might see "on the mat" and reinforce the idea that "cat" is likely a noun acting as the subject of the verb "sat".  By combining the information from both RNNs, the BRNN can confidently assign the "Noun" tag to the word "cat".

Another example is **Named Entity Recognition (NER)**. Detecting named entities (people, organizations, locations) also benefits from contextual information.  A BRNN can use the forward and backward passes to accurately identify and classify named entities even if they are ambiguous or have different meanings in different contexts.

## 3) Python method (if possible)
```python
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, LSTM, Dense

# Example: Bidirectional LSTM for sequence classification

# Define the model
model = tf.keras.Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(None, 10)), # Input shape: (sequence length, features)
    Bidirectional(LSTM(32)),
    Dense(1, activation='sigmoid') # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Print the model summary
model.summary()

# Generate some dummy data for training
import numpy as np
X_train = np.random.rand(100, 20, 10) # 100 samples, sequence length 20, 10 features
y_train = np.random.randint(0, 2, 100)

# Train the model
model.fit(X_train, y_train, epochs=2, batch_size=32)

```

**Explanation:**

*   **`tensorflow.keras.layers.Bidirectional`:** This layer wraps another recurrent layer (e.g., `LSTM`, `GRU`, `SimpleRNN`) and creates a bidirectional version of it.  It automatically creates the forward and backward RNNs.
*   **`LSTM(64, return_sequences=True)`:**  This creates an LSTM layer with 64 hidden units.  `return_sequences=True` is important in the first Bidirectional layer if you want to stack multiple Bidirectional layers. It means that the LSTM layer will output a sequence of hidden states for each time step, rather than just the final hidden state.
*   **`LSTM(32)`:** The second LSTM layer has 32 units. Because `return_sequences` is not set, the second Bidirectional layer receives only the final hidden state.
*   **`input_shape=(None, 10)`:** Specifies the shape of the input data.  `None` indicates that the sequence length can vary.  `10` represents the number of features at each time step.
*   The code demonstrates a simple binary classification task. It shows how to define, compile, and train a BRNN using TensorFlow/Keras.  The dummy data is randomly generated for demonstration purposes.  Replace this with your actual dataset.

## 4) Follow-up question

What are the limitations of Bidirectional RNNs, and what are some techniques to address those limitations? For example, are there cases where non-recurrent methods like transformers would be preferable and why?