---
title: "Recurrent Neural Networks (RNNs) Architecture"
date: "2026-03-08"
week: 10
lesson: 4
slug: "recurrent-neural-networks-rnns-architecture"
---

# Topic: Recurrent Neural Networks (RNNs) Architecture

## 1) Formal definition (what is it, and how can we use it?)

Recurrent Neural Networks (RNNs) are a type of neural network designed to process sequential data. Unlike feedforward networks, RNNs have a "memory" of previous inputs, allowing them to handle data where the order and relationship between data points are important.  This memory is implemented through recurrent connections, where the output of a neuron at a given time step is fed back into the network at the next time step.

Formally, an RNN processes a sequence of inputs *x = (x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>T</sub>)* one element at a time.  At each time step *t*, the RNN maintains a hidden state *h<sub>t</sub>*, which is updated based on the current input *x<sub>t</sub>* and the previous hidden state *h<sub>t-1</sub>*. The update rule for the hidden state is typically:

*h<sub>t</sub> = f(W<sub>xh</sub> * x<sub>t</sub> + W<sub>hh</sub> * h<sub>t-1</sub> + b<sub>h</sub>)*

where:
*   *h<sub>t</sub>* is the hidden state at time *t*.
*   *x<sub>t</sub>* is the input at time *t*.
*   *W<sub>xh</sub>* is the weight matrix connecting the input to the hidden state.
*   *W<sub>hh</sub>* is the weight matrix connecting the previous hidden state to the current hidden state (the recurrent connection).
*   *b<sub>h</sub>* is a bias term.
*   *f* is an activation function (e.g., tanh, ReLU).

The output *y<sub>t</sub>* at each time step is then calculated based on the hidden state:

*y<sub>t</sub> = g(W<sub>hy</sub> * h<sub>t</sub> + b<sub>y</sub>)*

where:
*   *y<sub>t</sub>* is the output at time *t*.
*   *W<sub>hy</sub>* is the weight matrix connecting the hidden state to the output.
*   *b<sub>y</sub>* is a bias term.
*   *g* is an activation function (e.g., sigmoid, softmax).

We can use RNNs for:

*   **Sequence Classification:**  Classifying a sequence as a whole (e.g., sentiment analysis of a sentence). The final hidden state *h<sub>T</sub>* can be used for the classification.
*   **Sequence-to-Sequence Mapping:**  Mapping one sequence to another (e.g., machine translation, speech recognition). The network processes an input sequence and generates an output sequence.
*   **Time Series Prediction:** Predicting future values in a time series (e.g., stock prices, weather patterns).
*   **Language Modeling:**  Predicting the next word in a sequence (e.g., generating text).
*   **Named Entity Recognition:** Identifying named entities in text.

## 2) Application scenario

A common application scenario is **sentiment analysis of movie reviews**.  Imagine you have a sequence of words representing a movie review, such as "This movie was absolutely terrible!". An RNN can process this sequence word by word.  At each time step, the RNN receives a word (represented as a word embedding) and updates its hidden state based on the current word and the previous hidden state.  After processing the entire review, the final hidden state captures information about the overall sentiment expressed in the review. This final hidden state can then be fed into a classifier (e.g., a softmax layer) to predict whether the review is positive, negative, or neutral.  The recurrent connections allow the RNN to understand the context and relationships between words, enabling it to discern subtle nuances in the language used. For example, the word "terrible" has a strong negative connotation, but its impact might be lessened if preceded by an intensifier like "not".  The RNN's memory allows it to capture this contextual information.

## 3) Python method (if possible)
```python
import tensorflow as tf

# Define the RNN model
class SimpleRNN(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units, num_classes):
        super(SimpleRNN, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.rnn = tf.keras.layers.SimpleRNN(rnn_units)
        self.dense = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, x):
        # x: Input sequence of word indices (batch_size, sequence_length)
        x = self.embedding(x) # (batch_size, sequence_length, embedding_dim)
        x = self.rnn(x) # (batch_size, rnn_units) - last hidden state
        x = self.dense(x) # (batch_size, num_classes)
        return x

# Example usage:
vocab_size = 10000 # Size of vocabulary
embedding_dim = 64  # Dimension of word embeddings
rnn_units = 128     # Number of RNN units (hidden state dimension)
num_classes = 2     # Number of classes (e.g., positive/negative)
sequence_length = 20 # maximum length of input sequence

# Create an instance of the model
model = SimpleRNN(vocab_size, embedding_dim, rnn_units, num_classes)

# Generate some dummy data for testing
import numpy as np
dummy_data = np.random.randint(0, vocab_size, size=(32, sequence_length)) # batch size 32
dummy_labels = np.random.randint(0, num_classes, size=(32,)) # batch size 32

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(dummy_data, dummy_labels, epochs=2) # Train on dummy data

# Make predictions
predictions = model.predict(dummy_data)
print(predictions.shape) # (32, 2)

```

## 4) Follow-up question

What are the vanishing gradient and exploding gradient problems in RNNs, and how do more advanced architectures like LSTMs and GRUs attempt to address them?