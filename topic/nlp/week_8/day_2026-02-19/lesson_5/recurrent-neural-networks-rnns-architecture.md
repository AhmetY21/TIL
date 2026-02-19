---
title: "Recurrent Neural Networks (RNNs) Architecture"
date: "2026-02-19"
week: 8
lesson: 5
slug: "recurrent-neural-networks-rnns-architecture"
---

# Topic: Recurrent Neural Networks (RNNs) Architecture

## 1) Formal definition (what is it, and how can we use it?)

Recurrent Neural Networks (RNNs) are a type of neural network designed to handle sequential data. Unlike traditional feedforward neural networks, RNNs have a "memory" that allows them to use information from previous inputs to influence the current output. This memory is implemented through a recurrent connection that feeds the output of a previous time step back into the input of the current time step.

Formally, an RNN can be defined as a network where the output at time step *t*, *h<sub>t</sub>*, depends not only on the input at time step *t*, *x<sub>t</sub>*, but also on the hidden state from the previous time step, *h<sub>t-1</sub>*. The hidden state acts as a memory, storing information about the past. The equations governing a simple RNN are as follows:

*   *h<sub>t</sub>* = *f*( *W<sub>xh</sub>* *x<sub>t</sub>* + *W<sub>hh</sub>* *h<sub>t-1</sub>* + *b<sub>h</sub>* )
*   *y<sub>t</sub>* = *g*( *W<sub>hy</sub>* *h<sub>t</sub>* + *b<sub>y</sub>* )

Where:

*   *x<sub>t</sub>* is the input at time step *t*.
*   *h<sub>t</sub>* is the hidden state at time step *t*.
*   *y<sub>t</sub>* is the output at time step *t*.
*   *W<sub>xh</sub>* is the weight matrix connecting the input to the hidden state.
*   *W<sub>hh</sub>* is the weight matrix connecting the previous hidden state to the current hidden state.
*   *W<sub>hy</sub>* is the weight matrix connecting the hidden state to the output.
*   *b<sub>h</sub>* is the bias vector for the hidden state.
*   *b<sub>y</sub>* is the bias vector for the output.
*   *f* is an activation function (e.g., sigmoid, tanh, ReLU).
*   *g* is an activation function for the output layer (e.g., softmax for classification, linear for regression).

We can use RNNs for various tasks including:

*   **Sequence prediction:** Predicting the next element in a sequence (e.g., next word in a sentence).
*   **Sequence classification:** Assigning a label to an entire sequence (e.g., sentiment analysis).
*   **Sequence generation:** Generating new sequences (e.g., machine translation, text generation).
*   **Time series analysis:** Analyzing and predicting patterns in time-dependent data.

## 2) Application scenario

Consider a scenario where we want to perform sentiment analysis on movie reviews. We can represent each review as a sequence of words.  An RNN can process the review word by word, maintaining a hidden state that captures the overall sentiment expressed in the review. As the RNN reads each word, the hidden state is updated, taking into account both the current word and the information accumulated from previous words. After processing the entire review, the final hidden state can be used to classify the sentiment as positive, negative, or neutral.  The RNN learns to associate specific words and phrases with different sentiments, allowing it to accurately classify the overall sentiment of the review.

## 3) Python method (if possible)

We can implement an RNN using libraries like TensorFlow or PyTorch. Here's a basic example using TensorFlow/Keras to create a simple RNN for sequence classification:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# Define the model
model = Sequential()

# Embedding layer to convert words to vectors
model.add(Embedding(input_dim=10000,  # Vocabulary size
                    output_dim=32)) # Embedding dimension

# SimpleRNN layer with 32 units
model.add(SimpleRNN(units=32))

# Dense layer for classification (e.g., binary classification)
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Print the model summary
model.summary()

# Prepare dummy data for training (replace with your actual data)
import numpy as np
X_train = np.random.randint(0, 10000, size=(100, 20)) # 100 sequences, each of length 20
y_train = np.random.randint(0, 2, size=(100, 1))   # 100 labels (0 or 1)

# Train the model
model.fit(X_train, y_train, epochs=2) # Train for 2 epochs

# Evaluate the model (replace with your actual test data)
X_test = np.random.randint(0, 10000, size=(50, 20))
y_test = np.random.randint(0, 2, size=(50, 1))
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")
```

This code snippet defines a simple RNN model with an embedding layer, a SimpleRNN layer, and a dense output layer. The `Embedding` layer converts word indices into dense vectors. The `SimpleRNN` layer is the core of the RNN, and the `Dense` layer performs the classification. Remember to replace the dummy data with your actual dataset and adjust the hyperparameters (e.g., vocabulary size, embedding dimension, number of RNN units, number of epochs) according to your specific task.

## 4) Follow-up question

While RNNs are powerful for processing sequential data, they suffer from the vanishing gradient problem, especially when dealing with long sequences. This makes it difficult for them to learn long-range dependencies. What are some common architectures that address this issue, and how do they work?