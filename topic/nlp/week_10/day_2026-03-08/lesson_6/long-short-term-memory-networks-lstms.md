---
title: "Long Short-Term Memory Networks (LSTMs)"
date: "2026-03-08"
week: 10
lesson: 6
slug: "long-short-term-memory-networks-lstms"
---

# Topic: Long Short-Term Memory Networks (LSTMs)

## 1) Formal definition (what is it, and how can we use it?)

Long Short-Term Memory networks (LSTMs) are a type of recurrent neural network (RNN) architecture designed to address the vanishing gradient problem encountered in traditional RNNs. The vanishing gradient problem makes it difficult for RNNs to learn long-range dependencies in sequential data.

LSTMs achieve this through the introduction of a *memory cell* and *gates* that regulate the flow of information. The memory cell acts as a long-term memory store, holding information over extended periods.  The gates, implemented using sigmoid activation functions, control what information is written to the cell state, what information is read from the cell state, and what information is forgotten from the cell state.

Specifically, an LSTM cell consists of the following components:

*   **Cell State (C<sub>t</sub>):** The memory component that carries information across time steps.
*   **Hidden State (h<sub>t</sub>):** The output of the LSTM cell at time step *t*, carrying information to the next cell.

And three gates:

*   **Forget Gate (f<sub>t</sub>):** Determines which information to discard from the cell state. Its output (a value between 0 and 1) is multiplied element-wise with the previous cell state (C<sub>t-1</sub>), effectively "forgetting" information when the value is close to 0 and retaining it when the value is close to 1.  Calculated as sigmoid(W<sub>f</sub> * [h<sub>t-1</sub>, x<sub>t</sub>] + b<sub>f</sub>)
*   **Input Gate (i<sub>t</sub>):** Determines which new information to store in the cell state. It has two parts: a sigmoid layer that decides which values to update and a tanh layer that creates a vector of new candidate values (C'<sub>t</sub>) that could be added to the state.  Calculated as sigmoid(W<sub>i</sub> * [h<sub>t-1</sub>, x<sub>t</sub>] + b<sub>i</sub>)
    *   Candidate Values (C'<sub>t</sub>): Calculated as tanh(W<sub>C</sub> * [h<sub>t-1</sub>, x<sub>t</sub>] + b<sub>C</sub>)
*   **Output Gate (o<sub>t</sub>):** Determines what information to output based on the cell state. It applies a sigmoid layer to decide which parts of the cell state to output and then applies tanh to the cell state and multiplies it by the sigmoid output.  Calculated as sigmoid(W<sub>o</sub> * [h<sub>t-1</sub>, x<sub>t</sub>] + b<sub>o</sub>)

The equations governing LSTM operation at time step *t* are:

*   f<sub>t</sub> = sigmoid(W<sub>f</sub> * [h<sub>t-1</sub>, x<sub>t</sub>] + b<sub>f</sub>)
*   i<sub>t</sub> = sigmoid(W<sub>i</sub> * [h<sub>t-1</sub>, x<sub>t</sub>] + b<sub>i</sub>)
*   C'<sub>t</sub> = tanh(W<sub>C</sub> * [h<sub>t-1</sub>, x<sub>t</sub>] + b<sub>C</sub>)
*   C<sub>t</sub> = f<sub>t</sub> * C<sub>t-1</sub> + i<sub>t</sub> * C'<sub>t</sub>
*   o<sub>t</sub> = sigmoid(W<sub>o</sub> * [h<sub>t-1</sub>, x<sub>t</sub>] + b<sub>o</sub>)
*   h<sub>t</sub> = o<sub>t</sub> * tanh(C<sub>t</sub>)

Where:

*   x<sub>t</sub> is the input at time step *t*.
*   W<sub>f</sub>, W<sub>i</sub>, W<sub>C</sub>, W<sub>o</sub> are weight matrices.
*   b<sub>f</sub>, b<sub>i</sub>, b<sub>C</sub>, b<sub>o</sub> are bias vectors.
*   sigmoid is the sigmoid activation function.
*   tanh is the hyperbolic tangent activation function.

We can use LSTMs to model sequential data where long-range dependencies are important, such as time series forecasting, natural language processing (e.g., machine translation, text generation, sentiment analysis), speech recognition, and video analysis.

## 2) Application scenario

**Machine Translation:** Imagine translating a sentence from English to French. To accurately translate the sentence, the LSTM needs to remember the context of the entire sentence, including the relationships between words that are far apart.  For instance, pronouns often refer to nouns mentioned earlier in the sentence. A standard RNN might struggle to maintain this long-range dependency, potentially translating the pronoun incorrectly. An LSTM can effectively maintain the relevant context in its cell state, allowing for more accurate translation, especially for complex sentences. It encodes the English sentence into a vector representation (using an encoder LSTM) and then uses another LSTM (a decoder LSTM) to generate the French translation based on the encoded context.

## 3) Python method (if possible)

Using TensorFlow/Keras to implement an LSTM layer:

```python
import tensorflow as tf

# Define a sequential model
model = tf.keras.Sequential()

# Add an embedding layer (optional, but often used for text data)
model.add(tf.keras.layers.Embedding(input_dim=10000, output_dim=128)) # Vocabulary size = 10000, embedding dimension = 128

# Add an LSTM layer
model.add(tf.keras.layers.LSTM(units=64))  # 64 LSTM units

# Add a dense layer for output (e.g., classification)
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))  # 10 output classes with softmax activation

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

#Example Input data (replace with your actual data)
import numpy as np
dummy_input = np.random.randint(0, 10000, size=(32, 20)) # 32 samples, sequence length of 20, vocab index input
dummy_output = np.random.randint(0, 10, size=(32,)) # 32 samples, output indices

#Convert to one-hot encoded data
dummy_output = tf.keras.utils.to_categorical(dummy_output, num_classes=10)

# Train the model (replace with your actual data)
model.fit(dummy_input, dummy_output, epochs=2) #Train for two epochs.

```

**Explanation:**

1.  **`tf.keras.Sequential()`:** Creates a sequential model, where layers are added one after another.
2.  **`tf.keras.layers.Embedding()`:** (Optional but common for text) Converts integer-encoded words into dense vectors of fixed size.  `input_dim` is the vocabulary size, and `output_dim` is the embedding dimension.
3.  **`tf.keras.layers.LSTM(units=64)`:**  Adds an LSTM layer with 64 LSTM units (memory cells). The `units` argument determines the dimensionality of the hidden state and cell state.
4.  **`tf.keras.layers.Dense()`:** Adds a fully connected (dense) layer for output. The number of units and the activation function depend on the task (e.g., classification or regression).
5.  **`model.compile()`:** Configures the model for training, specifying the optimizer, loss function, and metrics.
6.  **`model.summary()`:** Prints a summary of the model architecture, including the number of parameters.
7.  **`model.fit()`:** Trains the model using the provided input data (`x`) and target data (`y`).  `epochs` specifies the number of training iterations.
8.  The dummy input and output are used to demonstrate the basic format required for inputting data to a Keras LSTM.

## 4) Follow-up question

What are some variations of LSTMs, such as GRUs (Gated Recurrent Units), and how do they differ from standard LSTMs in terms of architecture and performance?