---
title: "Long Short-Term Memory Networks (LSTMs)"
date: "2026-02-20"
week: 8
lesson: 1
slug: "long-short-term-memory-networks-lstms"
---

# Topic: Long Short-Term Memory Networks (LSTMs)

## 1) Formal definition (what is it, and how can we use it?)

Long Short-Term Memory networks (LSTMs) are a special kind of Recurrent Neural Network (RNN) architecture designed to overcome the vanishing gradient problem, which hinders standard RNNs from learning long-term dependencies in sequential data. In essence, LSTMs are designed to "remember" information over long periods.

**What is it?**

LSTMs introduce a "cell state" (often denoted as `C_t`) and three "gates": the input gate (`i_t`), the forget gate (`f_t`), and the output gate (`o_t`). These gates regulate the flow of information into and out of the cell state. The cell state acts as a kind of conveyor belt, transporting relevant information across many time steps. The gates are essentially neural networks that output values between 0 and 1, representing the degree to which information should be allowed to pass through.

Here's a breakdown of the key components and their equations:

*   **Input Gate (`i_t`):** Determines what new information should be added to the cell state.
    *   `i_t = σ(W_i * [h_{t-1}, x_t] + b_i)`
*   **Forget Gate (`f_t`):** Determines what information should be discarded from the cell state.
    *   `f_t = σ(W_f * [h_{t-1}, x_t] + b_f)`
*   **Output Gate (`o_t`):** Determines what information from the cell state should be output as the hidden state.
    *   `o_t = σ(W_o * [h_{t-1}, x_t] + b_o)`
*   **Cell State Candidate (`C~_t`):** A candidate for the new cell state.
    *   `C~_t = tanh(W_c * [h_{t-1}, x_t] + b_c)`
*   **Cell State (`C_t`):** The updated cell state.
    *   `C_t = f_t * C_{t-1} + i_t * C~_t`
*   **Hidden State (`h_t`):** The output of the LSTM at time step `t`.
    *   `h_t = o_t * tanh(C_t)`

Where:
*   `x_t` is the input at time step `t`
*   `h_{t-1}` is the hidden state from the previous time step
*   `W` are weight matrices
*   `b` are bias vectors
*   `σ` is the sigmoid activation function
*   `tanh` is the hyperbolic tangent activation function

**How can we use it?**

LSTMs are used for tasks involving sequential data, where the order and context of the data points are important.  Examples include:

*   **Natural Language Processing:** Machine translation, text generation, sentiment analysis, part-of-speech tagging.
*   **Speech Recognition:** Converting speech to text.
*   **Time Series Analysis:** Predicting future values based on past observations (e.g., stock prices, weather patterns).
*   **Video Analysis:** Action recognition, video captioning.

## 2) Application scenario

**Application Scenario: Sentiment Analysis of Movie Reviews**

Imagine you want to build a system that can automatically classify movie reviews as either "positive" or "negative" based on the text of the review.  A standard bag-of-words approach might fail to capture the nuances of language, such as negation ("not good" vs. "good") or long-range dependencies (a review that starts positive but ends negative).

LSTMs can be used to address this. Each word in the review is converted into a word embedding (a vector representation).  The LSTM processes the sequence of word embeddings, one word at a time.  The hidden state of the LSTM at the last time step (or a pooling of hidden states across time steps) captures the overall sentiment of the review. This hidden state can then be fed into a fully connected layer followed by a sigmoid activation function to produce a probability score between 0 and 1, representing the likelihood of the review being positive. By training the LSTM on a large dataset of labeled movie reviews, the network learns to associate specific words and phrases with positive or negative sentiment, taking into account the context in which they appear. The "forget gate" can allow the LSTM to disregard irrelevant initial information, while the "input gate" and "output gate" allow the LSTM to selectively remember and utilize crucial pieces of information that contribute to the overall sentiment.

## 3) Python method (if possible)

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Define the vocabulary size and embedding dimension
vocab_size = 10000  # Example: the size of your word vocabulary
embedding_dim = 128 # Example: the dimensionality of the word embeddings
lstm_units = 64     # Example: the number of LSTM units in the LSTM layer
max_length = 200    # Example: maximum review length

# Build the LSTM model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    LSTM(lstm_units),
    Dense(1, activation='sigmoid') # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Print the model summary
model.summary()

# Example usage (replace with your actual data)
# Assuming you have training data like this:
# train_sequences: A list of tokenized and padded sequences of integers representing reviews.
# train_labels: A list of binary labels (0 or 1) representing negative or positive sentiment.
# X_train = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=max_length)
# y_train = np.array(train_labels)  # Convert list to NumPy array

# model.fit(X_train, y_train, epochs=2, batch_size=32) # Train the model
```

**Explanation:**

1.  **Embedding Layer:** Converts integer word indices into dense word embeddings. This layer learns a vector representation for each word in your vocabulary.  The `input_length` argument specifies the expected length of the input sequences.
2.  **LSTM Layer:** The core LSTM layer processes the sequence of word embeddings. `lstm_units` determines the number of LSTM cells in the layer.
3.  **Dense Layer:** A fully connected layer with a sigmoid activation function for binary classification (positive/negative sentiment).  Outputs a probability score between 0 and 1.
4.  **Compilation:** Defines the optimizer (Adam), loss function (binary cross-entropy, suitable for binary classification), and evaluation metric (accuracy).
5.  **Training (Commented Out):** The `model.fit()` function is used to train the model on your data.  You would replace `X_train` and `y_train` with your preprocessed training data. The code first pads the sequences to `max_length` to ensure all inputs have the same size.

## 4) Follow-up question

Given that LSTMs have a relatively complex architecture and are computationally expensive, what are some alternative approaches or architectures (e.g., simpler RNN variants, Transformers) that can be used to achieve similar performance in certain NLP tasks, and what are the trade-offs involved?