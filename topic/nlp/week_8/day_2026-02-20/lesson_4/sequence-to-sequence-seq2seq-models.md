---
title: "Sequence-to-Sequence (Seq2Seq) Models"
date: "2026-02-20"
week: 8
lesson: 4
slug: "sequence-to-sequence-seq2seq-models"
---

# Topic: Sequence-to-Sequence (Seq2Seq) Models

## 1) Formal definition (what is it, and how can we use it?)

A Sequence-to-Sequence (Seq2Seq) model is a type of neural network architecture designed to transform an input sequence into an output sequence. Crucially, the input and output sequences can have different lengths.

Formally, a Seq2Seq model consists of two main components:

*   **Encoder:** The encoder takes the input sequence (e.g., "Hello, world!") and processes it to create a fixed-length vector representation called the "context vector" or "thought vector."  This vector aims to encapsulate the semantic meaning of the entire input sequence.  The encoder is typically an RNN (Recurrent Neural Network) like an LSTM (Long Short-Term Memory) or GRU (Gated Recurrent Unit).  The final hidden state of the encoder is often used as the context vector.

*   **Decoder:** The decoder takes the context vector produced by the encoder as input and generates the output sequence (e.g., "Bonjour, le monde!").  The decoder is also typically an RNN (LSTM or GRU).  It starts by using the context vector as its initial hidden state.  Then, at each time step, it predicts the next token in the output sequence, conditioned on the previous token and its current hidden state. A special "start-of-sequence" token is often used as the initial input to the decoder, and a "end-of-sequence" token signals the end of the generated sequence.

We can use Seq2Seq models for tasks where the input and output are both sequences but may have varying lengths, such as:

*   **Machine Translation:** Translating text from one language to another.
*   **Text Summarization:** Generating a shorter version of a longer document.
*   **Chatbots:** Generating responses to user input.
*   **Speech Recognition:** Converting audio to text.
*   **Code Generation:** Generating code from natural language descriptions.

## 2) Application scenario

Consider the application scenario of **machine translation**.  We want to translate the English sentence "How are you?" to French.

1.  **Encoding:** The encoder, say an LSTM, takes the sequence of English words "How", "are", "you", "?" (represented as numerical vectors, often using word embeddings) as input. It processes each word sequentially, updating its hidden state at each step. The final hidden state of the encoder becomes the context vector. This vector represents the meaning of the entire English sentence.

2.  **Decoding:** The decoder, also an LSTM, is initialized with the context vector from the encoder. It starts by generating the first word of the French translation, given the context vector and a "start-of-sequence" token. Let's say it predicts "Comment".

3.  **Iteration:** The decoder then takes "Comment" as input (again, as a numerical vector) and predicts the next word.  It might predict "allez-vous".

4.  **Termination:** The decoder continues generating words until it predicts an "end-of-sequence" token, indicating that the translation is complete. The final output sequence would be "Comment allez-vous ?".

## 3) Python method (if possible)

Using TensorFlow and Keras, we can create a simple Seq2Seq model.  This is a simplified example for demonstration purposes.  A real-world translation model would require much larger datasets and more sophisticated architectures (like attention mechanisms).

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, LSTM, Dense

# Define the model
# Input sequence
encoder_inputs = Input(shape=(None, 100)) # Assuming each word is represented by a 100-dimensional vector

# Encoder LSTM
encoder_lstm = LSTM(256, return_state=True) # 256 units in the LSTM layer
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

# Keep the states
encoder_states = [state_h, state_c]

# Decoder input
decoder_inputs = Input(shape=(None, 100))

# Decoder LSTM, using encoder states as initial state
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

# Dense layer for output
decoder_dense = Dense(100, activation='softmax') # Assuming vocabulary size is 100
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model
model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# Training data (dummy data for illustration - replace with actual data)
import numpy as np
encoder_input_data = np.random.rand(1000, 20, 100) # 1000 samples, sequence length 20, vector size 100
decoder_input_data = np.random.rand(1000, 20, 100)
decoder_target_data = np.random.rand(1000, 20, 100)

# Train the model
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=64,
          epochs=10)
```

**Explanation:**

*   **Inputs:** We define input layers for the encoder and decoder.  We assume that words are represented as 100-dimensional vectors (word embeddings).
*   **Encoder:**  An LSTM is used to encode the input sequence.  `return_state=True` makes it return its hidden state (`state_h`) and cell state (`state_c`), which are used to initialize the decoder.
*   **Decoder:**  Another LSTM decodes the sequence.  Crucially, `initial_state=encoder_states` initializes the decoder with the encoder's final states, thus passing the "context" information. `return_sequences=True` is important as we need the LSTM to output a sequence for each time step.
*   **Dense Layer:** A dense (fully connected) layer is used to predict the next word in the sequence.  The `softmax` activation ensures that the output is a probability distribution over the vocabulary.  We are assuming a vocabulary size of 100 in this example.
*   **Model Definition:** `keras.Model` connects the encoder and decoder to form the complete Seq2Seq model.
*   **Training:**  The model is trained using `categorical_crossentropy` loss, which is suitable for multi-class classification problems (where each word in the vocabulary is a class). The example shows random data as placeholders, which must be replaced by actual data when training for translation.

**Important Notes:**

*   This is a basic example. Real-world translation models use more advanced techniques like attention mechanisms to improve performance.
*   The input and output data need to be preprocessed and vectorized (e.g., using word embeddings or one-hot encoding).
*   This example uses a fixed sequence length.  Padding or masking is often used to handle variable-length sequences.

## 4) Follow-up question

How does the "Attention Mechanism" improve upon the basic Seq2Seq model, and what problem does it address?