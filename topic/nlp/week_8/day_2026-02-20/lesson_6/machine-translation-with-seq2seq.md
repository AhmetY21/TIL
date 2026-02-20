---
title: "Machine Translation with Seq2Seq"
date: "2026-02-20"
week: 8
lesson: 6
slug: "machine-translation-with-seq2seq"
---

# Topic: Machine Translation with Seq2Seq

## 1) Formal definition (what is it, and how can we use it?)

Machine Translation (MT) with Seq2Seq (Sequence-to-Sequence) is a neural network architecture designed to map an input sequence (e.g., a sentence in English) to an output sequence (e.g., the same sentence in French). The "sequence" aspect is crucial as it handles the variable-length nature of sentences.

Formally, a Seq2Seq model typically consists of two main components:

*   **Encoder:** The encoder takes the input sequence as input and compresses it into a fixed-length vector representation called the "context vector" or "thought vector".  This vector aims to encapsulate the semantic meaning of the entire input sequence.  It's often implemented as a Recurrent Neural Network (RNN), such as an LSTM or GRU, processing the input sequence word by word.  The final hidden state of the RNN encoder becomes the context vector.

*   **Decoder:** The decoder takes the context vector produced by the encoder and generates the output sequence word by word. It's also usually an RNN (LSTM or GRU) initialized with the context vector as its initial hidden state.  At each time step, the decoder predicts the next word in the output sequence, conditioned on the context vector and the previously generated words.

During training, the model learns to map input sequences to their corresponding output sequences. This is typically done using a parallel corpus, which contains pairs of sentences in the source and target languages that are translations of each other. The model is trained to minimize a loss function, such as cross-entropy, which measures the difference between the predicted output sequence and the actual target sequence.

We can use Seq2Seq models for various tasks, including:

*   **Machine Translation:** Translating text from one language to another.
*   **Text Summarization:** Condensing a long piece of text into a shorter summary.
*   **Chatbots:** Generating responses to user input.
*   **Code Generation:**  Generating code from natural language descriptions.
*   **Image Captioning:**  Generating text descriptions of images (in this case, the encoder would be a CNN extracting features from the image).

## 2) Application scenario

Imagine we want to translate English sentences into German. We can use a Seq2Seq model for this purpose.

**Input:** "Hello, how are you?" (English)
**Output:** "Hallo, wie geht es dir?" (German)

1.  The **Encoder** takes the English sentence "Hello, how are you?" as input. It processes the sentence word by word (or subword) using an RNN (e.g., an LSTM). The final hidden state of the LSTM represents the "context" of the English sentence.

2.  The **Decoder** receives the context vector from the encoder. It initializes its own LSTM with this context vector. Then, it starts generating the German translation.

3.  At the first time step, the decoder might predict "Hallo". This prediction is based on the context vector and a special "start-of-sequence" token.

4.  At the second time step, the decoder takes "Hallo" (or its embedding representation) as input, along with the previous hidden state, and predicts "wie".

5.  This process continues until the decoder predicts a special "end-of-sequence" token, indicating the end of the German sentence.

The training data would consist of many English-German sentence pairs.  The model learns the statistical relationships between the two languages by minimizing the error between the predicted German sentences and the actual German translations in the training data.  Attention mechanisms (mentioned in the follow-up question) significantly improve the performance of this process, especially for longer sentences.

## 3) Python method (if possible)

Here's a simplified example using TensorFlow/Keras to illustrate the basic structure of a Seq2Seq model.  A full implementation for machine translation would be much more complex, involving padding, masking, tokenization, and attention mechanisms. This provides the *structure* but is not directly runnable for translation without significant addition work to prepare the data.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define model parameters
embedding_dim = 64  # Dimensionality of word embeddings
units = 128  # Number of LSTM units
input_vocab_size = 10000 #Size of English Vocabulary.  Needs to be built from actual data.
output_vocab_size = 8000 #Size of German Vocabulary. Needs to be built from actual data.
max_length = 20 #Maximum length of sentences. Needs to be determined by data analysis.

# Encoder
encoder_inputs = keras.Input(shape=(max_length,), dtype="int64")
x = layers.Embedding(input_vocab_size, embedding_dim)(encoder_inputs)
encoder_lstm = layers.LSTM(units, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(x)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = keras.Input(shape=(max_length,), dtype="int64")
decoder_embedding = layers.Embedding(output_vocab_size, embedding_dim)
x = decoder_embedding(decoder_inputs)
decoder_lstm = layers.LSTM(units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(x, initial_state=encoder_states)
decoder_dense = layers.Dense(output_vocab_size, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model
model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model
model.compile(
    optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Print model summary
model.summary()

# Example usage (needs preprocessed data - indices of words)
import numpy as np
encoder_input_data = np.random.randint(0, input_vocab_size, size=(100, max_length))
decoder_input_data = np.random.randint(0, output_vocab_size, size=(100, max_length))
decoder_target_data = np.random.randint(0, output_vocab_size, size=(100, max_length)) #shifted decoder_input_data, one time step ahead for training. Crucial!

model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=64, epochs=10)
```

**Explanation:**

*   **Encoder:** Takes integer sequences as input (representing word indices).  An embedding layer converts those integer indices to dense vectors. The LSTM processes these embeddings and produces the context vector (hidden states).
*   **Decoder:**  Also takes integer sequences as input. Uses an embedding layer and an LSTM.  Crucially, it's initialized with the encoder's hidden states.  A dense layer with a softmax activation predicts the probability distribution over the output vocabulary.
*   **Training:** During training, `decoder_target_data` is a one-step-ahead version of `decoder_input_data`. This is because the decoder is learning to predict the next word given the current word and the context.

**Important Notes:**

*   This is a very basic example and lacks crucial components for real-world translation, such as:
    *   **Data Preprocessing:** Tokenization, vocabulary creation, padding, and masking.
    *   **Attention Mechanisms:**  To allow the decoder to focus on different parts of the input sequence when generating each output word (greatly improves translation quality).
    *   **Beam Search:**  To improve the decoding process by exploring multiple possible translations.
    *   **Subword Tokenization (e.g., BPE):** To handle rare words and improve generalization.
*   The input and output data are represented as integer indices, assuming you have already built a vocabulary and tokenized your text data.
*   The `sparse_categorical_crossentropy` loss function is commonly used when the output labels are integers (word indices).

## 4) Follow-up question

Seq2Seq models are powerful, but the basic encoder-decoder architecture described above suffers from performance degradation when dealing with long sequences. This is because the entire input sequence is compressed into a single fixed-length context vector, which can become a bottleneck for long sentences.

How can we improve the performance of Seq2Seq models for long sequences? Explain the core idea and functionality of the "Attention Mechanism" in the context of Seq2Seq models.