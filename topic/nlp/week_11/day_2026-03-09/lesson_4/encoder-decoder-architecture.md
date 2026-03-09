---
title: "Encoder-Decoder Architecture"
date: "2026-03-09"
week: 11
lesson: 4
slug: "encoder-decoder-architecture"
---

# Topic: Encoder-Decoder Architecture

## 1) Formal definition (what is it, and how can we use it?)

The Encoder-Decoder architecture is a neural network architecture widely used in sequence-to-sequence tasks. It consists of two main components: an **Encoder** and a **Decoder**.

*   **Encoder:** The encoder takes a sequence as input (e.g., a sentence in the source language) and processes it to create a fixed-length vector representation, often called the "context vector" or "thought vector." This vector aims to capture the essence and meaning of the entire input sequence.  The encoder is typically implemented using recurrent neural networks (RNNs) like LSTMs or GRUs, or transformer-based architectures, which iteratively process each element of the input sequence and update its internal state.

*   **Decoder:** The decoder takes the context vector produced by the encoder and generates a new sequence as output (e.g., a sentence in the target language, or a summary of the input).  The decoder is also typically implemented using RNNs, LSTMs, GRUs, or transformers. It uses the context vector as its initial hidden state or incorporates it throughout the decoding process.  The decoder generates the output sequence one element at a time, conditioned on the context vector and the previously generated elements.  Crucially, the decoder needs a mechanism (often an end-of-sequence token) to signal when it has finished generating the output.

**How can we use it?** The encoder-decoder architecture is used whenever we need to transform one sequence into another sequence. Examples include:

*   **Machine Translation:** Translating a sentence from one language to another.
*   **Text Summarization:** Generating a shorter summary of a longer text.
*   **Image Captioning:** Generating a textual description of an image (where the image features are the encoder input).
*   **Speech Recognition:** Converting audio input to text.
*   **Chatbots and Question Answering:** Responding to user queries.
*   **Code generation:** Transforming natural language description into executable code.

A significant improvement over the basic encoder-decoder is the **attention mechanism**.  Attention allows the decoder to focus on different parts of the input sequence when generating each element of the output sequence.  This mitigates the information bottleneck inherent in compressing the entire input into a single fixed-length vector.

## 2) Application scenario

Let's consider the application scenario of **Machine Translation**.

Imagine we want to translate the English sentence "Hello, how are you?" to Spanish, which would be "Hola, ¿cómo estás?".

1.  **Encoder:** The English sentence "Hello, how are you?" is fed into the encoder.  The encoder, typically an LSTM or a Transformer, processes each word sequentially (or in parallel for Transformers), updating its internal state at each step.  After processing the entire sentence, the encoder produces a context vector that represents the meaning of the sentence.

2.  **Decoder:** The decoder receives the context vector from the encoder.  Starting with a special "start-of-sentence" token, the decoder generates the Spanish translation one word at a time. At each step, the decoder uses the context vector and the previously generated words to predict the next word in the Spanish sentence. For example:

    *   It might first predict "Hola,".
    *   Then, given the context vector and "Hola,", it might predict "¿".
    *   Then, given the context vector, "Hola," and "¿", it might predict "cómo".
    *   And so on, until it predicts the "end-of-sentence" token.

3.  **Attention (Optional):** If attention is used, the decoder doesn't rely solely on the fixed-length context vector. Instead, at each step of the decoding process, it attends to different parts of the English sentence based on its relevance to the current word being generated in Spanish. For instance, when predicting "estás", the attention mechanism might focus more on "you" in the English sentence.

The decoder then combines these attention weights with the encoder's hidden states to form a context vector that's specific to the current decoding step, leading to more accurate and fluent translations.

## 3) Python method (if possible)

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x) # (batch_size, seq_len, embedding_size)
        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs: (batch_size, seq_len, hidden_size)
        # hidden: (num_layers, batch_size, hidden_size)
        # cell: (num_layers, batch_size, hidden_size)

        return hidden, cell

# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, num_layers):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        # x: (batch_size, ) - single word at a time
        # hidden: (num_layers, batch_size, hidden_size)
        # cell: (num_layers, batch_size, hidden_size)

        x = x.unsqueeze(1) # (batch_size, 1)
        embedded = self.embedding(x) # (batch_size, 1, embedding_size)
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        # outputs: (batch_size, 1, hidden_size)
        # hidden: (num_layers, batch_size, hidden_size)
        # cell: (num_layers, batch_size, hidden_size)
        predictions = self.fc(outputs.squeeze(1)) # (batch_size, output_size)
        return predictions, hidden, cell

# Example usage
input_size = 10000  # Size of the input vocabulary
output_size = 8000   # Size of the output vocabulary
embedding_size = 256
hidden_size = 512
num_layers = 2
learning_rate = 0.001
batch_size = 64

encoder = Encoder(input_size, embedding_size, hidden_size, num_layers)
decoder = Decoder(output_size, embedding_size, hidden_size, num_layers)

# Example input sequences (replace with actual data and padding)
source_sequence = torch.randint(0, input_size, (batch_size, 20))  # (batch_size, seq_len)
target_sequence = torch.randint(0, output_size, (batch_size, 25))  # (batch_size, seq_len)

# Optimizer
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

# Training loop (simplified)
num_epochs = 10
for epoch in range(num_epochs):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Encoder forward pass
    hidden, cell = encoder(source_sequence)

    # Decoder forward pass (one word at a time)
    decoder_input = torch.randint(0, output_size, (batch_size,)) # Start token  # (batch_size, )
    loss = 0
    for t in range(target_sequence.shape[1]):
        prediction, hidden, cell = decoder(decoder_input, hidden, cell)
        target = target_sequence[:, t]
        loss += nn.CrossEntropyLoss()(prediction, target)
        decoder_input = target # Feed the actual target as the next input (teacher forcing)
        # Or, decoder_input = torch.argmax(prediction, dim=1) # Use the predicted word as the next input (no teacher forcing)

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
```

**Explanation:**

*   **Encoder:**  Takes an input sequence (represented as integers) and outputs the hidden and cell states.  `nn.Embedding` converts integers into dense vectors. `nn.LSTM` processes the sequence and generates the hidden and cell states, which encode the input sequence.
*   **Decoder:** Takes the current input word (represented as integers) and the hidden and cell states from the previous step. It outputs a prediction for the next word and the updated hidden and cell states. The `nn.Linear` layer maps the hidden state to a probability distribution over the output vocabulary.
*   **Training Loop:**
    *   The encoder processes the entire input sequence to produce the initial hidden and cell states for the decoder.
    *   The decoder then generates the output sequence one word at a time.
    *   The `CrossEntropyLoss` calculates the difference between the predicted probability distribution and the actual target word.
    *   **Teacher forcing:** The correct target word is fed back into the decoder as the next input. This helps the model learn faster, but can lead to instability if the model relies too heavily on the provided target words.
    *   **No teacher forcing:** The model's prediction becomes the next input. This can be more challenging but results in a more robust model.
*   **Example Usage:** The code sets up a basic encoder-decoder model using LSTMs. It simulates the training process with random data, but you'd replace this with your actual dataset and adjust the parameters as needed.  Crucially, this code shows the *structure* and *flow*, not a fully functional training loop that produces meaningful results.

**Important Considerations:**

*   **Padding:** Sequences often have varying lengths. Padding ensures all sequences in a batch have the same length.
*   **Vocabulary:** You'll need to create a vocabulary (mapping words to integers) for both the input and output languages.
*   **Attention Mechanism:** Adding an attention mechanism would significantly improve performance.
*   **Evaluation:** You'll need appropriate evaluation metrics (e.g., BLEU score for machine translation).

## 4) Follow-up question

How does the Transformer architecture differ from the traditional RNN-based Encoder-Decoder architecture, and what advantages does it offer in sequence-to-sequence tasks?