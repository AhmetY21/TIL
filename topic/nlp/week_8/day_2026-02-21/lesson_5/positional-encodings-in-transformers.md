---
title: "Positional Encodings in Transformers"
date: "2026-02-21"
week: 8
lesson: 5
slug: "positional-encodings-in-transformers"
---

# Topic: Positional Encodings in Transformers

## 1) Formal definition (what is it, and how can we use it?)

Positional encodings are a crucial component in the Transformer architecture that provides information about the position of tokens in the input sequence. Since Transformers, unlike Recurrent Neural Networks (RNNs), don't inherently process the input sequentially, they lack the capacity to understand the order of words. Positional encodings address this by injecting information about the position of each token directly into the token embeddings.

Formally, a positional encoding is a vector added to the word embedding vector. This vector represents the position of the word within the sequence. There are two main types of positional encodings:

*   **Learned Positional Encodings:** These are learnable embedding vectors, just like word embeddings. The model learns to associate each position with a specific vector.

*   **Fixed Positional Encodings:** These are predefined using mathematical functions, typically sinusoidal functions. The original Transformer paper (Attention is All You Need) used fixed positional encodings.

The most common fixed positional encoding is defined as follows:

*   PE(pos, 2i) = sin(pos / (10000<sup>2i/d<sub>model</sub></sup>))
*   PE(pos, 2i+1) = cos(pos / (10000<sup>2i/d<sub>model</sub></sup>))

where:
*   `pos` is the position of the word in the sequence (0-indexed).
*   `i` is the dimension of the positional encoding vector.
*   `d<sub>model</sub>` is the dimensionality of the word embeddings (and the positional encoding).

This formula generates a unique positional encoding vector for each position in the input sequence. The use of sine and cosine functions with different frequencies allows the model to attend to relative positions. Words that are close to each other will have similar positional encodings, while words that are far apart will have more dissimilar encodings.

We use positional encodings by adding them to the word embeddings *before* feeding the combined vector into the first layer of the Transformer (typically an attention layer). This ensures the model has access to both the meaning of the word (from the word embedding) and its position in the sequence (from the positional encoding).

## 2) Application scenario

Positional encodings are used in virtually every application of Transformers, since the Transformer architecture by itself has no notion of order.  Here are some example application scenarios:

*   **Machine Translation:** In translating a sentence from English to French, the order of words is crucial for grammatical correctness. Positional encodings help the Transformer model maintain the correct word order during translation.

*   **Text Summarization:** When generating a summary of a long document, the position of key sentences or phrases might be important. Positional encodings help the model capture these positional dependencies and generate a more coherent summary.

*   **Question Answering:** In answering a question based on a given context, understanding the relationships between different parts of the context and the question itself requires positional information.  For instance, knowing that a specific fact appears *before* a claim might be crucial for understanding the argument's structure and answering the question correctly.

*   **Text Generation (Language Modeling):** In generating text, the model needs to know the position of each word to create grammatically correct and meaningful sentences. Positional encodings are vital for predicting the next word in a sequence based on the preceding words.

## 3) Python method (if possible)

Here's a Python implementation of the fixed sinusoidal positional encoding:

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape (max_len, 1, d_model)
        self.register_buffer('pe', pe) # Not a learnable parameter

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Example Usage
if __name__ == '__main__':
    d_model = 512 # Embedding dimension
    max_len = 100 # Maximum sequence length
    batch_size = 32 # Example batch size

    # Instantiate the PositionalEncoding layer
    pos_encoder = PositionalEncoding(d_model, dropout=0.1, max_len=max_len)

    # Create some dummy input data (representing word embeddings)
    input_data = torch.randn(max_len, batch_size, d_model) # (seq_len, batch, feature)

    # Apply the positional encoding
    output_data = pos_encoder(input_data)

    print("Input data shape:", input_data.shape)
    print("Output data shape (after positional encoding):", output_data.shape) # Should be same
```

Key points:

*   `d_model`: The dimension of the embeddings and positional encodings.
*   `max_len`: The maximum sequence length that the positional encoding can handle.
*   `register_buffer`: This ensures that the `pe` tensor is not treated as a learnable parameter, but is still saved with the model. This is useful because the positional encodings are fixed.
*   The `forward` function adds the positional encoding to the input embeddings and applies dropout.
* The dimensions of the positional encoding and the input embedding must match for addition. Typically, the input to the transformer is arranged with the shape (sequence length, batch size, embedding dimension). This code produces a positional encoding with dimensions (max_len, 1, d_model) which allows it to be added via broadcasting.

## 4) Follow-up question

What are the advantages and disadvantages of using fixed positional encodings (like sinusoidal functions) compared to learned positional encodings? Are there any scenarios where one might be preferred over the other?