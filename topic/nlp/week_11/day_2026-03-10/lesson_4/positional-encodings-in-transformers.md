---
title: "Positional Encodings in Transformers"
date: "2026-03-10"
week: 11
lesson: 4
slug: "positional-encodings-in-transformers"
---

# Topic: Positional Encodings in Transformers

## 1) Formal definition (what is it, and how can we use it?)

Positional encodings are added to the input embeddings in Transformer models to provide information about the position of tokens in a sequence. Unlike recurrent neural networks (RNNs) like LSTMs, Transformers are inherently order-agnostic due to their attention mechanism processing all tokens in parallel.  Therefore, they don't inherently "know" the order in which words appear in the input sequence. Positional encodings solve this problem.

Formally, a positional encoding is a vector of the same dimension as the word embeddings, which is added element-wise to each word embedding. The resulting vector then contains both the semantic meaning of the word and information about its position in the sequence. There are two common ways to generate positional encodings:

*   **Learned Positional Encodings:** These are trainable embeddings, similar to word embeddings, that are learned during training.  A separate embedding is learned for each position in the sequence, up to a maximum sequence length.

*   **Fixed Positional Encodings:** These are pre-calculated encodings based on mathematical functions, typically sine and cosine functions of different frequencies. The original Transformer paper used fixed positional encodings, defined as:

    ```
    PE(pos, 2i)   = sin(pos / (10000^(2i/d_model)))
    PE(pos, 2i+1) = cos(pos / (10000^(2i/d_model)))
    ```

    where:

    *   `pos` is the position of the word in the sequence (e.g., 0, 1, 2, ...)
    *   `i` is the dimension index (e.g., 0, 1, 2, ..., d_model/2)
    *   `d_model` is the dimension of the embedding vector

Using sine and cosine functions allows the model to attend to relative positions because, for any fixed offset `k`, `PE(pos+k)` can be represented as a linear function of `PE(pos)`.  This helps the model generalize to sequences of lengths not seen during training.

We use positional encodings by adding them to the word embeddings *before* feeding the result into the first layer of the Transformer. This provides positional information throughout the entire network.  The combined vector contains both word and position information.

## 2) Application scenario

A typical application scenario is machine translation. Consider translating the sentence "The cat sat on the mat" from English to French. Without positional encodings, the Transformer might treat "cat," "sat," "mat," and "the" as equally important, without knowing their order. This could lead to a poorly translated sentence.  With positional encodings, the model understands that "The" is the first word, "cat" is the second, and so on, allowing it to produce a more accurate translation, e.g., "Le chat était assis sur le tapis".

More broadly, positional encodings are crucial for *any* sequence-to-sequence task or sequence classification task that relies on the order of the words in the sequence. Examples include:

*   **Text summarization:** Understanding the beginning and ending of sentences is important.
*   **Question answering:** The position of keywords within the question and context passage matters.
*   **Sentiment analysis:** The order of words like "not good" vs. "good not" drastically changes the sentiment.
*   **Code generation:** The order of code statements is critical for correct execution.

## 3) Python method (if possible)

Here's a Python implementation of fixed positional encodings, as used in the original Transformer paper:

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


if __name__ == '__main__':
  # Example Usage:
  d_model = 512  # Embedding dimension
  seq_len = 20   # Sequence length
  batch_size = 32 # Batch size

  # Create a random input tensor with word embeddings
  input_embeddings = torch.randn(seq_len, batch_size, d_model)

  # Instantiate the PositionalEncoding module
  pos_encoder = PositionalEncoding(d_model)

  # Add positional encodings to the input embeddings
  output = pos_encoder(input_embeddings)

  print("Input embeddings shape:", input_embeddings.shape)
  print("Output shape (embeddings + positional encodings):", output.shape)
```

This code creates a `PositionalEncoding` module that generates the positional encodings using sine and cosine functions as described above.  The `forward` method adds the pre-computed positional encodings to the input embeddings.  The `register_buffer` call ensures that the `pe` tensor is saved and loaded with the model's state. The example usage demonstrates how to create random input embeddings and add positional information. The input tensor is of shape `[seq_len, batch_size, embedding_dim]` as that is what the standard PyTorch transformer expects.

## 4) Follow-up question

Are there other ways to represent positional information besides fixed or learned positional encodings? If so, what are their advantages and disadvantages compared to fixed and learned encodings?