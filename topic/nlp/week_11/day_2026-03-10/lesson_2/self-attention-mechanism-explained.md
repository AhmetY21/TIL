---
title: "Self-Attention Mechanism Explained"
date: "2026-03-10"
week: 11
lesson: 2
slug: "self-attention-mechanism-explained"
---

# Topic: Self-Attention Mechanism Explained

## 1) Formal definition (what is it, and how can we use it?)

The self-attention mechanism is a neural network layer that allows a model to attend to different parts of the input sequence when processing each element of the sequence. Unlike traditional recurrent neural networks (RNNs) or convolutional neural networks (CNNs), self-attention does not rely on recurrence or convolution to capture relationships between elements. Instead, it directly computes the relationships between all pairs of elements in the input sequence.

Formally, given an input sequence $X = (x_1, x_2, ..., x_n)$, where each $x_i$ is a vector representation of a word or token, the self-attention mechanism transforms $X$ into a new sequence $Z = (z_1, z_2, ..., z_n)$ where each $z_i$ represents the contextualized representation of $x_i$.

The process involves the following steps:

1.  **Linear Projections:** The input sequence $X$ is transformed into three matrices: Queries (Q), Keys (K), and Values (V) using three different learned linear transformations (weight matrices):
    *   $Q = XW_Q$
    *   $K = XW_K$
    *   $V = XW_V$
    where $W_Q$, $W_K$, and $W_V$ are the weight matrices learned during training.  These matrices project the input into different representational spaces, allowing the model to learn different aspects of the input.

2.  **Attention Weights Calculation:** The attention weights are calculated by taking the dot product of the Query matrix $Q$ with the Key matrix $K$, scaling the result by the square root of the dimension of the keys ($d_k$), and then applying a softmax function:
    *   `Attention(Q, K, V) = softmax((QK^T) / sqrt(d_k))V`
    *   The dot product $QK^T$ computes the compatibility score between each query and each key. The higher the score, the more attention the query pays to the corresponding key (and value).
    *   Scaling by $\sqrt{d_k}$ helps to stabilize training by preventing the dot products from becoming too large, which can lead to very small gradients after the softmax function.
    *   The softmax function normalizes the scores into a probability distribution, representing the attention weights for each value.

3.  **Weighted Sum:**  Finally, the attention weights are used to compute a weighted sum of the Value matrix $V$:
    *   The output $Z$ is the result of multiplying the attention weights by the Value matrix.  Each $z_i$ is a weighted sum of all the values in $V$, where the weights are determined by the attention mechanism. This allows each $z_i$ to incorporate information from the entire input sequence, weighted by their relevance to the $i$-th element.

**How to Use It:**
Self-attention is used to create contextualized word embeddings, improving the quality of NLP models. It enables the model to attend to relevant parts of the input when processing each word, addressing limitations of earlier sequence models. It is a key component of transformer-based architectures and used for a variety of tasks such as machine translation, text summarization, question answering, and natural language inference.

## 2) Application scenario

**Scenario:** Machine Translation

Consider the task of translating the English sentence "The cat sat on the mat" into French.  A model using self-attention can effectively capture the relationships between words to produce an accurate translation.

*   **Encoding:** The English sentence is fed into the encoder part of the transformer model. The self-attention mechanism within the encoder allows each word to "attend" to all other words in the sentence. For instance, when processing the word "cat," the model can attend to "the" to understand the phrase "the cat." Similarly, it can attend to "sat" and "mat" to understand the overall context.

*   **Decoding:** In the decoder, the model generates the French translation "Le chat était assis sur le tapis."  Here, self-attention is used in two ways. First, within the decoder itself, to relate different parts of the French output sequence as it's being built. Second, to attend back to the encoded English sentence. For example, when generating "chat", the model will attend to the "cat" from the encoded English sentence, helping to ensure the correct word is chosen.  When generating "assis", the model may attend more strongly to "sat", indicating the relationship between the two.

**Benefits:**

*   **Handling Long-Range Dependencies:** Self-attention can effectively capture relationships between words that are far apart in the sentence, which is crucial for accurately translating complex sentences.
*   **Parallelization:** Unlike RNNs, self-attention allows for parallel processing of the input sequence, leading to faster training and inference.
*   **Improved Contextualization:** By attending to different parts of the input, self-attention provides richer contextualized representations of words, resulting in more accurate translations.

Other application scenarios include:

*   **Text Summarization:** Identifying and weighting the most important sentences or phrases.
*   **Question Answering:** Attending to the relevant parts of the context to answer a given question.
*   **Sentiment Analysis:** Identifying words or phrases that contribute the most to the overall sentiment.

## 3) Python method (if possible)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, queries, mask):
        # Get number of training examples
        N = queries.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(queries)  # (N, query_len, heads, head_dim)

        # Einsum does matrix multiplication for each head of dimension head_dim
        # with optional scaling to preserve gradients
        # energies shape: (N, heads, query_len, key_len)
        energies = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, head_dim), keys shape: (N, key_len, heads, head_dim)

        # Mask padded indices so their weights become 0
        if mask is not None:
            energies = energies.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energies / (self.embed_size ** (1/2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len), values shape: (N, value_len, heads, head_dim)
        # out after einsum (N, query_len, heads, head_dim) then reshape

        # Linear layer to send it through
        out = self.fc_out(out)
        # (N, query_len, embed_size)
        return out


if __name__ == '__main__':
    # Example usage:
    embed_size = 256
    heads = 8
    seq_length = 10
    batch_size = 32

    attention = SelfAttention(embed_size, heads)

    # Create dummy input
    values = torch.randn((batch_size, seq_length, embed_size))
    keys = torch.randn((batch_size, seq_length, embed_size))
    queries = torch.randn((batch_size, seq_length, embed_size))
    mask = torch.ones((batch_size, 1, seq_length, seq_length)).bool() # No masking for simplicity

    output = attention(values, keys, queries, mask)
    print(output.shape) # Should be (batch_size, seq_length, embed_size)
```

**Explanation:**

*   The code defines a `SelfAttention` class that implements the self-attention mechanism.
*   It takes `embed_size` (the dimensionality of the input embeddings) and `heads` (the number of attention heads) as input.
*   The `forward` method takes the `values`, `keys`, `queries`, and an optional `mask` as input. The mask is used to prevent the model from attending to padding tokens.
*   The input is projected into query, key, and value vectors using linear layers.  The heads are then split up, creating multiple attention mechanisms to attend to different parts of the input.
*   The attention weights are computed using the scaled dot-product attention formula.
*   The output is a weighted sum of the values, where the weights are determined by the attention mechanism.
*   The `einsum` function from the `torch` library is used to perform efficient matrix multiplication. This provides a more compact notation for matrix multiplication with multiple dimensions.
*   The code also includes a simple example of how to use the `SelfAttention` class.  It creates a dummy input and feeds it to the self-attention layer.

## 4) Follow-up question

How does the multi-head attention mechanism, which uses multiple self-attention layers in parallel, improve upon a single self-attention layer, and what are the trade-offs involved in using more heads?