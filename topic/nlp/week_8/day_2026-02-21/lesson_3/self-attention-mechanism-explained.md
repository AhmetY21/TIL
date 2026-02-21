---
title: "Self-Attention Mechanism Explained"
date: "2026-02-21"
week: 8
lesson: 3
slug: "self-attention-mechanism-explained"
---

# Topic: Self-Attention Mechanism Explained

## 1) Formal definition (what is it, and how can we use it?)

The self-attention mechanism, also known as intra-attention, is an attention mechanism relating different positions of a single sequence to compute a representation of the sequence. Unlike traditional attention which focuses on aligning a source sequence to a target sequence (e.g., in machine translation), self-attention focuses on finding relationships within the *same* sequence.

In essence, self-attention allows a model to attend to different parts of the input sequence when processing each part. This is crucial for capturing long-range dependencies and understanding the context of each element in the sequence.

The core idea is to calculate a weighted sum of the input sequence's representations, where the weights determine how much "attention" each part of the sequence should receive.  These weights are calculated based on the similarity between pairs of elements in the sequence.

Here's a breakdown of the key steps involved in self-attention:

1. **Input Embedding:** The input sequence is first embedded into a set of vectors.  Let's say we have an input sequence of *n* words represented as vectors *X = (x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>n</sub>)*, where each *x<sub>i</sub>* is an embedding vector.

2. **Key, Query, and Value Projections:**  Three different linear transformations are applied to the input embeddings to create three new sets of vectors: Queries (Q), Keys (K), and Values (V).  These are calculated as follows:

   * Q = XW<sub>Q</sub>
   * K = XW<sub>K</sub>
   * V = XW<sub>V</sub>

   Where W<sub>Q</sub>, W<sub>K</sub>, and W<sub>V</sub> are learnable weight matrices.  Each *x<sub>i</sub>* is transformed into a corresponding *q<sub>i</sub>*, *k<sub>i</sub>*, and *v<sub>i</sub>*. The query represents what we're looking for, the key represents what we're comparing against, and the value represents the information we want to extract.

3. **Attention Scores:** The attention scores are calculated by taking the dot product of the query vectors with the key vectors. This effectively measures the similarity between each query and each key.

   * Attention Scores = QK<sup>T</sup>

4. **Scaling:** The attention scores are scaled down by the square root of the dimension of the key vectors (d<sub>k</sub>). This prevents the dot products from becoming too large, which can lead to vanishing gradients during training.

   * Scaled Attention Scores = (QK<sup>T</sup>) / sqrt(d<sub>k</sub>)

5. **Softmax:** A softmax function is applied to the scaled attention scores to normalize them into probabilities.  This ensures that the attention weights sum to 1 for each query.

   * Attention Weights = softmax((QK<sup>T</sup>) / sqrt(d<sub>k</sub>))

6. **Weighted Sum:** Finally, the attention weights are used to compute a weighted sum of the value vectors.  This weighted sum represents the output of the self-attention mechanism.

   * Output = Attention Weights * V

The output is a new representation of the input sequence, where each element is now aware of the other elements in the sequence, weighted by their relevance.  This output can then be fed into subsequent layers of the neural network.

We can use self-attention in many NLP tasks, including machine translation, text summarization, sentiment analysis, and question answering. Its ability to model long-range dependencies makes it especially effective for understanding context in long sequences.

## 2) Application scenario

Consider the sentence: "The cat sat on the mat because it was comfortable."

Without self-attention, a traditional recurrent neural network (RNN) might struggle to understand what "it" refers to. Specifically, it might have difficulty maintaining the context of "cat" over several words.

With self-attention, the model can explicitly learn the relationships between "it" and other words in the sentence. When processing the word "it," the self-attention mechanism will assign high attention weights to "cat" and potentially other relevant words like "mat" and "comfortable".  This allows the model to understand that "it" refers to the "mat," because the mat is what is comfortable.  This is because "mat" will have a larger attention weight when computing the representation for the word "it".

In general, self-attention is beneficial in any scenario where understanding the relationships between different parts of a sequence is important. This includes tasks such as:

*   **Machine Translation:** Aligning words in the source and target languages, and capturing long-range dependencies in both.
*   **Text Summarization:** Identifying the most important sentences and phrases to include in the summary.
*   **Question Answering:** Finding the relevant parts of the context document that answer the question.
*   **Natural Language Inference (NLI):** Determining the relationship (entailment, contradiction, or neutrality) between two sentences.
*   **Sentiment Analysis:** Understanding the overall sentiment of a text, taking into account the relationships between different words and phrases.

## 3) Python method (if possible)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads # Ensure even split
        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size needs to be divisible by num_heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(num_heads * self.head_dim, embed_size)


    def forward(self, values, keys, query, mask=None):
        # Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into self.num_heads pieces
        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        query = query.reshape(N, query_len, self.num_heads, self.head_dim)


        values = self.values(values)  # (N, value_len, num_heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, num_heads, head_dim)
        queries = self.queries(query)  # (N, query_len, num_heads, head_dim)



        # Scaled dot-product attention
        # Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, num_heads, head_dim)
        # keys shape: (N, key_len, num_heads, head_dim)
        # energy shape: (N, num_heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20")) #mask irrelevant items so they have negligible influence

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # attention shape: (N, num_heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.num_heads * self.head_dim
        )
        # attention shape: (N, num_heads, query_len, key_len)
        # values shape: (N, value_len, num_heads, head_dim)
        # out shape: (N, query_len, num_heads, head_dim) then flatten to (N, query_len, embed_size)

        # Linear layer to send it into a residual connection
        out = self.fc_out(out)
        # out shape: (N, query_len, embed_size)
        return out


if __name__ == '__main__':
    # Example usage
    embed_size = 512
    num_heads = 8
    seq_len = 64 # Sequence length
    batch_size = 32 # Number of examples to pass

    # Assume we have input sequence embeddings
    values = torch.randn((batch_size, seq_len, embed_size))
    keys = torch.randn((batch_size, seq_len, embed_size))
    query = torch.randn((batch_size, seq_len, embed_size))

    attention = SelfAttention(embed_size, num_heads)
    output = attention(values, keys, query)

    print(f"Input shape: {query.shape}")
    print(f"Output shape: {output.shape}") # Expected (batch_size, seq_len, embed_size)
```

This code implements a single self-attention layer.  The `forward` function calculates the attention weights and applies them to the value vectors to produce the output. The `torch.einsum` function provides a concise way to express tensor contractions. The example usage shows how to create and use the `SelfAttention` module with random input tensors. Remember that in a real-world application, `values`, `keys`, and `query` would typically be derived from the same input sequence embeddings.  Masking (optional) allows certain positions (e.g., padding tokens) to be ignored in the attention computation.
## 4) Follow-up question

How does the "multi-head attention" mechanism build upon the basic self-attention mechanism described above, and why is it beneficial in practice?