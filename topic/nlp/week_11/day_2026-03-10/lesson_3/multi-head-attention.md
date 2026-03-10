---
title: "Multi-Head Attention"
date: "2026-03-10"
week: 11
lesson: 3
slug: "multi-head-attention"
---

# Topic: Multi-Head Attention

## 1) Formal definition (what is it, and how can we use it?)

Multi-Head Attention is an attention mechanism that allows a model to attend to different parts of the input sequence and capture different types of relationships at different positions. It is an extension of the self-attention mechanism and is a key component of the Transformer architecture.

**What is it?**

The core idea behind Multi-Head Attention is to run the attention mechanism multiple times in parallel, with different learned linear projections of the query, key, and value matrices. Each "head" learns to attend to the input in a different way.

Formally:

1.  **Linear Projections:** Given the query (Q), key (K), and value (V) matrices (derived from the input sequence), we first project them *h* times using different learned linear transformations:

    *   Qᵢ = Q Wᵢ<sup>Q</sup>
    *   Kᵢ = K Wᵢ<sup>K</sup>
    *   Vᵢ = V Wᵢ<sup>V</sup>

    Where *Wᵢ<sup>Q</sup>*, *Wᵢ<sup>K</sup>*, and *Wᵢ<sup>V</sup>* are the learned weight matrices for the *i*-th head. These matrices project the original Q, K, and V into lower-dimensional spaces or different representation spaces.

2.  **Scaled Dot-Product Attention (per head):**  For each head *i*, we compute scaled dot-product attention:

    *   Attentionᵢ = softmax(Qᵢ Kᵢᵀ / √dₖ) Vᵢ

    Where dₖ is the dimensionality of the keys.  The scaling factor (√dₖ) prevents the dot products from growing too large, which can lead to small gradients after the softmax operation.

3.  **Concatenation and Final Projection:** The outputs of all the *h* attention heads are then concatenated:

    *   Concat = Concatenate(Attention₁, Attention₂, ..., Attentionₕ)

    Finally, the concatenated output is projected linearly to produce the final output:

    *   MultiHead(Q, K, V) = Concat W<sup>O</sup>

    Where W<sup>O</sup> is a learned weight matrix.

**How can we use it?**

Multi-Head Attention allows the model to capture richer and more diverse relationships within the input sequence. By attending to different parts of the input in different ways, each head can specialize in capturing specific patterns or dependencies. This leads to improved performance in tasks such as machine translation, text summarization, and question answering.  Crucially, the use of multiple attention heads helps mitigate the limitations of a single attention mechanism, preventing the model from getting stuck in local optima or focusing on only one particular type of relationship.  It also provides a form of model averaging within the attention process.

## 2) Application scenario

A primary application scenario for Multi-Head Attention is within the Transformer architecture, used extensively in Natural Language Processing. Specifically:

*   **Machine Translation:** In a translation task, the encoder uses multi-head self-attention to understand the relationships between words in the source sentence.  Each head can focus on different aspects, like syntactic dependencies or semantic relationships. The decoder also uses multi-head self-attention to focus on the target sentence being generated, as well as multi-head *encoder-decoder attention* which connects to the encoder outputs to align the source and target. This allows the decoder to attend to the relevant parts of the source sentence when generating each word in the target sentence.

*   **Text Summarization:**  When generating a summary, multi-head attention helps the model to identify and focus on the most important phrases and concepts in the input document, allowing it to generate a concise and informative summary. One head might focus on noun phrases, another on verbs and modifiers, and so on.

*   **Question Answering:** In a QA system, multi-head attention helps the model to attend to the relevant parts of both the question and the context passage, allowing it to identify the answer.  For instance, one head might focus on finding named entities, another on finding verbs related to the question.

Beyond NLP, Multi-Head Attention can be applied in other sequence modeling tasks like time series analysis and image recognition (with appropriate adaptations for image data).

## 3) Python method (if possible)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_k = d_model // num_heads  # Dimension of each head's key, query, and value

        self.W_Q = nn.Linear(d_model, d_model)  # Query projection
        self.W_K = nn.Linear(d_model, d_model)  # Key projection
        self.W_V = nn.Linear(d_model, d_model)  # Value projection
        self.W_O = nn.Linear(d_model, d_model)  # Output projection

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Compute scaled dot-product attention.

        Args:
            Q: Query tensor (batch_size, num_heads, seq_len, d_k)
            K: Key tensor (batch_size, num_heads, seq_len, d_k)
            V: Value tensor (batch_size, num_heads, seq_len, d_k)
            mask: Optional mask to prevent attention to certain positions (batch_size, seq_len, seq_len)

        Returns:
            Attention tensor (batch_size, num_heads, seq_len, d_k) and attention weights (batch_size, num_heads, seq_len, seq_len)
        """
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))  # Scaled dot-product

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask[:, None, :, :] == 0, float('-inf')) # Apply mask

        attn_weights = F.softmax(attn_scores, dim=-1)  # Softmax over the last dimension to get attention weights
        output = torch.matmul(attn_weights, V)  # Weighted sum of values

        return output, attn_weights


    def split_heads(self, x):
        """
        Split the last dimension into (num_heads, d_k).
        """
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2) # (batch_size, num_heads, seq_len, d_k)



    def combine_heads(self, x):
          """
          Combine the heads back into a single tensor.
          """
          batch_size, num_heads, seq_len, d_k = x.size()
          return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model) # (batch_size, seq_len, d_model)


    def forward(self, Q, K, V, mask=None):
        """
        Forward pass for multi-head attention.

        Args:
            Q: Query tensor (batch_size, seq_len, d_model)
            K: Key tensor (batch_size, seq_len, d_model)
            V: Value tensor (batch_size, seq_len, d_model)
            mask: Optional mask to prevent attention to certain positions (batch_size, seq_len, seq_len)

        Returns:
            Output tensor (batch_size, seq_len, d_model) and attention weights (batch_size, num_heads, seq_len, seq_len)
        """

        Q = self.W_Q(Q)  # Linear projection
        K = self.W_K(K)  # Linear projection
        V = self.W_V(V)  # Linear projection


        Q = self.split_heads(Q)  # Split heads
        K = self.split_heads(K)  # Split heads
        V = self.split_heads(V)  # Split heads



        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)  # Scaled dot-product attention for each head

        output = self.combine_heads(attn_output) # Concatenate heads

        output = self.W_O(output)  # Linear projection

        return output, attn_weights


if __name__ == '__main__':
    # Example usage
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8

    # Create random input tensors
    Q = torch.randn(batch_size, seq_len, d_model)
    K = torch.randn(batch_size, seq_len, d_model)
    V = torch.randn(batch_size, seq_len, d_model)

    # Create a mask (optional)
    mask = torch.ones(batch_size, seq_len, seq_len)  # All ones means no masking
    # Example of masking the future positions of the sequence
    mask = torch.tril(mask)

    # Instantiate the multi-head attention layer
    multihead_attn = MultiHeadAttention(d_model, num_heads)

    # Perform the forward pass
    output, attn_weights = multihead_attn(Q, K, V, mask)

    # Print the output shape
    print("Output shape:", output.shape)  # Expected: torch.Size([2, 10, 512])
    print("Attention weights shape:", attn_weights.shape) # Expected: torch.Size([2, 8, 10, 10])
```

## 4) Follow-up question

How can we adapt Multi-Head Attention for use with graph-structured data instead of sequential data? What modifications would be needed in the input representation, attention mechanism, and potential output processing?