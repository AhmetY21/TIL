---
title: "Multi-Head Attention"
date: "2026-02-21"
week: 8
lesson: 4
slug: "multi-head-attention"
---

# Topic: Multi-Head Attention

## 1) Formal definition (what is it, and how can we use it?)

Multi-Head Attention is an attention mechanism used in neural networks, particularly in the Transformer architecture, that allows the model to attend to different parts of the input sequence with different learned linear projections.  It's an extension of the scaled dot-product attention mechanism.

Here's the breakdown:

*   **Scaled Dot-Product Attention:** The foundation. It calculates attention weights by computing the dot product of a query (Q) with all keys (K), scales the result by the square root of the dimension of the keys (to prevent exploding gradients), and applies a softmax function to obtain probabilities. These probabilities represent the importance of each key for the given query. The final output is a weighted sum of the values (V), where the weights are the attention probabilities. Mathematically:

    `Attention(Q, K, V) = softmax((Q K^T) / sqrt(d_k)) V`

    where `d_k` is the dimension of the keys.

*   **Multiple Heads:** Instead of performing a single attention calculation, Multi-Head Attention performs *h* independent scaled dot-product attentions in parallel. Each attention head uses different learned linear projections to transform the input queries, keys, and values.  Specifically:

    1.  The input queries (Q), keys (K), and values (V) are each linearly projected *h* times using different weight matrices.  This means we learn *h* different weight matrices `W_i^Q`, `W_i^K`, and `W_i^V` for each head *i*.

    2.  Each projected (Q, K, V) tuple is then fed into the scaled dot-product attention mechanism independently, producing *h* different attention outputs.

    3.  The *h* attention outputs are concatenated along a specified dimension.

    4.  The concatenated output is linearly projected again using a final weight matrix `W^O` to produce the final Multi-Head Attention output.

    Mathematically:

    `MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O`

    where:

    `head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)`

    and `W_i^Q`, `W_i^K`, `W_i^V`, and `W^O` are learnable parameter matrices.

**How can we use it?**

Multi-Head Attention allows the model to simultaneously attend to different parts of the input sequence and to capture different relationships between words or sub-words. Each head can learn different aspects of the input, such as syntactic dependencies, semantic relationships, or long-range dependencies.  This makes the model more powerful and capable of capturing more complex patterns in the data. It's crucial for understanding the context of a word in a sentence.  It is used in encoder and decoder architectures.

## 2) Application scenario

A primary application scenario for Multi-Head Attention is in **machine translation**. Consider translating the sentence "The animal didn't cross the street because it was too tired."

*   A single attention mechanism might struggle to correctly associate "it" with either "animal" or "street."

*   Multi-Head Attention allows some heads to focus on syntactic dependencies (e.g., identifying the subject of the verb "was"), while other heads can focus on semantic relationships (e.g., understanding that animals are more likely to be tired than streets). One head might correctly establish that "it" refers to the animal, leading to a correct translation that reflects that meaning. Without this nuanced attention, a translation engine might incorrectly translate the sentence, attributing the tiredness to the street.

Other application scenarios include:

*   **Text Summarization:** Identifying the most important sentences or phrases in a document.
*   **Question Answering:** Finding the answer to a question within a given context.
*   **Sentiment Analysis:** Understanding the overall sentiment expressed in a piece of text.
*   **Image Captioning:** Generating textual descriptions of images (with attention mechanisms applied to different regions of the image).
*   **Self-Attention in Language Models:** Modeling relationships between words in a sentence for tasks like language generation.

## 3) Python method (if possible)

While implementing Multi-Head Attention from scratch can be complex, deep learning frameworks like TensorFlow and PyTorch provide built-in layers or convenient functions to facilitate its use. Here's an example using TensorFlow/Keras:

```python
import tensorflow as tf
from tensorflow.keras.layers import Layer

class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)


    def scaled_dot_product_attention(self, q, k, v, mask):
        """Calculate the attention weights."""
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)  

        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        output = tf.matmul(attention_weights, v)

        return output, attention_weights


    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])


    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

# Example Usage:
# Assuming you have sequences q, k, v
# d_model = 512 # Embedding dimensionality
# num_heads = 8 # Number of attention heads
# batch_size = 64
# seq_len = 20

# q = tf.random.normal((batch_size, seq_len, d_model))
# k = tf.random.normal((batch_size, seq_len, d_model))
# v = tf.random.normal((batch_size, seq_len, d_model))

# mha = MultiHeadAttention(d_model, num_heads)
# output, attention_weights = mha(v, k, q, mask=None)

# print(f"Output shape: {output.shape}") # Expected: (batch_size, seq_len, d_model)
# print(f"Attention weights shape: {attention_weights.shape}") # Expected: (batch_size, num_heads, seq_len, seq_len)
```

**Explanation:**

1.  **`MultiHeadAttention(Layer)` Class:** Defines a Keras layer for Multi-Head Attention.
2.  **`__init__`:** Initializes the layer with the model dimension (`d_model`), number of heads (`num_heads`), and calculates the dimension of each head (`depth`). Also defines the linear projection layers (`wq`, `wk`, `wv`, and `dense`).
3.  **`scaled_dot_product_attention`:** Implements the scaled dot-product attention mechanism. Includes optional masking to prevent attending to padding tokens or future tokens.
4.  **`split_heads`:** Splits the last dimension (d_model) into multiple heads (num_heads, depth).
5.  **`call`:** The main function that performs the Multi-Head Attention. It projects the queries, keys, and values; splits them into heads; calculates the scaled dot-product attention for each head; concatenates the results; and projects the concatenated output to the final dimension.
6.  **Example Usage:** Shows how to create an instance of the `MultiHeadAttention` layer and use it with sample input tensors.

This code provides a basic implementation. You can modify it further by adding dropout layers, layer normalization, or other techniques to improve performance.

## 4) Follow-up question

How does the number of attention heads affect the performance and computational cost of a model using Multi-Head Attention, and what are some practical considerations when choosing the number of heads?