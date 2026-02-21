---
title: "The Attention Mechanism (Bahdanau Attention)"
date: "2026-02-21"
week: 8
lesson: 2
slug: "the-attention-mechanism-bahdanau-attention"
---

# Topic: The Attention Mechanism (Bahdanau Attention)

## 1) Formal definition (what is it, and how can we use it?)

Bahdanau Attention, also known as Additive Attention, is an attention mechanism introduced by Bahdanau et al. in their 2014 paper "Neural Machine Translation by Jointly Learning to Align and Translate." It addresses a key limitation of traditional sequence-to-sequence models (like those using LSTMs or GRUs) with a fixed-length context vector: the inability to effectively handle long sequences.

Instead of forcing the encoder to compress the entire input sequence into a single fixed-length vector, Bahdanau attention allows the decoder to "attend" to different parts of the input sequence when generating each output token.  This means the decoder has access to the entire encoded input sequence throughout the decoding process.

Here's the mathematical breakdown of how it works:

1. **Encoder:** The encoder processes the input sequence *x = (x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>T<sub>x</sub></sub>)* and generates a sequence of hidden states *h = (h<sub>1</sub>, h<sub>2</sub>, ..., h<sub>T<sub>x</sub></sub>)*. Each *h<sub>i</sub>* represents the encoder's understanding of the input at position *i*.

2. **Decoder:**  At decoding step *t*, the decoder has a hidden state *s<sub>t-1</sub>*.

3. **Attention Weights:** The core of the attention mechanism is calculating the attention weights *α<sub>ti</sub>* for each encoder hidden state *h<sub>i</sub>*.  This determines how much attention the decoder should pay to each input element when generating the output at time step *t*.

   * **Alignment Score (e<sub>ti</sub>):** A score is calculated for each encoder hidden state *h<sub>i</sub>* based on its relevance to the decoder's previous hidden state *s<sub>t-1</sub>*:

     *e<sub>ti</sub> = a(s<sub>t-1</sub>, h<sub>i</sub>)*

     The function *a* is a feed-forward neural network, typically a single-layer perceptron, jointly trained with the rest of the model. This network learns to measure the alignment between the previous decoder state and the encoder states. It transforms the concatenated decoder state and encoder hidden state and applies a tanh activation followed by a linear projection and then computes a score.  More formally:
     *e<sub>ti</sub> = v<sup>T</sup> tanh(W<sub>a</sub>[s<sub>t-1</sub>; h<sub>i</sub>])*

     Where:
         * *v* and *W<sub>a</sub>* are learnable parameters of the attention network.
         * [;] represents concatenation.

   * **Normalization (α<sub>ti</sub>):** The alignment scores are then normalized using a softmax function to produce the attention weights:

     *α<sub>ti</sub> = exp(e<sub>ti</sub>) / ∑<sub>k=1</sub><sup>T<sub>x</sub></sup> exp(e<sub>tk</sub>)*

     This ensures the attention weights sum to 1, representing a probability distribution over the input positions.  Therefore, *α<sub>ti</sub>* represents the weight the decoder gives to the *i*-th input element when generating the *t*-th output element.

4. **Context Vector (c<sub>t</sub>):** The attention weights are then used to compute a context vector *c<sub>t</sub>*, which is a weighted sum of the encoder hidden states:

   *c<sub>t</sub> = ∑<sub>i=1</sub><sup>T<sub>x</sub></sup> α<sub>ti</sub>h<sub>i</sub>*

   This context vector represents a summary of the relevant parts of the input sequence for the decoder at time step *t*.

5. **Decoder Update:** The context vector *c<sub>t</sub>* is then combined with the previous decoder state *s<sub>t-1</sub>* (often concatenated) and used to compute the current decoder state *s<sub>t</sub>* and the output *y<sub>t</sub>*:

   *s<sub>t</sub> = f(s<sub>t-1</sub>, y<sub>t-1</sub>, c<sub>t</sub>)*
   *y<sub>t</sub> = g(s<sub>t</sub>, c<sub>t</sub>, y<sub>t-1</sub>)*

   Where *f* and *g* are typically functions implemented by LSTMs or GRUs. *g* might involve a softmax layer for classification tasks like machine translation.

In essence, Bahdanau attention allows the decoder to dynamically focus on different parts of the input sequence when generating each output token.  This significantly improves performance, especially for longer sequences, compared to fixed-length context vector approaches.  The key is the learnable alignment score function *a* that allows the model to learn which parts of the input are most relevant for each output token.

## 2) Application scenario

A classic application scenario is **Neural Machine Translation (NMT)**. Imagine translating a long English sentence into French.

Without attention, the encoder would try to compress the entire English sentence into a single fixed-length vector. As the sentence gets longer, information gets lost in this compression, leading to poorer translation quality, especially at the end of the sentence.

With Bahdanau attention, the decoder, when generating each French word, can attend to different parts of the English sentence.  For example, when translating "The cat sat on the mat," the decoder, when generating "chat," can focus on "cat," and when generating "sur," can focus on "on."

This allows the model to handle longer sentences more effectively and produce more accurate translations.  It also provides interpretability: by visualizing the attention weights, we can see which parts of the input the model is focusing on when generating each output token. This helps us understand how the model is working and can be used for debugging.

Another application can be found in **image captioning**. Instead of encoding the entire image into a single vector, different regions of the image are represented by the encoder. The decoder then attends to relevant regions when generating each word in the caption. For example, when generating the word "dog," the decoder might attend to the region of the image containing the dog.

## 3) Python method (if possible)

Here's a simplified example using PyTorch to illustrate the core attention mechanism (without the full encoder-decoder structure):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(encoder_hidden_size, decoder_hidden_size)  # Attention weights
        self.Ua = nn.Linear(decoder_hidden_size, decoder_hidden_size)  # decoder hidden state weights
        self.Va = nn.Linear(decoder_hidden_size, 1)  # final score layer

    def forward(self, encoder_outputs, decoder_hidden):
        """
        Args:
            encoder_outputs: (batch_size, seq_len, encoder_hidden_size) - Encoder hidden states
            decoder_hidden: (batch_size, decoder_hidden_size) - Previous decoder hidden state

        Returns:
            context_vector: (batch_size, encoder_hidden_size) - Weighted sum of encoder outputs
            attention_weights: (batch_size, seq_len) - Attention weights for each encoder output
        """

        batch_size, seq_len, encoder_hidden_size = encoder_outputs.size()

        # Alignment scores: e_ti = v^T tanh(W_a h_i + U_a s_{t-1})
        Wa_h = self.Wa(encoder_outputs)  # (batch_size, seq_len, decoder_hidden_size)
        Ua_s = self.Ua(decoder_hidden).unsqueeze(1)  # (batch_size, 1, decoder_hidden_size) - Add seq_len dimension to broadcast across seq_len
        Ua_s = Ua_s.repeat(1,seq_len,1) # (batch_size, seq_len, decoder_hidden_size)

        tanh_output = torch.tanh(Wa_h + Ua_s)  # (batch_size, seq_len, decoder_hidden_size)
        alignment_scores = self.Va(tanh_output).squeeze(2)  # (batch_size, seq_len)

        # Attention weights: alpha_ti = softmax(e_ti)
        attention_weights = F.softmax(alignment_scores, dim=1)  # (batch_size, seq_len)

        # Context vector: c_t = sum(alpha_ti * h_i)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1) # (batch_size, encoder_hidden_size)


        return context_vector, attention_weights


# Example usage:
encoder_hidden_size = 256
decoder_hidden_size = 128
batch_size = 32
seq_len = 20

# Dummy encoder outputs and decoder hidden state
encoder_outputs = torch.randn(batch_size, seq_len, encoder_hidden_size)
decoder_hidden = torch.randn(batch_size, decoder_hidden_size)

# Initialize the attention mechanism
attention = BahdanauAttention(encoder_hidden_size, decoder_hidden_size)

# Calculate the context vector and attention weights
context_vector, attention_weights = attention(encoder_outputs, decoder_hidden)

print("Context Vector Shape:", context_vector.shape)
print("Attention Weights Shape:", attention_weights.shape)
```

Key points about the code:

*   `BahdanauAttention` class encapsulates the attention mechanism.
*   `forward` method calculates the context vector and attention weights.
*   The `Wa`, `Ua`, and `Va` are linear layers that perform the necessary transformations.
*   `F.softmax` normalizes the alignment scores into attention weights.
*   `torch.bmm` performs batch matrix multiplication to compute the context vector.

This is a simplified version. A complete NMT system would involve an encoder, a decoder, and the integration of this attention mechanism into the decoding process.  You'd typically incorporate the `context_vector` into the decoder's hidden state update or output generation.

## 4) Follow-up question

How does Bahdanau Attention compare to other attention mechanisms, such as Dot-Product Attention or Self-Attention (Transformer attention), in terms of computational complexity, performance on different types of tasks, and interpretability? In what scenarios might one be preferred over the others?