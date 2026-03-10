---
title: "The Attention Mechanism (Bahdanau Attention)"
date: "2026-03-10"
week: 11
lesson: 1
slug: "the-attention-mechanism-bahdanau-attention"
---

# Topic: The Attention Mechanism (Bahdanau Attention)

## 1) Formal definition (what is it, and how can we use it?)

Bahdanau Attention, also known as additive attention, is a type of attention mechanism commonly used in sequence-to-sequence (seq2seq) models, particularly in Neural Machine Translation (NMT). It allows the decoder to focus on different parts of the input sequence when generating each word in the output sequence. Unlike basic seq2seq models that rely on a fixed-length context vector (the last hidden state of the encoder), attention allows the decoder to dynamically create a context vector that is a weighted sum of all the encoder's hidden states.

Formally, given:

*   **Encoder Hidden States:**  `H = [h1, h2, ..., hTx]`, where `hi` is the hidden state of the encoder at position `i`, and `Tx` is the length of the input sequence.
*   **Decoder Hidden State:** `st-1` is the hidden state of the decoder at time step `t-1`.

Bahdanau Attention works as follows:

1.  **Calculate Alignment Scores (Energies):** For each encoder hidden state `hi`, compute an alignment score `ei,t` that measures how well it matches the decoder hidden state `st-1`. This is done using a feedforward neural network (the *alignment model*):

    `ei,t = a(st-1, hi)`

    Where `a` is a single-layer feedforward neural network with a tanh activation function and parameter matrices `Wa`, `Ua`, and `va`:

    `ei,t = va^T tanh(Wa st-1 + Ua hi)`

2.  **Compute Attention Weights:** Normalize the alignment scores using a softmax function to obtain attention weights `αi,t`:

    `αi,t = exp(ei,t) / Σj exp(ej,t)`

    The attention weights represent the importance of each encoder hidden state for generating the current decoder output.

3.  **Calculate Context Vector:** Compute the context vector `ct` as a weighted sum of the encoder hidden states, using the attention weights:

    `ct = Σi αi,t hi`

    The context vector `ct` captures the relevant information from the entire input sequence for generating the current output word.

4.  **Generate Output:** The context vector `ct` is then concatenated with the decoder hidden state `st-1`, and passed through another neural network layer to generate the current decoder hidden state `st` and ultimately the predicted output word `yt`. This often involves combining the context vector and previous decoder state using an activation function, or using it as input to a softmax layer.

    `st = f(st-1, ct, yt-1)` (f is a non-linear function like LSTM or GRU)

    `yt = g(st, ct)` (g is a function predicting the output, often a softmax)

Using Bahdanau attention, the model can learn to focus on different parts of the input sequence for different output words. This addresses the bottleneck issue in standard seq2seq models where the entire input sequence is compressed into a single fixed-length vector.

## 2) Application scenario

The primary application scenario for Bahdanau Attention is **Neural Machine Translation (NMT)**. It significantly improved the performance of seq2seq models in translation tasks by allowing the decoder to selectively attend to different parts of the source sentence when generating the target sentence.

Beyond NMT, it can also be applied to other sequence-to-sequence tasks, such as:

*   **Text Summarization:** Focusing on the most important sentences in a document to generate a concise summary.
*   **Image Captioning:** Attending to different regions of an image to generate relevant captions.
*   **Speech Recognition:** Attending to specific segments of an audio signal to transcribe the spoken words.
*   **Chatbots and Dialogue Systems:** Paying attention to the important parts of the user's query to generate an appropriate response.

Any task where the output sequence depends on different parts of the input sequence can benefit from using attention mechanisms.

## 3) Python method (if possible)

Here's a simplified example of how to implement Bahdanau Attention in PyTorch:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, decoder_hidden, encoder_outputs):
        """
        Args:
            decoder_hidden (torch.Tensor): Shape (batch_size, hidden_size) - Previous decoder hidden state
            encoder_outputs (torch.Tensor): Shape (batch_size, seq_len, hidden_size) - All encoder hidden states

        Returns:
            context (torch.Tensor): Shape (batch_size, hidden_size) - Context vector
            attn_weights (torch.Tensor): Shape (batch_size, seq_len) - Attention weights
        """

        batch_size, seq_len, hidden_size = encoder_outputs.size()

        # (batch_size, seq_len, hidden_size)
        Wa_s = self.Wa(decoder_hidden).unsqueeze(1).repeat(1, seq_len, 1)
        Ua_h = self.Ua(encoder_outputs)

        # (batch_size, seq_len, 1)
        energies = self.Va(torch.tanh(Wa_s + Ua_h))

        # (batch_size, seq_len)
        attn_weights = F.softmax(energies.squeeze(-1), dim=1)

        # (batch_size, 1, seq_len) @ (batch_size, seq_len, hidden_size) -> (batch_size, 1, hidden_size)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attn_weights

if __name__ == '__main__':
    # Example usage
    batch_size = 32
    hidden_size = 256
    seq_len = 10

    # Create dummy inputs
    decoder_hidden = torch.randn(batch_size, hidden_size)
    encoder_outputs = torch.randn(batch_size, seq_len, hidden_size)

    # Initialize attention module
    attention = BahdanauAttention(hidden_size)

    # Calculate context vector and attention weights
    context, attn_weights = attention(decoder_hidden, encoder_outputs)

    print("Context vector shape:", context.shape)
    print("Attention weights shape:", attn_weights.shape)

```

Key improvements and explanations in the code:

*   **`nn.Module` Inheritance:**  The `BahdanauAttention` class inherits from `nn.Module`, making it a proper PyTorch module.
*   **Linear Layers:**  `nn.Linear` layers are used for the `Wa`, `Ua`, and `Va` transformations.  This is the proper way to implement learned linear transformations in PyTorch.
*   **`forward` Method:** The `forward` method defines the computation performed when you call the attention module.  This is essential.
*   **Broadcasting for Efficiency:**  `decoder_hidden` is unsqueezed and repeated to match the shape of `encoder_outputs` for efficient element-wise addition. This leverages PyTorch's broadcasting capabilities.  Critically important for performance.
*   **`tanh` Activation:** The `tanh` activation function is applied after the linear transformations.
*   **`F.softmax`:**  The `F.softmax` function is used to normalize the energies into attention weights along the correct dimension (`dim=1` - the sequence length dimension).
*   **`torch.bmm`:**  `torch.bmm` (batch matrix multiplication) is used for the weighted sum of the encoder outputs. This is the most efficient way to do this operation in batches.
*   **Shape Comments:**  Comments clearly indicate the shape of each tensor at each step. This makes the code easier to understand and debug.
*   **Example Usage:** The `if __name__ == '__main__':` block provides a complete example of how to use the `BahdanauAttention` module with dummy data.  This is invaluable for testing and understanding.
*   **Clear Variable Names:**  Meaningful variable names (e.g., `Wa_s`, `Ua_h`) are used to improve readability.

## 4) Follow-up question

How does Bahdanau Attention compare to Luong Attention (another common type of attention mechanism)? What are the key differences in their formulations and performance characteristics?