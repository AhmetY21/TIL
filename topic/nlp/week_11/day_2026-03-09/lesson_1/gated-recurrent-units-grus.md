---
title: "Gated Recurrent Units (GRUs)"
date: "2026-03-09"
week: 11
lesson: 1
slug: "gated-recurrent-units-grus"
---

# Topic: Gated Recurrent Units (GRUs)

## 1) Formal definition (what is it, and how can we use it?)

Gated Recurrent Units (GRUs) are a type of recurrent neural network (RNN) architecture designed to address the vanishing gradient problem often encountered when training standard RNNs on long sequences. They act as "gates" that control the flow of information within the sequence, selectively remembering or forgetting information at each time step. This enables GRUs to capture long-range dependencies more effectively than traditional RNNs.

Specifically, a GRU cell has two main gates:

*   **Update gate (z_t):** Determines how much of the *previous* cell state should be updated with the *new* candidate cell state. A value close to 1 means the past information is heavily considered, and a value close to 0 means the past information is mostly discarded.

*   **Reset gate (r_t):** Determines how much of the *previous* hidden state should be ignored. A value close to 0 forces the hidden state to ignore the previous state, effectively "resetting" it. A value close to 1 allows the previous hidden state to be considered.

The GRU updates are defined by the following equations:

1.  **Update gate:**  `z_t = σ(W_z x_t + U_z h_{t-1} + b_z)`
2.  **Reset gate:**   `r_t = σ(W_r x_t + U_r h_{t-1} + b_r)`
3.  **Candidate hidden state:** `h'_t = tanh(W_h x_t + U_h (r_t * h_{t-1}) + b_h)`
4.  **Hidden state:** `h_t = (1 - z_t) * h_{t-1} + z_t * h'_t`

Where:

*   `x_t` is the input at time step `t`.
*   `h_{t-1}` is the hidden state at the previous time step.
*   `h'_t` is the candidate hidden state at time step `t`.
*   `h_t` is the hidden state at time step `t`.
*   `W_z`, `U_z`, `b_z`, `W_r`, `U_r`, `b_r`, `W_h`, `U_h`, `b_h` are the weight matrices and bias vectors.
*   `σ` is the sigmoid activation function.
*   `tanh` is the hyperbolic tangent activation function.
*   `*` represents element-wise multiplication (Hadamard product).

We use GRUs as a drop-in replacement for simple RNN layers in tasks such as:
* Machine translation
* Speech recognition
* Time series prediction
* Text generation
* Sentiment analysis

## 2) Application scenario

Consider a sentiment analysis task where we want to determine the sentiment of a movie review. A movie review might contain long-range dependencies, where the meaning of a word depends on words that appeared much earlier in the review.

For instance: "The acting was terrible and the plot was confusing, but the *ending* was surprisingly *good*." A standard RNN might struggle to connect "good" with the negative parts of the sentence. A GRU, however, can use its gates to remember the initial negative sentiments and then correctly adjust its understanding when it encounters "good" near the end. This allows the GRU to more accurately classify the overall sentiment of the review.

Another scenario is time series prediction. For instance, predicting stock prices. The current stock price may depend not only on the immediate past values, but also on values from weeks or months prior. A GRU is better suited to capture these long-term dependencies.

## 3) Python method (if possible)

```python
import torch
import torch.nn as nn

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True) # batch_first=True means input tensors are provided as (batch, seq, feature)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # Forward propagate GRU
        out, _ = self.gru(x, h0.detach()) # detaching the hidden state from the computation graph

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Example Usage
input_dim = 10  # Number of features in the input
hidden_dim = 20 # Number of features in the hidden state
num_layers = 2  # Number of recurrent layers
output_dim = 1 # Number of output features

# Create a GRU model
model = GRUNet(input_dim, hidden_dim, output_dim, num_layers)

# Example input
batch_size = 32
seq_len = 50
input_tensor = torch.randn(batch_size, seq_len, input_dim)

# Pass the input through the model
output = model(input_tensor)

print(output.shape) #Expected: torch.Size([32, 1])
```

This code snippet shows how to implement a GRU network using PyTorch. The `nn.GRU` module handles the GRU cell computations, and the `batch_first=True` argument allows the input tensor to be of shape (batch size, sequence length, input dimension). The output of the GRU is then passed to a fully connected layer (`nn.Linear`) to produce the final output.

## 4) Follow-up question

How do GRUs compare to LSTMs (Long Short-Term Memory networks)? What are the trade-offs between using GRUs and LSTMs in different situations?