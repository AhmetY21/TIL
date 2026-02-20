---
title: "Gated Recurrent Units (GRUs)"
date: "2026-02-20"
week: 8
lesson: 2
slug: "gated-recurrent-units-grus"
---

# Topic: Gated Recurrent Units (GRUs)

## 1) Formal definition (what is it, and how can we use it?)

A Gated Recurrent Unit (GRU) is a type of recurrent neural network (RNN) architecture. It is a variant of the Long Short-Term Memory (LSTM) network, and like LSTMs, it addresses the vanishing gradient problem that plagues traditional RNNs, allowing them to learn long-range dependencies in sequential data.  GRUs are simpler than LSTMs, having fewer parameters, which can lead to faster training times and better generalization performance, especially on smaller datasets.

**What is it?**

A GRU cell has two main gates:

*   **Update Gate (z_t):** Determines how much of the previous hidden state (h_{t-1}) should be updated with the new candidate activation.  A high update gate value means the previous hidden state is mostly forgotten, and the candidate activation is used. A low value means the previous hidden state is mostly kept.

*   **Reset Gate (r_t):** Determines how much of the previous hidden state (h_{t-1}) is used to compute the new candidate activation (h'_t). A low reset gate value indicates that the previous hidden state is mostly ignored when calculating the candidate activation, which allows the network to effectively forget past information.

**How can we use it?**

GRUs are used for tasks involving sequential data, such as:

*   **Natural Language Processing (NLP):**
    *   Machine Translation
    *   Text Summarization
    *   Sentiment Analysis
    *   Language Modeling
    *   Question Answering
*   **Speech Recognition**
*   **Time Series Prediction**
*   **Video Analysis**

The GRU receives an input x_t at each time step t, along with the previous hidden state h_{t-1}, and outputs a new hidden state h_t.  The calculations within a GRU cell are as follows:

1.  **Update Gate:** z_t = sigmoid(W_z * x_t + U_z * h_{t-1})
2.  **Reset Gate:** r_t = sigmoid(W_r * x_t + U_r * h_{t-1})
3.  **Candidate Activation:** h'_t = tanh(W_h * x_t + U_h * (r_t * h_{t-1}))
4.  **Hidden State Update:** h_t = (1 - z_t) * h_{t-1} + z_t * h'_t

Where:

*   x_t is the input at time t.
*   h_{t-1} is the hidden state from the previous time step.
*   z_t is the update gate.
*   r_t is the reset gate.
*   h'_t is the candidate activation.
*   h_t is the new hidden state.
*   W_z, U_z, W_r, U_r, W_h, and U_h are weight matrices that are learned during training.
*   sigmoid is the sigmoid activation function.
*   tanh is the hyperbolic tangent activation function.

## 2) Application scenario

**Scenario:** Sentiment Analysis of Movie Reviews

Imagine you want to build a system that can automatically classify movie reviews as either positive or negative. You can use a GRU for this task.

**Process:**

1.  **Data Preparation:** You would collect a dataset of movie reviews labeled with their sentiment (positive or negative).  The text of each review would need to be preprocessed, including tokenization (splitting the text into individual words or sub-word units), lowercasing, and potentially removing stop words (common words like "the," "a," "is").  Each token would then be converted into a numerical representation using techniques like word embeddings (e.g., Word2Vec, GloVe, or pre-trained embeddings like those from BERT).

2.  **Model Building:** You would create a GRU network. The input to the GRU would be a sequence of word embeddings representing the movie review.  The GRU would process the sequence one word at a time, updating its hidden state based on the current word and the previous hidden state.

3.  **Training:** The GRU network would be trained on the labeled dataset. During training, the network learns to adjust its weights to predict the correct sentiment for each review. The output of the GRU (the final hidden state) could be fed into a fully connected layer followed by a sigmoid activation function to produce a probability between 0 and 1, representing the likelihood of the review being positive.

4.  **Evaluation:** After training, the model is evaluated on a held-out test set to assess its performance. Metrics like accuracy, precision, recall, and F1-score can be used to evaluate the model's ability to correctly classify movie reviews.

5.  **Deployment:** Once the model is trained and evaluated, it can be deployed to classify new, unseen movie reviews.

The GRU is well-suited for this task because it can capture the sequential nature of language and learn long-range dependencies between words in the review, which can be crucial for determining the overall sentiment. For example, the presence of words like "not" or "but" can completely flip the sentiment of a sentence.

## 3) Python method (if possible)

```python
import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate GRU
        out, _ = self.gru(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Example usage:
# Assume we have input data of shape (batch_size, sequence_length, input_size)
input_size = 10  # Example: Each word is represented by a 10-dimensional embedding vector
hidden_size = 20 # Number of hidden units in the GRU
output_size = 2  # Example: Binary classification (positive/negative sentiment)
num_layers = 2   # Number of GRU layers

model = GRUModel(input_size, hidden_size, output_size, num_layers)

# Create a dummy input tensor
batch_size = 32
sequence_length = 50
input_tensor = torch.randn(batch_size, sequence_length, input_size)

# Pass the input through the model
output = model(input_tensor)

print(output.shape) # Output shape: (batch_size, output_size)
```

**Explanation:**

1.  **`GRUModel` class:** Defines the GRU model.
2.  **`__init__`:**  Initializes the GRU layer (`nn.GRU`) and a fully connected layer (`nn.Linear`). `batch_first=True` means the input tensor is expected to have the shape (batch_size, sequence_length, input_size).
3.  **`forward`:** Defines the forward pass through the network.
    *   `h0 = torch.zeros(...)`: Initializes the hidden state to zero. It's crucial to initialize the hidden state before processing each input sequence. `x.device` ensures the hidden state is on the same device (CPU or GPU) as the input tensor.
    *   `out, _ = self.gru(x, h0)`:  Passes the input `x` and the initial hidden state `h0` through the GRU layer. `out` contains the hidden states for all time steps, and the underscore `_` ignores the final hidden state (which is also returned by the GRU layer). The shape of `out` is (batch_size, sequence_length, hidden_size).
    *   `out = self.fc(out[:, -1, :])`: Takes the hidden state of the *last* time step (`out[:, -1, :]`) as input to the fully connected layer. The assumption here is that the last hidden state captures the relevant information from the entire sequence.
    *   `return out`: Returns the output of the fully connected layer.
4.  **Example usage:** Demonstrates how to create a GRU model, generate a dummy input tensor, and pass it through the model.

## 4) Follow-up question

How does the performance of GRUs compare to LSTMs in different NLP tasks, and what factors might influence the choice between using a GRU versus an LSTM?