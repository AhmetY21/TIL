---
title: "Privacy in NLP (Differential Privacy)"
date: "2026-02-28"
week: 9
lesson: 6
slug: "privacy-in-nlp-differential-privacy"
---

# Topic: Privacy in NLP (Differential Privacy)

## 1) Formal definition (what is it, and how can we use it?)

Differential Privacy (DP) is a mathematical framework that provides a rigorous guarantee that the presence or absence of any individual's data in a dataset will not substantially alter the outcome of an analysis performed on that dataset.  This means an adversary, even with auxiliary information, can't infer whether a specific individual's data was used to train a model or compute statistics.

Formally, a randomized algorithm *M* satisfies (ε, δ)-differential privacy if for any two neighboring datasets *D* and *D'* differing by at most one record, and for all possible output sets *S*:

P(*M*(D) ∈ S) ≤ exp(ε) * P(*M*(D') ∈ S) + δ

Where:

*   *M* is the randomized algorithm (e.g., a trained model, a query on a database).
*   *D* and *D'* are neighboring datasets.  Neighboring datasets are identical except for the presence or absence of one individual's data.
*   *S* is any possible subset of the algorithm's output space.
*   ε (epsilon) is the privacy budget.  It controls the level of privacy loss. A smaller ε indicates stronger privacy.  It is the multiplicative factor in the privacy bound.
*   δ (delta) is the probability that the privacy guarantee fails.  It represents the additive loss and is usually a very small number (often 0).  It allows for a small probability of catastrophic privacy loss.  If δ=0, the mechanism is said to satisfy ε-differential privacy (pure differential privacy).

**How can we use it?**

DP is achieved by injecting noise into the process.  This noise can be added to:

*   **Input Data (Input Perturbation):** Adding noise directly to the raw data.  Less common in NLP.
*   **Model Parameters (Output Perturbation):** Adding noise to the model parameters after training.
*   **Objective Function (Objective Perturbation):** Adding noise to the objective function during training.  Commonly used in differentially private SGD (DP-SGD).
*   **Query Results (Output Perturbation):** Adding noise to the results of queries on a dataset.

**Key concepts for application:**

*   **Sensitivity:** The maximum amount that a query's output can change when a single individual's data is added or removed from the dataset.  This is crucial for determining the amount of noise needed.
*   **Privacy Budget Composition:** When multiple differentially private mechanisms are applied, the privacy loss accumulates.  There are composition theorems that specify how to calculate the overall privacy loss (ε, δ) after multiple applications. There are two main composition theorems: basic composition and advanced composition. Advanced composition usually gives tighter bounds.

## 2) Application scenario

**Scenario:** Consider a hospital using patient medical records to train a language model for predicting patient readmission rates based on doctors' notes. They want to share this model with external researchers without revealing sensitive information about individual patients.

**Application of DP:** The hospital can use differentially private training methods (like DP-SGD) to train the language model. This involves adding noise to the gradients during training. The magnitude of the noise is carefully calibrated based on the sensitivity of the gradient calculation and the desired privacy parameters (ε and δ). By releasing the differentially private model, the hospital allows researchers to use the model for prediction and analysis while ensuring that the presence or absence of any single patient's data has a limited impact on the model's output. This protects patient privacy while enabling valuable research.  Specifically, we protect that a researcher cannot determine if a specific patient's data was used to train the model or not.

## 3) Python method (if possible)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager

# Example model (replace with your actual NLP model)
class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        return self.fc(output[-1])  # Only using the last time step's output


# Example usage with Opacus (a popular DP library)
def train_dp_model(model, train_loader, optimizer, epochs, epsilon, delta, max_grad_norm, batch_size, sample_rate):
    """Trains a model using differentially private SGD (DP-SGD) with Opacus."""

    # Validate model
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        print("Model validation errors:")
        for error in errors:
            print(f"- {error}")
        raise ValueError("Model is not compatible with differential privacy.  Fix validation errors first.")

    model.train()  # Set the model to training mode
    privacy_engine = PrivacyEngine(accountant="rdp", secure_mode=False)
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=epochs,
        epsilon=epsilon,
        delta=delta,
        max_grad_norm=max_grad_norm,
        sample_rate=sample_rate,
    )
    print(f"Using secure model: {privacy_engine.secure_mode}") #secure_mode is false by default
    # Train the model (using the DP-SGD optimizer)
    for epoch in range(epochs):
        losses = []
        for i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(
            f"Epoch {epoch+1}: \t"
            f"Loss: {sum(losses)/len(losses):.6f} \t"
            f"(ε = {privacy_engine.get_epsilon(delta):.2f}, δ = {delta})"
        )



# Example setup (replace with your actual data and model)
if __name__ == "__main__":
    # Dummy data and setup
    vocab_size = 1000
    embedding_dim = 128
    hidden_dim = 256
    output_dim = 10
    model = SimpleLSTM(vocab_size, embedding_dim, hidden_dim, output_dim)

    # Dummy data loader (replace with your actual data loader)
    from torch.utils.data import DataLoader, TensorDataset
    data = torch.randint(0, vocab_size, (100, 20))  # 100 samples, sequence length 20
    target = torch.randint(0, output_dim, (100,))
    dataset = TensorDataset(data, target)
    batch_size = 16 #keep batch_size fixed. This is because the sample_rate is determined from batch_size
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # DP parameters
    epsilon = 1.0  # Privacy budget
    delta = 1e-5  # Failure probability
    max_grad_norm = 1.2 #Grad clip
    sample_rate = batch_size/len(dataset)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 5
    train_dp_model(model, train_loader, optimizer, num_epochs, epsilon, delta, max_grad_norm, batch_size, sample_rate)
```

**Explanation:**

*   The code uses the `opacus` library to implement DP-SGD.
*   The `PrivacyEngine` is initialized with the desired privacy parameters (ε and δ).
*   `privacy_engine.make_private_with_epsilon()` modifies the model and optimizer to perform DP-SGD. The sample_rate argument needs to be specified so that epsilon and delta can be calculated accurately. The other similar method, `make_private` is deprecated.
*   During training, gradients are clipped, and noise is added to them before updating the model parameters.
*   The `ModuleValidator` checks if the model is compatible with Opacus.
*   `privacy_engine.get_epsilon(delta)` allows us to track the privacy budget consumption during training.

**Important Notes:**

*   This is a simplified example.  Real-world NLP models are typically much more complex.
*   Choosing appropriate values for ε, δ, and the noise multiplier requires careful consideration and experimentation. There is a trade-off between privacy and utility. Lower epsilon means higher privacy, but also lower utility, since we add more noise.
*   Gradient clipping is essential for controlling the sensitivity of the gradient calculation.
*   Opacus also supports secure aggregation (using a trusted execution environment) for increased security.

## 4) Follow-up question

How does the choice of the privacy budget (ε and δ) affect the utility of the trained model in NLP tasks, and what strategies can be used to find a good balance between privacy and utility in practice? Consider both text classification and text generation scenarios.