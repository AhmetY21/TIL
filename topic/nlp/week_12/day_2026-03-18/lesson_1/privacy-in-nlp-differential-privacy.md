---
title: "Privacy in NLP (Differential Privacy)"
date: "2026-03-18"
week: 12
lesson: 1
slug: "privacy-in-nlp-differential-privacy"
---

# Topic: Privacy in NLP (Differential Privacy)

## 1) Formal definition (what is it, and how can we use it?)

Differential Privacy (DP) is a mathematical definition of privacy that provides a rigorous guarantee against information leakage from statistical databases or models. It aims to protect the privacy of individuals whose data is used in training a model, even when the model is publicly available.

**Definition:** An algorithm *A* satisfies ε-differential privacy if for any two adjacent datasets *D* and *D'* (differing by at most one record) and for all possible output sets *S*, the following holds:

`Pr[A(D) ∈ S] ≤ exp(ε) * Pr[A(D') ∈ S]`

Where:

*   *A* is the randomized algorithm (e.g., a machine learning model training process).
*   *D* and *D'* are adjacent datasets.
*   *S* is a set of possible outputs.
*   *ε* (epsilon) is the privacy budget, a non-negative real number that controls the privacy loss. A smaller *ε* provides stronger privacy guarantees but might impact model accuracy.
*   `Pr[...]` denotes the probability.
*   `exp(ε)` is the exponential function of *ε*.

**How it works:** DP works by adding calibrated noise to the algorithm's output or intermediate computations. This noise ensures that the output distribution remains similar regardless of whether a specific individual's data is included or excluded from the dataset. The amount of noise added is determined by the *sensitivity* of the algorithm, which is the maximum amount the algorithm's output can change when a single record is changed. The privacy budget *ε* then controls the scaling of this noise.

**Composition Theorems:** DP has useful composition properties:

*   **Sequential Composition:** If we apply *k* mechanisms with privacy parameters ε<sub>1</sub>, ε<sub>2</sub>, ..., ε<sub>k</sub> sequentially to the same dataset, the overall privacy cost is ∑ε<sub>i</sub>.
*   **Parallel Composition:** If we apply *k* mechanisms with privacy parameters ε<sub>1</sub>, ε<sub>2</sub>, ..., ε<sub>k</sub> to *disjoint* datasets, the overall privacy cost is max(ε<sub>1</sub>, ε<sub>2</sub>, ..., ε<sub>k</sub>).  This is important for distributed training.

**Using DP in NLP:** DP can be applied at various stages of the NLP pipeline:

*   **Data Collection:** By adding noise to the user data before it's stored.  However, this is often impractical due to the need to maintain utility.
*   **Model Training:** By adding noise to the gradients during training (Differential Private Stochastic Gradient Descent, or DP-SGD).  This is the most common approach.
*   **Model Output:** By adding noise to the model's predictions or outputs.

## 2) Application scenario

Consider a hospital wanting to train a language model to predict patient readmission rates based on clinical notes. The hospital wants to make this model publicly available for research purposes but must protect patient privacy.

Without DP, an adversary could potentially query the model in a way that reveals sensitive information about individual patients. For example, they could craft queries to determine if a particular patient's data was used in training the model by observing changes in the model's output with specific inputs.

By applying DP-SGD during model training, the hospital can add noise to the gradients calculated from the clinical notes. This ensures that the model's parameters are not overly influenced by any single patient's data. While the noise might slightly reduce the model's accuracy, it provides a formal guarantee that the model's outputs are not too sensitive to the inclusion or exclusion of any individual patient's record.  The resulting model is then published, allowing for research while maintaining privacy. They need to carefully choose *ε* to balance privacy and utility.  Higher *ε* gives better utility but less privacy.

## 3) Python method (if possible)

One popular library for implementing differential privacy in PyTorch and TensorFlow is **Opacus**. Here's a simplified example using Opacus with PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader, TensorDataset

# 1. Define a simple model
class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out


# 2. Generate some random data
input_size = 10
hidden_size = 5
output_size = 2
batch_size = 64
data_size = 1000

data = torch.randn(data_size, input_size)
labels = torch.randint(0, output_size, (data_size,))
dataset = TensorDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 3. Instantiate the model, optimizer, and loss function
model = SimpleModel(input_size, hidden_size, output_size)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# 4. Validate the model (important before using PrivacyEngine)
errors = ModuleValidator.validate(model, [(torch.empty(input_size),)]) # Pass example inputs as a tuple
if errors:
    print("Model validation failed! Fix the errors before proceeding.")
    print(errors)
    exit()
else:
    model = ModuleValidator.fix(model) # Fix model if possible, or use an inherently DP-compatible architecture.


# 5. Instantiate PrivacyEngine
privacy_engine = PrivacyEngine()

model, optimizer, dataloader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=dataloader,
    noise_multiplier=1.1,  # Adjust based on sensitivity and privacy budget
    max_grad_norm=1.0,       # Clip gradients to limit sensitivity
)

# 6. Training loop
epochs = 10
for epoch in range(epochs):
    for data, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

print("Training complete with differential privacy!")
```

**Explanation:**

*   The code defines a simple neural network.
*   It generates random data and creates a data loader.
*   It instantiates the model, optimizer, and loss function.
*   **Crucially, it uses `PrivacyEngine` from `opacus` to make the training process differentially private.**  This involves clipping gradients and adding noise.
*   `noise_multiplier` controls the amount of noise added.  A higher value provides more privacy but might reduce accuracy. The appropriate value depends on `epsilon` and the sensitivity of the computation.
*   `max_grad_norm` clips the gradients to limit the sensitivity of the training process.
*   `ModuleValidator` is used to ensure that the model is DP-compatible (e.g., no layers with unbound sensitivity).

**Note:**  This is a simplified example. Real-world applications of DP-SGD in NLP often require careful tuning of the parameters and more sophisticated techniques to balance privacy and utility. You'll also need to calculate or estimate the privacy budget (*ε*, *δ*) being spent during training. Opacus provides tools for this.

## 4) Follow-up question

How does the choice of the privacy budget (ε) and the noise multiplier impact the trade-off between privacy and utility in a DP-SGD training setup for NLP models? Specifically, what are the guidelines or best practices for selecting appropriate values for these parameters, and how can we evaluate the effectiveness of the chosen parameters in preserving privacy while maintaining acceptable model performance?