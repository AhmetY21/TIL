---
title: "Model Quantization and Pruning"
date: "2026-03-17"
week: 12
lesson: 3
slug: "model-quantization-and-pruning"
---

# Topic: Model Quantization and Pruning

## 1) Formal definition (what is it, and how can we use it?)

**Model Quantization:**

Quantization is a technique that reduces the precision of the weights and/or activations in a neural network. Instead of using floating-point numbers (e.g., 32-bit or 16-bit), quantization uses lower-precision integers (e.g., 8-bit or even binary). This reduction in precision leads to smaller model sizes, faster inference speeds, and lower power consumption.  It can be performed after training (post-training quantization) or during training (quantization-aware training).

*   **How it works:** The continuous range of floating-point values is mapped to a discrete set of integer values. A scaling factor and a zero point are typically used to map between the floating-point and integer representations.

*   **Uses:**
    *   **Reduced Model Size:** Smaller models can be deployed on devices with limited storage, like mobile phones or embedded systems.
    *   **Faster Inference:** Integer arithmetic is generally faster than floating-point arithmetic, leading to faster prediction times.
    *   **Lower Power Consumption:** Reduced memory access and simpler computations reduce power consumption, which is crucial for battery-powered devices.
    *   **Edge Deployment:** Makes it feasible to run complex models on edge devices.

**Model Pruning:**

Pruning is a technique that removes connections (weights) or neurons from a neural network. This reduces the model's complexity, leading to smaller model sizes and potentially faster inference speeds. Pruning can be unstructured (removing individual weights) or structured (removing entire neurons, filters, or layers).

*   **How it works:** Weights with small magnitudes (or less significant based on other criteria like gradient information) are set to zero. After pruning, the model is often fine-tuned to recover accuracy.

*   **Uses:**
    *   **Reduced Model Size:** Fewer weights means a smaller model.
    *   **Faster Inference:** Fewer computations are needed for inference, especially with sparse matrix operations optimized for pruned models.
    *   **Improved Generalization:** By removing less important connections, pruning can sometimes improve the model's ability to generalize to unseen data, preventing overfitting.
    *   **Energy Efficiency:** Similar to quantization, fewer computations also improve energy efficiency.

In summary, both quantization and pruning reduce the complexity and size of neural networks, making them more efficient to deploy and run, particularly on resource-constrained devices. They can be used individually or in combination for maximum impact.

## 2) Application scenario

**Quantization:**

*   **Mobile NLP Applications:** Imagine a mobile app that provides real-time translation. Quantizing the translation model allows it to run efficiently on the phone without draining the battery or requiring a powerful processor.
*   **Embedded Systems:**  Deploying a sentiment analysis model in a smart sensor for monitoring customer feedback requires a lightweight model. Quantization makes this possible.
*   **Edge AI:** A chatbot running on a smart speaker needs to be fast and power-efficient. Quantization ensures low latency and minimal power consumption.

**Pruning:**

*   **Real-time Dialogue Systems:** Pruning large language models used in dialogue systems can reduce the computational cost and latency, improving the responsiveness of the system.
*   **Computer Vision on Drones:** Drones often have limited battery life and processing power. Pruning image recognition models allows them to perform object detection and other vision tasks without quickly running out of power.
*   **Personalized Recommendation Systems:** Pruning can reduce the size of the recommendation models, allowing for more efficient storage and retrieval of user preferences.

**Combined Quantization and Pruning:**

For example, consider training a large language model for a chatbot.  The model can first be pruned to reduce the number of parameters.  Then, the weights of the pruned model can be quantized to further reduce its size and improve inference speed, making it suitable to run efficiently on resource-constrained devices.

## 3) Python method (if possible)

The `torch.quantization` module in PyTorch and the TensorFlow Model Optimization Toolkit provide tools for quantization and pruning.  Here's a simple example using PyTorch for post-training quantization:

```python
import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create an instance of the model
model_fp32 = SimpleModel()

# Load a pre-trained model (or train one)
# model_fp32.load_state_dict(torch.load("model.pth"))
# Set model to eval mode
model_fp32.eval()

# Perform dynamic quantization
model_quantized = quantize_dynamic(
    model_fp32, {nn.Linear}, dtype=torch.qint8
)

# Now model_quantized is quantized and can be used for inference.

# Example of Pruning in Pytorch (requires >=1.7):

import torch.nn.utils.prune as prune
import torch.nn as nn

# Define a simple model
class SimpleModelPrune(nn.Module):
    def __init__(self):
        super(SimpleModelPrune, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create an instance of the model
model = SimpleModelPrune()


# Apply pruning globally to the whole network
parameters_to_prune = (
    (model.fc1, 'weight'),
    (model.fc2, 'weight'),
)

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.5,
)

# Print sparsity of the layers
print(
    "Sparsity in fc1.weight: {:.2f}%".format(
        100. * torch.sum(model.fc1.weight == 0) / model.fc1.weight.nelement()
    )
)
print(
    "Sparsity in fc2.weight: {:.2f}%".format(
        100. * torch.sum(model.fc2.weight == 0) / model.fc2.weight.nelement()
    )
)

# Remove the pruning mask to make it a normal model.
prune.remove(model.fc1, 'weight')
prune.remove(model.fc2, 'weight')

```

**Explanation:**

*   **Quantization:** The `quantize_dynamic` function in PyTorch dynamically quantizes the model during inference. The `dtype=torch.qint8` argument specifies that 8-bit integers should be used for quantization.  Notice that we need to specify which layers we wish to quantize, in this case `nn.Linear`.
*   **Pruning:**  `torch.nn.utils.prune` is used here. We select which layers to prune, and then use `prune.global_unstructured` to apply pruning based on L1 norm. A sparsity level is set and the pruning is applied. `prune.remove` makes the changes permanent by removing the mask that was applied to the weights.

**Important Notes:**

*   These are simplified examples. In practice, quantization and pruning often require careful tuning and fine-tuning to maintain accuracy. Quantization-aware training is often used for better accuracy than post-training quantization.
*   TensorFlow's Model Optimization Toolkit provides similar functionalities.

## 4) Follow-up question

What are some challenges or potential drawbacks of using model quantization and pruning, and how can these be addressed?  Specifically, how does quantization-aware training help, and what are the trade-offs when choosing the pruning granularity (e.g., unstructured vs. structured pruning)?