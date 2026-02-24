---
title: "Low-Rank Adaptation (LoRA)"
date: "2026-02-24"
week: 9
lesson: 1
slug: "low-rank-adaptation-lora"
---

# Topic: Low-Rank Adaptation (LoRA)

## 1) Formal definition (what is it, and how can we use it?)

Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning technique for large pre-trained language models (PLMs).  Instead of updating all the parameters of the PLM, which can be computationally expensive and require significant memory, LoRA freezes the pre-trained weights and injects trainable rank-decomposition matrices into each layer of the Transformer architecture. Specifically, for a given weight matrix *W* in the original model, LoRA adds a decomposed matrix *BA*, where *B* has shape (d, r) and *A* has shape (r, k). Here, *W* has shape (d, k) and *r << min(d, k)*, making *r* the *rank* of the decomposition.

The update to the original weight *W* becomes *W + BA*. During training, only *A* and *B* are updated, while *W* remains frozen.  The output of the layer is thus computed as:

*h = Wx + BAx*

Where *x* is the input to the layer and *h* is the output.

We can use LoRA to adapt a pre-trained model to a specific downstream task with significantly fewer trainable parameters compared to full fine-tuning. This drastically reduces GPU memory usage, making it feasible to fine-tune very large models on resource-constrained environments.  The key idea is that the weight updates during adaptation have a low intrinsic rank.

## 2) Application scenario

LoRA is particularly well-suited for fine-tuning large language models (LLMs) for various downstream tasks, including:

*   **Text classification:** Adapting a pre-trained language model like BERT or RoBERTa for sentiment analysis, topic classification, or spam detection.
*   **Text generation:** Fine-tuning a generative model like GPT-3 or T5 for tasks like text summarization, machine translation, or dialogue generation.
*   **Question answering:** Adapting a language model to answer questions based on a given context.
*   **Code generation:** Fine-tuning code-generating models for specific programming languages or frameworks.

In resource-constrained environments like edge devices or cloud platforms with limited GPU memory, LoRA allows users to efficiently adapt LLMs for specific tasks without the need for extensive hardware upgrades or expensive cloud resources. It is also useful for rapidly experimenting with different task-specific adaptations since the training time and storage requirements are significantly reduced. LoRA also facilitates better multitask learning.

## 3) Python method (if possible)
```python
# Example using the `peft` library from Hugging Face Transformers

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# 1. Load pre-trained model and tokenizer
model_name = "facebook/opt-350m" # Replace with your desired model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. Configure LoRA
lora_config = LoraConfig(
    r=8,                       # Rank of the decomposition matrices
    lora_alpha=32,             # Scaling factor
    lora_dropout=0.05,         # Dropout probability
    bias="none",               # Bias type: "none", "bais", or "all"
    task_type=TaskType.CAUSAL_LM # Task type (e.g., CAUSAL_LM for text generation)
    #target_modules=["q_proj", "v_proj"] # Optional: Specify specific layers to adapt
)

# 3. Wrap the model with LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Check trainable parameter count

# At this point, you can train the 'model' as you normally would,
# using a training loop, Trainer API, etc.  Only the LoRA parameters
# will be updated during training.

# Example training loop (minimal example - replace with your actual training logic)
import torch
from torch.optim import AdamW

#Dummy data for example only
train_dataset = ["This is a test sentence.", "Another test sentence."]
optimizer = AdamW(model.parameters(), lr=1e-4)

model.train()
for text in train_dataset:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(model.device)
    labels = inputs["input_ids"].clone()
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# After training:
# 4. Save the LoRA weights (only A and B matrices)
# model.save_pretrained("my_lora_model")

# 5. Later, to load the LoRA model:
# from peft import PeftModel
# model = AutoModelForCausalLM.from_pretrained(model_name)
# model = PeftModel.from_pretrained(model, "my_lora_model")

```

## 4) Follow-up question

How does the choice of rank *r* impact the performance and efficiency of LoRA, and what strategies can be used to determine an optimal value for *r*?