---
title: "Low-Rank Adaptation (LoRA)"
date: "2026-03-13"
week: 11
lesson: 1
slug: "low-rank-adaptation-lora"
---

# Topic: Low-Rank Adaptation (LoRA)

## 1) Formal definition (what is it, and how can we use it?)

Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning technique for large pre-trained language models (LLMs). Instead of fine-tuning all the parameters of the model (which is computationally expensive and requires a lot of memory), LoRA introduces trainable *low-rank* matrices to specific layers of the pre-trained model, while keeping the original model parameters frozen.

Formally, for a given weight matrix W (e.g., within an attention layer or a fully connected layer), LoRA introduces two low-rank matrices A and B such that:

W' = W + BA

Where:
*   W is the original pre-trained weight matrix (frozen during fine-tuning).
*   B is a low-rank matrix of shape (d, r).
*   A is a low-rank matrix of shape (r, k).
*   d and k are the dimensions of W.
*   r is the rank of the low-rank matrices (r << min(d, k)).
*   W' is the adapted weight matrix used during fine-tuning and inference.

During fine-tuning, only A and B are trained, significantly reducing the number of trainable parameters compared to full fine-tuning. The rank 'r' controls the capacity of the adaptation. A smaller rank reduces the number of trainable parameters but might limit the expressiveness of the adaptation. A larger rank allows for more complex adaptations but increases the number of trainable parameters.

LoRA can be used to adapt pre-trained language models to specific tasks or domains while minimizing computational cost and memory requirements.  It makes LLM fine-tuning accessible on more limited hardware and allows for faster experimentation. The adapted weights (A and B) can be easily swapped, enabling different adaptations for different tasks from the same base model without modifying the base model. This is useful in scenarios where you have several similar tasks and want to avoid creating separate full models for each.

## 2) Application scenario

A typical application scenario for LoRA is adapting a large pre-trained language model (like Llama 2, Mistral, or similar) for a specific task, such as:

*   **Sentiment analysis of customer reviews:** Instead of fine-tuning the entire LLM on a dataset of customer reviews, LoRA allows you to train only a small fraction of the parameters, adapting the model to accurately classify the sentiment expressed in the reviews. This is significantly cheaper than full fine-tuning.
*   **Text summarization for a specific domain (e.g., legal documents):** You can fine-tune an LLM using LoRA on a dataset of legal documents and their corresponding summaries. This allows the LLM to generate more accurate and relevant summaries for legal texts compared to using the original pre-trained model directly.
*   **Code generation for a specific programming language or framework:** LoRA can be used to adapt an LLM to generate code in a specific language (e.g., Rust) or framework (e.g., React).
*   **Generating creative writing in a specific style:** Fine-tuning an LLM with LoRA on a dataset of poems, scripts, or novels can help it learn to generate text with a similar tone, structure, and vocabulary.
*   **Training on privacy-sensitive data:** Since LoRA significantly reduces the number of parameters that need to be stored and transferred, it can be beneficial in settings where data privacy is a concern. The LoRA weights are much smaller than the entire model, reducing the risk of data leakage.

## 3) Python method (if possible)

LoRA is commonly implemented using libraries like `transformers` from Hugging Face and the `peft` (Parameter-Efficient Fine-Tuning) library.  Here's a basic example using `peft`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Load the pre-trained model and tokenizer
model_name = "google/flan-t5-base"  # Replace with your desired model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure LoRA
lora_config = LoraConfig(
    r=8, # Rank of the low-rank matrices
    lora_alpha=32, # Scaling factor for the LoRA weights
    lora_dropout=0.05, # Dropout probability for LoRA layers
    bias="none", # Bias type for LoRA layers
    task_type="CAUSAL_LM", # Specify the task type
    target_modules=["q", "v"] # target the query and value projection matrices
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters() # Prints the number of trainable parameters

# Now you can fine-tune the model as usual using the transformers Trainer

# Example of fine-tuning (requires a training dataset)
# from transformers import Trainer, TrainingArguments
#
# training_args = TrainingArguments(
#     output_dir="./lora-finetuned",
#     per_device_train_batch_size=4,
#     num_train_epochs=3,
#     save_strategy="epoch",
#     logging_steps=100,
# )
#
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=your_training_dataset, # replace with actual dataset
#     data_collator=your_data_collator, # replace with actual data collator
#     tokenizer=tokenizer
# )
#
# trainer.train()
```

**Explanation:**

1.  **Import necessary libraries:** `transformers` for loading the pre-trained model and tokenizer, and `peft` for LoRA implementation.
2.  **Load the pre-trained model and tokenizer:**  Replace `"google/flan-t5-base"` with the actual name of the model you want to use.
3.  **Configure LoRA:**
    *   `r`:  Specifies the rank of the low-rank matrices.  A higher rank increases the number of trainable parameters but may lead to overfitting.
    *   `lora_alpha`:  A scaling factor that controls the magnitude of the LoRA updates. It is multiplied with the BA product. Usually set to `2*r` or `4*r`.
    *   `lora_dropout`:  Applies dropout to the LoRA layers to prevent overfitting.
    *   `bias`: Controls whether a bias term is added to the LoRA layers. `"none"` means no bias.  Other options might be `"all"` or `"lora_only"`.
    *   `task_type`: Specifies the type of task the model will be used for (e.g., "CAUSAL_LM" for causal language modeling, "SEQ_2_SEQ_LM" for sequence-to-sequence language modeling).
    *   `target_modules`: Specifies the modules in the model to which LoRA should be applied. Common choices are the query (`"q"`), value (`"v"`), and key (`"k"`) projection matrices in the attention layers, or the linear layers in the feedforward networks.  This can depend on the model architecture.
4.  **Apply LoRA to the model:**  The `get_peft_model` function from the `peft` library wraps the original model and adds the LoRA layers.
5.  **Fine-tune the model:**  You can then fine-tune the model using the `transformers` `Trainer` class. Replace `your_training_dataset` and `your_data_collator` with appropriate objects for your training data. The commented out section includes a basic example of the training setup but won't work without a prepared dataset.

## 4) Follow-up question

What are some strategies for determining the optimal rank (r) for LoRA in a given task?  Are there any automated methods for finding the best value for 'r', or is it primarily determined through experimentation?