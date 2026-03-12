---
title: "Parameter Efficient Fine-Tuning (PEFT)"
date: "2026-03-12"
week: 11
lesson: 5
slug: "parameter-efficient-fine-tuning-peft"
---

# Topic: Parameter Efficient Fine-Tuning (PEFT)

## 1) Formal definition (what is it, and how can we use it?)

Parameter Efficient Fine-Tuning (PEFT) is a collection of techniques designed to fine-tune large pre-trained language models (PLMs) using a *fraction* of the parameters that would be updated in full fine-tuning. Instead of updating all (or most) of the parameters of the PLM, PEFT methods typically introduce a small number of trainable parameters while keeping the original PLM parameters frozen. This offers several advantages:

*   **Reduced computational cost and memory requirements:** Updating fewer parameters significantly reduces the computational burden during training and lowers the memory footprint.  This makes fine-tuning feasible on resource-constrained devices or with very large models.
*   **Faster training and inference:** Fewer parameters translate to faster training convergence and potentially faster inference, especially when compared to full fine-tuning.
*   **Lower storage requirements:** Storing and deploying a model with significantly fewer updated parameters is much more efficient.
*   **Mitigated catastrophic forgetting:** Freezing the pre-trained weights helps retain the original knowledge encoded in the PLM, reducing the risk of catastrophic forgetting on previously learned tasks.
*   **Improved parameter sharing and modularity:** PEFT methods often result in modular components that can be easily swapped or combined with different PLMs or tasks.

We can use PEFT techniques to adapt a pre-trained model to a specific downstream task, such as text classification, question answering, or text generation, without having to retrain the entire model from scratch. This allows us to leverage the knowledge already learned by the PLM while specializing it for a new application.

Common PEFT techniques include:

*   **Adapters:** Adding small, trainable neural network modules (adapters) to the existing PLM architecture.  These modules are typically inserted after the attention or feedforward layers.
*   **Prefix-tuning:** Adding trainable task-specific prefix vectors to the input of each transformer layer.
*   **Prompt tuning:** Optimizing a soft prompt (continuous vectors) prepended to the input sequence. The PLM parameters remain frozen.
*   **Low-Rank Adaptation (LoRA):**  Decomposing the weight matrices of existing layers into low-rank matrices and only training these low-rank matrices.
*   **BitFit:** Only training the bias terms in the pre-trained model, freezing all other weights.
*   **IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations):**  Scaling the activations of internal layers using learned vectors.

## 2) Application scenario

Imagine you have a very large language model, like a version of GPT, trained on a massive dataset of text and code. You want to use this model for sentiment analysis of customer reviews for a specific product category (e.g., analyzing reviews of smartphones to determine if they are positive, negative, or neutral).

Performing full fine-tuning on such a large model for this relatively specific task would be computationally expensive and require a lot of resources. Furthermore, if you later wanted to apply the model to sentiment analysis of *another* product category (e.g., analyzing reviews of laptops), you would need to fine-tune the entire model again.

PEFT techniques, such as LoRA or adapter tuning, would be ideal in this scenario:

1.  **Efficiency:** You can fine-tune the model for sentiment analysis of smartphone reviews by only updating a small fraction of the parameters.
2.  **Resource savings:**  The reduced memory requirements and computational cost make it feasible to fine-tune the model even on machines with limited resources.
3.  **Task-Specific Adapters:** You can create separate adapters for different product categories (smartphones, laptops, etc.). These adapters can be easily swapped or combined to tailor the model to specific needs.
4.  **Knowledge Retention:** The pre-trained knowledge of the base language model remains largely intact, allowing it to still perform well on other tasks.

This approach allows for efficient adaptation of the model to multiple specific tasks without retraining the entire model each time.

## 3) Python method (if possible)

The `transformers` library from Hugging Face, combined with libraries like `peft`, provides excellent support for implementing PEFT techniques.  Here's an example using LoRA:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import torch
from datasets import load_dataset

# 1. Load the dataset
dataset = load_dataset("imdb", split="train[:1000]") # limit the size for demonstration
dataset = dataset.map(lambda examples: {'label': examples['label'], 'text': examples['text']})


# 2. Load the tokenizer and model
model_name_or_path = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=2)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 3. Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,  # Rank of the low-rank matrices
    lora_alpha=32, # scaling factor.
    lora_dropout=0.05,
    bias="none",
    target_modules=["query", "value"] #Specify which layers to apply LoRA to
)

# 4. Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# 5. Define training arguments
training_args = TrainingArguments(
    output_dir="lora_sentiment_analysis",
    learning_rate=1e-3,
    per_device_train_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)


# 6. Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
    tokenizer=tokenizer,
    compute_metrics=lambda p: {"accuracy": (p.predictions.argmax(-1) == p.label_ids).mean()}
)


trainer.train()
```

Key elements of this code:

*   **LoraConfig:** Configures the LoRA parameters.  `r` controls the rank of the low-rank matrices, which determines the number of trainable parameters introduced. `target_modules` specifies the modules to apply LoRA.
*   **get_peft_model:**  Applies the LoRA configuration to the base model, wrapping it with the PEFT adapters.
*   **TrainingArguments:** Sets up the training process.  Important parameters include the learning rate and number of epochs.
*   **Trainer:** Facilitates the training process using the `transformers` library.  It handles the training loop, evaluation, and saving of the model.

This is a simplified example.  More complex scenarios might involve custom data preprocessing, hyperparameter tuning, and more sophisticated evaluation metrics. Libraries like `bitsandbytes` can further optimize memory usage during training, especially when using 4-bit or 8-bit quantization.

## 4) Follow-up question

How does the choice of `target_modules` in LoRA configuration impact the performance and efficiency of the fine-tuning process, and what are some guidelines for selecting these modules for different types of tasks (e.g., text generation vs. text classification)?