---
title: "Low-Rank Adaptation (LoRA)"
date: "2026-03-12"
week: 11
lesson: 6
slug: "low-rank-adaptation-lora"
---

# Topic: Low-Rank Adaptation (LoRA)

## 1) Formal definition (what is it, and how can we use it?)

Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning technique designed for large pre-trained language models (LLMs). Instead of fine-tuning *all* the parameters of a large model, LoRA freezes the pre-trained weights and introduces trainable *low-rank* matrices to specific layers of the model.

Mathematically, consider a weight matrix `W` within the LLM, originally of shape `(d, k)`, where `d` and `k` are the dimensions of the matrix.  LoRA introduces two smaller matrices, `A` of shape `(d, r)` and `B` of shape `(r, k)`, where `r << min(d, k)`. `r` is the *rank* of the adaptation. During training, the original weight matrix `W` is kept frozen, and the update is applied as:

`W' = W + BA`

where `W'` is the effective weight matrix used during the forward pass. Only `A` and `B` are trained. The original frozen weights, `W`, remain unchanged.

**How can we use it?**

LoRA allows us to efficiently adapt a pre-trained model to a specific task or dataset. By only training the low-rank matrices, the number of trainable parameters is significantly reduced compared to full fine-tuning. This leads to:

*   **Reduced GPU memory requirements:**  Smaller trainable parameter size allows fine-tuning on less powerful hardware.
*   **Faster training:** Fewer parameters to update results in faster training times.
*   **Easy switching between tasks:** Multiple LoRA modules, each trained for a specific task, can be easily plugged into and out of the same pre-trained model. You just load different `A` and `B` matrices.
*   **Reduced storage requirements:** LoRA weights are much smaller than the full model.

LoRA can be applied to different layers of a Transformer architecture, typically the attention layers (Q, K, V) and the feed-forward network layers. The choice of which layers to adapt and the rank `r` are hyperparameters that can be tuned.

## 2) Application scenario

A common application scenario for LoRA is fine-tuning a large language model for a specific task, such as:

*   **Text summarization:** Adapting a general-purpose LLM to summarize research papers, news articles, or customer reviews.
*   **Question answering:** Fine-tuning an LLM on a dataset of question-answer pairs to improve its ability to answer specific types of questions.
*   **Code generation:** Adapting an LLM to generate code in a specific programming language or for a specific task.
*   **Chatbot development:** Fine-tuning an LLM on a dataset of conversational data to create a chatbot with a specific persona or expertise.
*   **Style Transfer:** Adapting an LLM to mimic the writing style of a particular author.

In these scenarios, LoRA allows you to leverage the knowledge already encoded in the pre-trained LLM while adapting it to the specific requirements of the target task with significantly reduced computational resources. Imagine you have a massive LLM trained on general knowledge, and you want it to answer questions related to medical diagnosis. Training the *entire* model is expensive and time-consuming. With LoRA, you freeze the pre-trained weights and only train a small set of low-rank matrices on medical Q&A data, drastically cutting down on the resource burden.

## 3) Python method (if possible)

The `peft` library (Parameter-Efficient Fine-Tuning) from Hugging Face provides an easy way to implement LoRA.

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

# Load the pre-trained model
model_name_or_path = "google/flan-t5-base"  # Or any other LLM
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)


# Configure LoRA
lora_config = LoraConfig(
    r=8,  # Rank of the adaptation matrices
    lora_alpha=32, #Scaling factor.  Usually r*lora_alpha is roughly a constant.
    lora_dropout=0.05, # Dropout probability for LoRA layers
    bias="none", #Bias type for LoRA layers. Can be "none", "bais" or "all"
    task_type=TaskType.SEQ_2_SEQ_LM, #Sequence-to-Sequence Language Modeling
    target_modules=['q', 'v'] #Specify which layers to apply LoRA to (e.g., attention layers).  You'll need to inspect your model.
)

# Wrap the model with LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# Now you can train the model as usual, but only the LoRA parameters will be updated
# Example using Hugging Face Trainer:
# from transformers import Trainer, TrainingArguments

# training_args = TrainingArguments(
#     output_dir="./lora-trained-model",
#     learning_rate=1e-3,
#     per_device_train_batch_size=32,
#     num_train_epochs=3,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=your_dataset, # Replace with your training dataset
# )

# trainer.train()

# To save the LoRA adapters
# model.save_pretrained("lora-adapter")
```

**Explanation:**

1.  **Import necessary libraries:** `peft` for LoRA and `transformers` for loading pre-trained models.
2.  **Load the pre-trained model:**  Select a pre-trained model from the Hugging Face Model Hub (e.g., `google/flan-t5-base`) and load it using `AutoModelForCausalLM.from_pretrained()`.  Choose a model that is appropriate for your task.
3.  **Configure LoRA:** Create a `LoraConfig` object to specify the LoRA hyperparameters:
    *   `r`: The rank of the low-rank matrices.  A lower rank implies fewer trainable parameters.
    *   `lora_alpha`: A scaling factor for the LoRA weights. It effectively controls the magnitude of the low-rank updates.  Higher values of `lora_alpha` increase the influence of the LoRA adapters.
    *   `lora_dropout`: The dropout probability applied to the LoRA layers. This helps prevent overfitting.
    *   `bias`: Whether or not to train bias terms in the LoRA layers.  The options are `"none"`, `"bais"`, and `"all"`.
    *   `task_type`: Specifies the type of task, which helps the library appropriately configure the model.
    *   `target_modules`: A list of layer names (or modules) to which LoRA will be applied.  You'll need to inspect the model architecture to determine which layers are most suitable for adaptation.  Common choices are attention layers (Q, K, V) or feed-forward layers. You will want to target those layers within your LLM.
4.  **Wrap the model with LoRA:** Use the `get_peft_model()` function to wrap the pre-trained model with the LoRA configuration. This adds the LoRA layers to the specified target modules.
5.  **Train the model:**  Train the wrapped model as you would normally, but only the LoRA parameters will be updated.  The example code shows how to integrate with the Hugging Face `Trainer`.
6.  **Save the LoRA adapters:** Save the trained LoRA adapters using `model.save_pretrained()`.  This saves only the weights of the LoRA layers, resulting in a significantly smaller file size compared to saving the entire model.

## 4) Follow-up question

How does LoRA compare to other parameter-efficient fine-tuning techniques like Adapter layers or Prefix-Tuning in terms of performance, memory usage, and implementation complexity?  What are the trade-offs between these methods, and in what scenarios would one be preferred over the others?