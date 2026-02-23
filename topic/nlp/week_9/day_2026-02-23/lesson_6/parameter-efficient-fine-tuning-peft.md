---
title: "Parameter Efficient Fine-Tuning (PEFT)"
date: "2026-02-23"
week: 9
lesson: 6
slug: "parameter-efficient-fine-tuning-peft"
---

# Topic: Parameter Efficient Fine-Tuning (PEFT)

## 1) Formal definition (what is it, and how can we use it?)

Parameter Efficient Fine-Tuning (PEFT) is a set of techniques used to adapt large pre-trained language models (PLMs) to specific downstream tasks using significantly fewer trainable parameters than full fine-tuning.  Instead of updating all the parameters of the PLM, PEFT methods strategically update a small subset of parameters or introduce new, small modules. This dramatically reduces computational costs, memory footprint, and storage requirements, making it feasible to fine-tune large models even with limited resources.

We can use PEFT to:

*   **Adapt PLMs to new tasks:**  Transfer knowledge from a general-purpose PLM to a specific domain or task (e.g., sentiment analysis, question answering, text summarization) without training millions or billions of parameters.
*   **Improve model performance:**  Fine-tuning, even with PEFT, can improve the performance of PLMs on downstream tasks compared to zero-shot or few-shot learning.
*   **Enable personalized or customized models:** Create specialized models for individual users or specific applications while minimizing the storage and deployment costs associated with full model duplication.
*   **Mitigate catastrophic forgetting:** By freezing most of the pre-trained weights, PEFT helps prevent the model from forgetting the general knowledge it acquired during pre-training, which can occur during full fine-tuning, especially with limited data.

Common PEFT Techniques:

*   **Adapter Modules:** Insert small, task-specific layers (adapters) within the PLM architecture.  Only these adapter layers are trained, while the original PLM weights remain frozen.
*   **Prompt Tuning:**  Instead of modifying the model's parameters, we learn task-specific prompts (input sequences) that guide the PLM to produce the desired output.  This is often applied to the input embedding layer.
*   **Prefix Tuning:** Similar to prompt tuning, but learnable vectors ("prefix") are prepended to each layer of the model, conditioning the model's internal computations without altering its original weights.
*   **Low-Rank Adaptation (LoRA):** Decomposes weight updates into low-rank matrices, significantly reducing the number of trainable parameters. Specifically, instead of directly updating a large weight matrix, a low-rank decomposition is used (e.g., `W = W0 + BA` where `B` and `A` are low-rank matrices).
*   **BitFit:** Only the bias terms in the model are tuned.
*   **IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations):** This technique learns a set of scaling factors applied to the hidden activations of the pre-trained model.
## 2) Application scenario

Consider a company that wants to build a customer service chatbot using a large language model like Llama 2 (70B parameters). Full fine-tuning of such a model would be computationally expensive and require a massive dataset of customer service conversations.  Furthermore, deploying multiple full fine-tuned versions (e.g., one for each product line) would be impractical due to storage and memory constraints.

By using PEFT, the company can:

1.  Choose a PEFT technique like LoRA or adapter modules.
2.  Freeze the majority of the Llama 2's parameters.
3.  Train the chosen PEFT modules (LoRA matrices or adapter layers) using a relatively smaller dataset of customer service conversations.
4.  Deploy the fine-tuned model, which has a significantly smaller memory footprint than the fully fine-tuned version. They can then deploy multiple adapters, one for each product line.

This approach allows the company to leverage the power of a large language model for their specific customer service needs without incurring the prohibitive costs of full fine-tuning and deployment. They can also quickly adapt to new product lines or changes in customer service requirements by training new PEFT modules.
## 3) Python method (if possible)

The `peft` library from Hugging Face simplifies the implementation of various PEFT techniques. Here's an example demonstrating LoRA using `peft` and `transformers`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

# Model and tokenizer
model_name_or_path = "meta-llama/Llama-2-7b-hf" # Replace with your base model
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id  # Add padding token if missing

model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

# LoRA configuration
lora_config = LoraConfig(
    r=8,  # Rank of the update matrices
    lora_alpha=32, #Scaling factor
    lora_dropout=0.05, #Dropout probability for LoRA layers
    bias="none",
    task_type="CAUSAL_LM", #Important for generation tasks
    target_modules=["q_proj", "v_proj"] #Target modules for applying LoRA
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Example usage after training
# Assuming you have a trained model 'model' and tokenizer 'tokenizer'
# Input text
prompt = "Translate English to German: Hello, how are you?"

# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

# Generate text
outputs = model.generate(**inputs, max_length=200)

# Decode and print the generated text
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

#Save the trained adapter:
# model.save_pretrained("lora_adapter")

#To Load the adapter again
#loaded_model = PeftModel.from_pretrained(model, "lora_adapter")

```

Key points:

*   **`LoraConfig`**: Defines the LoRA parameters, like the rank (`r`), the `lora_alpha` scaling factor, the dropout, the bias type and crucially, the `target_modules` to which LoRA is applied. The choice of `target_modules` depends on the model architecture. Common targets in transformer models include the query (`q_proj`), value (`v_proj`), and key (`k_proj`) projection matrices in the attention layers, and the feedforward layers.
*   **`get_peft_model`**:  Wraps the original model with the PEFT configuration, adding the LoRA layers.
*   **`model.print_trainable_parameters()`**:  Prints the number of trainable parameters, which should be significantly smaller than the total number of parameters in the original model.  This confirms that we are indeed only training a small subset of the model's weights. The output shows how many parameters are trainable, which should be a fraction of the total parameters.
*   **`model.save_pretrained` and `PeftModel.from_pretrained`**: Enables saving and loading the PEFT adapter, allowing easy reuse and sharing of the fine-tuned adaptation.  This stores *only* the adapter weights, not the entire model.

The rest of the training process (e.g., preparing the dataset, defining the loss function, and using an optimizer) would follow standard PyTorch/Hugging Face practices. You would train the `model` object returned by `get_peft_model` using your training data.

## 4) Follow-up question

How does the choice of the `target_modules` within the `LoraConfig` affect the performance and efficiency of LoRA, and what guidelines or strategies can be used to select the optimal `target_modules` for a given task and model architecture?