---
title: "Masked Language Modeling (MLM)"
date: "2026-02-22"
week: 8
lesson: 5
slug: "masked-language-modeling-mlm"
---

# Topic: Masked Language Modeling (MLM)

## 1) Formal definition (what is it, and how can we use it?)

Masked Language Modeling (MLM) is a technique used to train language models by randomly masking some of the words in an input sequence and then training the model to predict the masked words. It's a type of *self-supervised learning* because the training data is generated automatically from the input text itself, without requiring human-labeled data.

Formally:

*   **Input:** A sequence of words (a sentence or a longer piece of text).
*   **Masking:** A percentage (typically 15%) of the input words are randomly selected and replaced with a special `[MASK]` token.
*   **Prediction:** The model takes the masked sequence as input and is trained to predict the original, unmasked words that were hidden behind the `[MASK]` tokens.
*   **Objective:** The model learns to understand the contextual relationships between words and to predict words based on their surrounding context. The training objective is usually to minimize the cross-entropy loss between the predicted probability distribution over the vocabulary for each masked token and the actual masked word.

How can we use it? MLM is primarily used as a **pre-training** technique for language models. The resulting pre-trained model can then be fine-tuned for various downstream NLP tasks, such as:

*   **Text Classification:** Classifying text into categories (e.g., sentiment analysis, topic classification).
*   **Question Answering:** Answering questions based on a given context.
*   **Named Entity Recognition (NER):** Identifying and classifying named entities in text (e.g., person names, locations, organizations).
*   **Natural Language Inference (NLI):** Determining the relationship between two sentences (e.g., entailment, contradiction, neutrality).
*   **Text Generation:** Generating new text, potentially conditioned on a prompt.

The pre-training via MLM allows the model to learn a general-purpose representation of language, which can then be adapted to specific tasks with relatively little task-specific training data.

## 2) Application scenario

Consider a scenario where you want to build a system to automatically correct grammatical errors in text. Directly training a system on error correction might require a large dataset of correctly and incorrectly written text, which can be expensive to create.

Instead, you can use MLM as follows:

1.  **Pre-train a language model with MLM:** Train a large language model (like BERT) on a large corpus of general text using MLM.  The model will learn the relationships between words and how to predict missing words based on context.

2.  **Fine-tune for error correction:**  Fine-tune the pre-trained language model on a smaller dataset of correctly and incorrectly written text. During fine-tuning, you would adapt the masked language modeling objective. For example, you might mask the incorrect words and train the model to predict the correct words.  You might also add specific layers or heads that are more suitable for error correction.

The pre-trained model, already having a strong understanding of language, will require less error correction data to achieve good performance compared to training a model from scratch. This is because the MLM pre-training phase has already taught the model a good general understanding of language. The fine-tuning phase then specializes this general knowledge to the specific task of error correction.

## 3) Python method (if possible)

Here's an example using the Hugging Face Transformers library, which provides easy access to pre-trained models and training pipelines for MLM.

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# 1. Load a dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# 2. Load a pre-trained tokenizer and model
model_name = "bert-base-uncased"  # Or any other MLM model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# 3. Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

# 4. Prepare data for MLM (using DataCollatorForLanguageModeling)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15) #mlm_probability is the masking percentage.

# 5. Configure training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,  # Set to True if you want to push your model to Hugging Face Hub
)

# 6. Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,  #Optional validation dataset
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("./my_mlm_model") # Save the trained model
tokenizer.save_pretrained("./my_mlm_model") # Save the tokenizer
```

Explanation:

*   We use the `transformers` library, specifically `AutoTokenizer` and `AutoModelForMaskedLM` to load a pre-trained BERT model and its tokenizer.  You can replace `bert-base-uncased` with other MLM pre-trained models, such as RoBERTa, DistilBERT, etc.
*   We load the `wikitext-2` dataset as a simple example.  You can replace this with your own text data.
*   The `tokenizer` converts text into numerical tokens that the model can understand.
*   The `DataCollatorForLanguageModeling` handles the masking and batching of the data.  It randomly masks a percentage (here, 15%) of the tokens in each input sequence.
*   The `Trainer` class manages the training loop, including optimization, evaluation, and saving the model.
*   The training arguments are configured to define the training process (e.g., learning rate, number of epochs).

## 4) Follow-up question

Given that MLM is effective as a pre-training strategy, how can we choose the optimal masking strategy (e.g., different masking probabilities, masking contiguous spans of text instead of individual tokens, or using dynamic masking where the masking pattern changes during training) and what are the trade-offs associated with each?