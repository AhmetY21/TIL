---
title: "Pre-training vs Fine-tuning BERT"
date: "2026-03-11"
week: 11
lesson: 3
slug: "pre-training-vs-fine-tuning-bert"
---

# Topic: Pre-training vs Fine-tuning BERT

## 1) Formal definition (what is it, and how can we use it?)

BERT (Bidirectional Encoder Representations from Transformers) is a powerful language model that utilizes a transformer-based architecture to learn contextualized word representations. The training process for BERT, and models like it, involves two distinct stages: **Pre-training** and **Fine-tuning**.

*   **Pre-training:** This is the initial training phase where BERT learns general-purpose language understanding from a large, unlabeled corpus of text (e.g., Wikipedia, books). The model is trained using two primary tasks:

    *   **Masked Language Modeling (MLM):** A certain percentage of the input tokens are randomly masked, and the model is trained to predict these masked tokens based on the surrounding context. This forces BERT to learn bidirectional representations, understanding words based on both preceding and following words.

    *   **Next Sentence Prediction (NSP):** BERT is given pairs of sentences and tasked with predicting whether the second sentence follows the first in the original document. This helps BERT understand relationships between sentences. Although later models have found NSP less crucial and sometimes detrimental, it was a key component of the original BERT model.

    The goal of pre-training is to equip BERT with a broad understanding of language, including vocabulary, grammar, and common sense knowledge. The output of pre-training is a set of model weights that can be used as a starting point for more specific tasks.

*   **Fine-tuning:** After pre-training, BERT is fine-tuned on a specific downstream task (e.g., text classification, question answering, named entity recognition).  This involves taking the pre-trained BERT model and further training it on a labeled dataset relevant to the target task.

    During fine-tuning, the entire pre-trained BERT model, including all layers, is updated using the task-specific data and loss function.  This allows the model to adapt its general-purpose language understanding to the nuances of the specific task. A task-specific layer (e.g., a linear layer for classification) is typically added on top of BERT's output.

    Using pre-trained models allows us to avoid training a large and complex model from scratch which is computationally expensive and data-intensive. We can leverage transfer learning and achieve good performance with less task-specific data.

## 2) Application scenario

Let's consider a **sentiment analysis** scenario.  We want to build a model that can automatically classify customer reviews (e.g., from an e-commerce website) as positive, negative, or neutral.

*   **Pre-training:** BERT has already been pre-trained on a massive dataset of text.  This means it already understands a lot about the English language, including words, grammar, and common expressions of sentiment. We *do not* need to perform pre-training ourselves; we can download the pre-trained weights from platforms like Hugging Face Model Hub.

*   **Fine-tuning:**  We take the pre-trained BERT model and fine-tune it on a dataset of labeled customer reviews.  The dataset consists of reviews labeled with their corresponding sentiment (positive, negative, or neutral). During fine-tuning, BERT learns to map the text of a review to its sentiment label.  A classification layer will be added on top of the BERT embeddings to make the sentiment prediction.  Because BERT is already familiar with language, it requires relatively few labeled examples to achieve high accuracy on this sentiment analysis task.

Other application scenarios include:
*   **Question Answering:** Fine-tuning BERT on datasets such as SQuAD.
*   **Named Entity Recognition (NER):** Fine-tuning BERT on datasets such as CoNLL-2003.
*   **Text Summarization:** Utilizing BERT in an encoder-decoder architecture and fine-tuning it on a summarization dataset.
*   **Natural Language Inference (NLI):** Fine-tuning BERT on datasets such as MNLI.

## 3) Python method (if possible)

Here's a basic example using the `transformers` library from Hugging Face, demonstrating fine-tuning BERT for sentiment analysis:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset

# 1. Load a pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"  # Or any other BERT variant
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3) # Assuming 3 labels (positive, negative, neutral)

# 2. Load a sentiment analysis dataset (e.g., from Hugging Face Datasets)
dataset = load_dataset("imdb", split="train[:1000]+test[:1000]") #Use only 1000 examples from train and test

# 3. Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 4. Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
)

# 5. Create a Trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
    tokenizer=tokenizer,
)

# 6. Fine-tune the model
trainer.train()

# 7. Evaluate the model
results = trainer.evaluate()
print(results)
```

**Explanation:**

1.  **Load Model and Tokenizer:**  We load a pre-trained BERT model and its corresponding tokenizer using `AutoTokenizer` and `AutoModelForSequenceClassification`.  We specify the desired BERT variant (e.g., "bert-base-uncased"). The number of labels must match the number of classes in your classification task.
2.  **Load Dataset:** We load a sentiment analysis dataset from the Hugging Face Datasets library. The `imdb` dataset is used here for simplicity. Limiting the number of examples for demonstration purposes.
3.  **Tokenize Data:** We tokenize the text data using the BERT tokenizer. This converts the text into numerical tokens that BERT can understand. `padding="max_length"` and `truncation=True` ensure all sequences have the same length.
4.  **Define Training Arguments:** We define the training arguments using `TrainingArguments`. This includes the output directory for saving the trained model, the evaluation strategy, the number of training epochs, and the batch size.
5.  **Create Trainer:** We create a `Trainer` object. This object handles the training and evaluation process.  It takes the model, training arguments, training dataset, evaluation dataset, and tokenizer as input.
6.  **Fine-tune Model:**  We fine-tune the model by calling the `trainer.train()` method. This updates the weights of the pre-trained BERT model based on the sentiment analysis dataset.
7.  **Evaluate Model:** The trained model is evaluated on the evaluation dataset.  Metrics like accuracy, precision, recall, and F1-score are calculated.

## 4) Follow-up question

How does the size of the pre-training dataset impact the performance of BERT after fine-tuning on a specific downstream task?  Are there diminishing returns as the pre-training dataset gets larger, or does performance continue to improve indefinitely? What are some of the considerations related to cost and efficiency when deciding on the size of the pre-training dataset?