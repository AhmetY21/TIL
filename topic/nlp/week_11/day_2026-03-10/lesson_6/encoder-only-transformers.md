---
title: "Encoder-only Transformers"
date: "2026-03-10"
week: 11
lesson: 6
slug: "encoder-only-transformers"
---

# Topic: Encoder-only Transformers

## 1) Formal definition (what is it, and how can we use it?)

An Encoder-only Transformer is a type of Transformer model architecture that consists only of the encoder part of the original Transformer architecture introduced in the "Attention is All You Need" paper. It processes input sequences and produces contextualized embeddings for each token in the input.

Here's a breakdown:

*   **Encoder Structure:** The encoder stack consists of multiple identical layers. Each layer typically contains two sub-layers:
    *   **Multi-Head Self-Attention:** This allows the model to attend to different parts of the input sequence when encoding a specific token. It captures relationships between different tokens in the input.
    *   **Feed Forward Network:** This is a fully connected feed-forward network applied to each token independently.

*   **Input Embedding and Positional Encoding:** The input sequence is first embedded into a high-dimensional vector space.  Positional encoding is added to the embeddings to provide information about the position of each token in the sequence, as self-attention itself is order-agnostic.

*   **How it works:** The input sequence is fed into the encoder stack. Each layer in the stack processes the output of the previous layer. The self-attention mechanism allows each token to attend to all other tokens in the sequence (including itself) to capture dependencies and contextual information. The feed-forward network further processes the output of the attention mechanism.  The final layer outputs contextualized embeddings for each token in the input sequence.

*   **Usage:** Encoder-only transformers are well-suited for tasks where the entire input sequence is available upfront and the goal is to extract contextual information or representations from it.  They are *not* typically used for sequence generation tasks, where the output is built step-by-step.

Examples of tasks suitable for encoder-only transformers include:

*   **Text Classification:** Classifying text into different categories (e.g., sentiment analysis, topic classification).
*   **Named Entity Recognition (NER):** Identifying and classifying named entities in text (e.g., person, organization, location).
*   **Question Answering (Extractive):** Extracting the answer to a question from a given text passage.
*   **Semantic Similarity:** Determining the degree of similarity between two text sequences.

## 2) Application scenario

Let's consider a **sentiment analysis** application.  Suppose we have a dataset of customer reviews, and we want to classify each review as either positive, negative, or neutral.

An encoder-only transformer (like BERT or RoBERTa) can be used as follows:

1.  **Input:** A customer review (e.g., "This product is amazing! I highly recommend it.").
2.  **Encoding:** The review is tokenized (broken down into individual words or subwords), and each token is embedded into a vector representation. Positional encodings are added. The sequence of embeddings is then fed into the encoder stack.
3.  **Contextualization:** The encoder layers process the input, allowing each token to attend to all other tokens in the review. This creates contextualized embeddings for each token. For example, the word "amazing" will have a different embedding in a positive review than it would in a negative or sarcastic review.
4.  **Classification:** The contextualized embedding corresponding to a special `[CLS]` token (added to the beginning of the input sequence) is typically used as an aggregate representation of the entire review. This embedding is then fed into a classification layer (e.g., a linear layer followed by a softmax activation) to predict the sentiment of the review.  Alternatively, the contextualized embedding of each token can be used for finer-grained sentiment analysis (e.g., identifying the sentiment towards specific aspects of the product).

In this scenario, the encoder-only transformer excels because it can capture the nuanced relationships between words in the review, enabling accurate sentiment classification.

## 3) Python method (if possible)

We can use the `transformers` library from Hugging Face to implement this. Here's a simple example using the BERT model for text classification:

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pre-trained BERT tokenizer and model
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3) # 3 labels: positive, negative, neutral

# Example input text
text = "This product is amazing! I highly recommend it."

# Tokenize the input text
inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt") #padding to make input same size, return pytorch tensors

# Perform inference
with torch.no_grad():
    outputs = model(**inputs) # **inputs unpacks the dictionary into keyword arguments
    predictions = torch.softmax(outputs.logits, dim=1)  # Get probabilities
    predicted_class = torch.argmax(predictions, dim=1) # Get the index of the highest probability

# Print the predicted class
labels = ["negative", "neutral", "positive"]
print(f"Predicted sentiment: {labels[predicted_class]}")
```

**Explanation:**

1.  **Import Libraries:** Imports the necessary classes from the `transformers` library.
2.  **Load Pre-trained Model and Tokenizer:** Loads a pre-trained BERT model and its corresponding tokenizer.  The `BertForSequenceClassification` model is specifically designed for text classification tasks. `num_labels` is set to 3 to match the number of sentiment classes.
3.  **Tokenize Input Text:** Uses the tokenizer to convert the input text into a format that the model can understand. This includes tokenization, padding (to ensure all sequences have the same length), and creating attention masks. The `return_tensors="pt"` argument tells the tokenizer to return PyTorch tensors.
4.  **Perform Inference:**  Passes the tokenized input to the model to generate predictions.  `torch.no_grad()` disables gradient calculation during inference, which reduces memory consumption and speeds up computation. The `**inputs` unpacks the dictionary returned by the tokenizer into keyword arguments that the model expects. The output of the model is the logits (raw, unnormalized scores) for each class. `torch.softmax` converts the logits into probabilities.
5.  **Get Predicted Class:** Uses `torch.argmax` to get the index of the class with the highest probability.
6.  **Print Predicted Sentiment:** Prints the predicted sentiment based on the index.

## 4) Follow-up question

How does fine-tuning an encoder-only transformer for a specific task (like the sentiment analysis example) affect its performance compared to using it directly without fine-tuning? What are the advantages and disadvantages of fine-tuning?