---
title: "BERT: Bidirectional Encoder Representations from Transformers"
date: "2026-02-22"
week: 8
lesson: 3
slug: "bert-bidirectional-encoder-representations-from-transformers"
---

# Topic: BERT: Bidirectional Encoder Representations from Transformers

## 1) Formal definition (what is it, and how can we use it?)

BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained language model that uses the Transformer architecture to learn contextual representations of words in text. Unlike earlier models that processed text unidirectionally (either left-to-right or right-to-left), BERT considers the entire context of a word simultaneously (bidirectionally). This bidirectional context awareness allows BERT to understand the nuances of language and produce more accurate and nuanced representations.

BERT is typically pre-trained on a massive corpus of text data using two main objectives:

*   **Masked Language Modeling (MLM):** A percentage (typically 15%) of the words in the input sequence are randomly masked. The model's task is to predict the masked words based on the surrounding context.  This forces the model to learn bidirectional context.

*   **Next Sentence Prediction (NSP):**  The model is given pairs of sentences and asked to predict whether the second sentence logically follows the first. This helps the model understand relationships between sentences.  (Note: Later research has shown NSP isn't always necessary and sometimes hinders performance).

After pre-training, BERT can be fine-tuned for a variety of downstream NLP tasks, including:

*   **Text Classification:**  Classifying text into different categories (e.g., sentiment analysis, spam detection).
*   **Question Answering:** Answering questions based on a given context.
*   **Named Entity Recognition (NER):** Identifying and classifying named entities in text (e.g., people, organizations, locations).
*   **Sentence Similarity:** Determining the semantic similarity between two sentences.
*   **Text Summarization:**  Generating a concise summary of a longer text.

We use BERT by first obtaining a pre-trained BERT model (e.g., from Hugging Face's Transformers library). Then, we fine-tune the model on a task-specific dataset. The fine-tuning process involves updating the model's parameters to optimize its performance on the target task. After fine-tuning, the model can be used to make predictions on new, unseen data.

## 2) Application scenario

Consider a customer service chatbot that needs to understand customer inquiries and route them to the appropriate department. Traditional keyword-based approaches might fail to recognize the underlying intent if the customer uses different phrasing or synonyms.

BERT can be used to perform **intent classification** in this scenario. The customer's inquiry is fed into a fine-tuned BERT model, which classifies the intent into categories like "billing issue," "technical support," "order status," etc. By using the contextual understanding provided by BERT, the chatbot can more accurately identify the customer's intent, even with variations in language. This leads to faster and more accurate routing, improving customer satisfaction.

For example, a customer might say:

*   "I'm having trouble understanding my bill."
*   "Why is my invoice so high this month?"
*   "My monthly charge doesn't seem right."

All of these sentences have the same underlying intent: "billing issue."  A keyword-based system might miss the connection between these phrasings, but a BERT model fine-tuned for intent classification can recognize the similar meaning despite the different wording.

## 3) Python method (if possible)

Here's a simple example using the Hugging Face `transformers` library to classify a single sentence using a pre-trained BERT model:

```python
from transformers import pipeline

# Load a pre-trained BERT model for text classification
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Input text
text = "This movie was incredibly disappointing."

# Classify the text
result = classifier(text)

# Print the results
print(result)
# Expected output (something similar):
# [{'label': 'NEGATIVE', 'score': 0.999...}]

#Another Example with a positive Sentence
text = "This movie was incredibly awesome!"
result = classifier(text)
print(result)

#Expected output (something similar):
# [{'label': 'POSITIVE', 'score': 0.999...}]
```

**Explanation:**

1.  **`from transformers import pipeline`**: Imports the `pipeline` function from the `transformers` library.  Pipelines are a simple way to use pre-trained models for inference.
2.  **`classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")`**: Creates a text classification pipeline using a pre-trained DistilBERT model (a smaller, faster version of BERT) that has been fine-tuned for sentiment analysis on the SST-2 dataset.
3.  **`text = "..."`**: Defines the input text.
4.  **`result = classifier(text)`**:  Passes the text to the classifier pipeline, which performs the classification using the pre-trained model.
5.  **`print(result)`**: Prints the classification result, which includes the predicted label (e.g., "NEGATIVE", "POSITIVE") and the corresponding confidence score.

This is a simplified example.  For real-world applications, you'd typically fine-tune BERT on your own dataset to achieve better performance on your specific task.  The Hugging Face library provides extensive tools for fine-tuning BERT and other transformer models.  You would typically use `AutoModelForSequenceClassification` or similar classes for fine-tuning.

## 4) Follow-up question

How does BERT handle out-of-vocabulary (OOV) words, and what are some techniques to mitigate the issues caused by OOV words when using BERT in practical applications?