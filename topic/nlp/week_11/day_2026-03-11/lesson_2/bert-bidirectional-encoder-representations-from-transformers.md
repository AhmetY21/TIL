---
title: "BERT: Bidirectional Encoder Representations from Transformers"
date: "2026-03-11"
week: 11
lesson: 2
slug: "bert-bidirectional-encoder-representations-from-transformers"
---

# Topic: BERT: Bidirectional Encoder Representations from Transformers

## 1) Formal definition (what is it, and how can we use it?)

BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based neural network architecture designed for natural language processing (NLP). It's pre-trained on a large corpus of text data (Wikipedia and BookCorpus, specifically for the original BERT) using two unsupervised learning tasks: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP). The pre-training allows BERT to learn rich contextual representations of words and sentences.

*   **What it is:** BERT consists of multiple transformer encoder layers stacked on top of each other. The core idea is to learn bidirectional representations by masking some of the input words (MLM) and predicting the masked words based on the surrounding context, both to the left and to the right. Additionally, it is trained to predict whether one sentence follows another (NSP), enabling it to understand relationships between sentences. Different BERT models exist, varying in size and number of parameters (e.g., BERT-Base, BERT-Large).

*   **How we can use it:** BERT is primarily used for fine-tuning on various downstream NLP tasks. Instead of training a model from scratch for each task, we can leverage the pre-trained BERT model as a starting point and fine-tune it with task-specific data. This approach significantly reduces training time and often yields better performance compared to training from scratch. Common applications include:

    *   **Text Classification:** Classifying documents or sentences into predefined categories (e.g., sentiment analysis, spam detection).
    *   **Named Entity Recognition (NER):** Identifying and classifying named entities such as people, organizations, and locations.
    *   **Question Answering:** Extracting answers from a given text based on a question.
    *   **Sentence Pair Tasks:** Determining the relationship between two sentences (e.g., paraphrase detection, textual entailment).

The benefit of using BERT lies in its ability to understand context and semantic relationships within text. This contextual understanding allows BERT to generate powerful and accurate representations that can be effectively used for a wide range of NLP tasks. After fine-tuning, the final layers of BERT provide task-specific predictions.

## 2) Application scenario

Let's consider the application scenario of **Sentiment Analysis of Customer Reviews**. Imagine a company that wants to automatically analyze customer reviews to understand the overall sentiment (positive, negative, or neutral) towards their products or services.

Without BERT, they might use traditional machine learning models like Naive Bayes or Support Vector Machines (SVM) with bag-of-words or TF-IDF features. However, these approaches often struggle to capture the nuances of language and context, leading to inaccurate sentiment predictions. For example, the phrase "not bad" might be misinterpreted as negative by a simple bag-of-words model.

Using BERT, the company can:

1.  **Download a pre-trained BERT model:** They can use a pre-trained BERT-base or BERT-large model from Hugging Face's Transformers library.
2.  **Fine-tune the BERT model:** They would fine-tune the pre-trained BERT model using a dataset of customer reviews labeled with their corresponding sentiment (positive, negative, or neutral). During fine-tuning, the BERT model learns to adapt its internal representations to the specific task of sentiment analysis. A classification layer is typically added on top of the BERT output layer.
3.  **Deploy the fine-tuned BERT model:** After fine-tuning, the company can deploy the model to automatically analyze new customer reviews and classify them based on their sentiment.

BERT's ability to understand context (e.g., "not bad" being a positive statement) leads to significantly more accurate sentiment predictions compared to traditional methods. This allows the company to gain better insights into customer opinions and improve their products or services accordingly.

## 3) Python method (if possible)

```python
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

# Load pre-trained BERT tokenizer and model for sequence classification
model_name = "bert-base-uncased"  # Or "bert-large-uncased", etc.
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3) # Assuming 3 classes: positive, negative, neutral

# Example usage for sentiment analysis using a pipeline
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

text = "This movie was fantastic! The acting was superb, and the plot kept me engaged."
result = classifier(text)
print(f"Sentiment: {result[0]['label']}, Score: {result[0]['score']}")

text = "The food was terrible and the service was slow."
result = classifier(text)
print(f"Sentiment: {result[0]['label']}, Score: {result[0]['score']}")

# Alternatively, you can tokenize and feed the input to the model directly for more control
example_text = "I am so happy today!"
encoded_input = tokenizer(example_text, return_tensors='pt') # pt = pytorch
output = model(**encoded_input)
scores = output.logits
#print(scores)
# In this case you'd need to further process the output scores (e.g. softmax)
# to obtain probabilities for each class, and then select the class with the highest probability.
```

**Explanation:**

1.  **Import necessary libraries:**  `BertTokenizer` for tokenizing the input text, `BertForSequenceClassification` for loading the pre-trained BERT model for sequence classification tasks, and `pipeline` for simplified usage.
2.  **Load pre-trained model and tokenizer:** Specifies the BERT model to use (e.g., "bert-base-uncased") and loads the corresponding tokenizer and model. `num_labels` is set based on the number of sentiment classes (positive, negative, neutral = 3).
3.  **Create a pipeline:** Utilizes the `pipeline` function from the `transformers` library to create a sentiment analysis pipeline using the loaded tokenizer and model. This abstracts away much of the lower-level processing.
4.  **Perform sentiment analysis:** Passes example text to the pipeline and prints the predicted sentiment label and score.
5.  **Direct Input (alternative):** The final section demonstrates how to directly tokenize the input and feed it to the model, giving more control over the process, but requires more manual post-processing of the output logits.

**Note:** This code requires the `transformers` library to be installed: `pip install transformers` and PyTorch installed.  The initial sentiment analysis using the pipeline might not be highly accurate without proper fine-tuning on a task-specific dataset. This example shows the basic usage of a pre-trained (but unfine-tuned) BERT model for sentiment analysis.

## 4) Follow-up question

What are the key differences between BERT and other transformer-based models like GPT (Generative Pre-trained Transformer), and how do these differences affect their suitability for different NLP tasks?