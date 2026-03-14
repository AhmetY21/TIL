---
title: "Question Answering Systems (Extractive)"
date: "2026-03-14"
week: 11
lesson: 3
slug: "question-answering-systems-extractive"
---

# Topic: Question Answering Systems (Extractive)

## 1) Formal definition (what is it, and how can we use it?)

Extractive Question Answering (QA) systems are a type of question answering system that answers a question by extracting a segment (span) of text directly from a given context (a document or a set of documents). In other words, instead of generating a new answer or retrieving a pre-defined answer, the system identifies the portion of the context that best answers the posed question.

**Formally:**

Given a context document *D* and a question *Q*, an extractive QA system aims to find the start and end positions (*s*, *e*) within *D* such that the span *D[s:e]* (the substring of *D* from position *s* to position *e*) contains the answer to *Q*.

**How can we use it?**

Extractive QA systems can be used in various applications where large amounts of textual data need to be searched for specific information. They are particularly useful when a precise and concise answer is expected, and the answer is explicitly stated within the context. We can use them to:

*   Quickly find specific facts within documents.
*   Build chatbots that can answer questions based on a knowledge base.
*   Create search engines that highlight the relevant answer within search results.
*   Automate the process of answering common questions in customer support.

## 2) Application scenario

Imagine a medical researcher needs to find information about the side effects of a specific drug. They have access to a large database of medical research papers and clinical trial reports.

Instead of manually reading through hundreds of pages of documents, they can use an extractive QA system. The researcher can ask the system: "What are the side effects of Drug X?"

The system would then analyze the database, identify relevant passages related to Drug X, and extract the text span that explicitly lists the side effects. For instance, the system might extract: "Common side effects of Drug X include nausea, headache, and dizziness."

This allows the researcher to quickly and efficiently find the specific information they need without having to spend hours reading through irrelevant material. The system provides the *exact text* from the document, providing a high degree of confidence in the accuracy of the answer.

## 3) Python method (if possible)

One popular library for implementing extractive QA systems in Python is Hugging Face's Transformers library. This library provides pre-trained models and tools for various NLP tasks, including question answering. We can use the `pipeline` function to easily load and use a pre-trained QA model.

```python
from transformers import pipeline

# Load a pre-trained question answering model
qa_pipeline = pipeline("question-answering")

# Define the question and context
question = "What is the capital of France?"
context = "France is a country located in Western Europe. Its capital is Paris."

# Get the answer
answer = qa_pipeline(question=question, context=context)

# Print the answer
print(f"Answer: {answer['answer']}")
print(f"Score: {answer['score']}")
print(f"Start: {answer['start']}")
print(f"End: {answer['end']}")
```

This code snippet demonstrates how to use a pre-trained model to extract the answer to a question from a given context.  The `answer` dictionary contains the extracted answer text (`answer['answer']`), a confidence score (`answer['score']`), and the start and end indices of the answer within the context (`answer['start']` and `answer['end']`).  You can experiment with different questions, contexts, and pre-trained models to explore the capabilities of extractive QA systems.

## 4) Follow-up question

What are some limitations of extractive question answering systems, and how do researchers attempt to overcome them?