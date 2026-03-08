---
title: "Introduction to Contextual Embeddings"
date: "2026-03-08"
week: 10
lesson: 2
slug: "introduction-to-contextual-embeddings"
---

# Topic: Introduction to Contextual Embeddings

## 1) Formal definition (what is it, and how can we use it?)

Contextual embeddings are representations of words where the meaning of each word is dependent on the context in which it appears in a sentence or document.  Unlike traditional word embeddings (like Word2Vec or GloVe) which assign a single, static vector to each word, contextual embeddings generate *different* vector representations for the *same* word based on its specific context.

Formally, let's say we have a sentence *S* = (w<sub>1</sub>, w<sub>2</sub>, ..., w<sub>n</sub>), where w<sub>i</sub> represents the i-th word in the sentence. A contextual embedding model, denoted as *f*, maps each word w<sub>i</sub> to a context-specific vector representation h<sub>i</sub>:

h<sub>i</sub> = *f*(w<sub>i</sub>, S)

where h<sub>i</sub> is the contextual embedding for w<sub>i</sub> in the context of sentence *S*. The function *f* typically represents a complex neural network architecture like a Transformer.

How can we use them?
* **Improved performance in NLP tasks:**  Since contextual embeddings capture the nuances of word meaning, they significantly improve performance in various NLP tasks such as:
    * **Sentiment Analysis:** Accurately determine sentiment even when the same words have different connotations in different contexts (e.g., "This movie is good" vs. "This is good for nothing").
    * **Named Entity Recognition (NER):** Identify entities more reliably by considering the context. For example, distinguishing "Apple" (the company) from "apple" (the fruit).
    * **Question Answering:** Understand the question's intent and the passage context to provide more accurate answers.
    * **Text Classification:** Categorize documents based on their overall meaning and subtle contextual cues.
    * **Machine Translation:**  Generate more fluent and accurate translations by understanding the source language's context.
    * **Text Summarization:** Capture the essence of a document more effectively, considering the relationships between sentences.
* **Fine-tuning Pre-trained Models:** Pre-trained contextual embedding models (like BERT, RoBERTa, and XLNet) can be fine-tuned on specific downstream tasks, leveraging the knowledge already encoded within the embeddings.  This drastically reduces the amount of task-specific data needed for training.
* **Analyzing Semantic Relationships:** Contextual embeddings allow for a more refined analysis of the relationships between words, taking into account their meaning within specific contexts.
## 2) Application scenario

Consider the word "bank". In the sentence "I went to the bank to deposit money," "bank" refers to a financial institution. In the sentence "The river bank was overgrown with weeds," "bank" refers to the side of a river.

Traditional word embeddings would assign the same vector to "bank" in both sentences, failing to capture the different meanings.

A contextual embedding model, like BERT, would generate *different* embeddings for "bank" in these two sentences. The embedding for "bank" in the first sentence would be closer to the embeddings of words like "money", "deposit", and "financial", while the embedding for "bank" in the second sentence would be closer to words like "river", "weeds", and "nature".

This allows a sentiment analysis model to correctly identify that "bank" is a neutral term in the first sentence and also in the second sentence, even though in other contexts, "bank" might have negative connotations (e.g., "The bank foreclosed on my house").

## 3) Python method (if possible)

```python
from transformers import pipeline

# Initialize a fill-mask pipeline using a pre-trained BERT model
fill_mask = pipeline("fill-mask", model="bert-base-uncased")

# Sentence 1: "I went to the bank to deposit money."
sentence1 = "I went to the bank to deposit money."
results1 = fill_mask(sentence1.replace("bank", "[MASK]"))

print(f"Context: {sentence1}")
print("Top predictions:", results1)

# Sentence 2: "The river bank was overgrown with weeds."
sentence2 = "The river bank was overgrown with weeds."
results2 = fill_mask(sentence2.replace("bank", "[MASK]"))

print(f"\nContext: {sentence2}")
print("Top predictions:", results2)

# Illustrative cosine similarity (demonstrates differing embeddings implicitly through differing fill probabilities)
# While this doesn't directly access the embeddings, it demonstrates how the model *infers* context.
# For direct embedding access, one would use the model layers directly.
```

This code uses the `transformers` library by Hugging Face. It initializes a `fill-mask` pipeline with the pre-trained "bert-base-uncased" model.  The `fill-mask` pipeline predicts the missing word in a sentence. While not directly accessing the word embeddings themselves, the different top predictions returned for "bank" in the two sentences *implicitly* demonstrates that BERT generates different contextual representations. The different probabilities associated with different fill words show how the model understands "bank" in different ways depending on context.

To directly access and compare the contextual embeddings themselves, you would need to access the output of specific layers of the BERT model. That process is more involved and requires working directly with the model's architecture and tokenization.

## 4) Follow-up question

How do contextual embeddings handle out-of-vocabulary (OOV) words, especially compared to older methods like Word2Vec? Explain the techniques commonly used to address this issue.