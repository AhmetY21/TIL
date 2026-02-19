---
title: "Introduction to Contextual Embeddings"
date: "2026-02-19"
week: 8
lesson: 3
slug: "introduction-to-contextual-embeddings"
---

# Topic: Introduction to Contextual Embeddings

## 1) Formal definition (what is it, and how can we use it?)

Contextual embeddings are vector representations of words where the meaning of each word is determined by its surrounding context in a sentence. Unlike static word embeddings (e.g., Word2Vec, GloVe), where each word has a single, fixed vector representation regardless of the sentence it appears in, contextual embeddings allow for different representations of the same word based on the specific context in which it is used.

**What is it?**

A contextual embedding model takes a sequence of words (a sentence or document) as input and produces a vector representation for each word in that sequence. These vectors capture not just the intrinsic meaning of the word but also its meaning within the specific context of the input. These models are typically large, pre-trained neural networks (often based on the Transformer architecture).

**How can we use it?**

Contextual embeddings can be used in a wide range of NLP tasks, including:

*   **Text classification:** Using the embeddings as input features for a classifier. This allows the classifier to understand nuanced meanings.
*   **Named entity recognition (NER):** Determining whether a word is a named entity and what type of entity it is, taking context into account.
*   **Question answering:** Understanding the question and the context of the passage to identify the correct answer.
*   **Sentiment analysis:** Determining the sentiment expressed in a text, accounting for the context in which words are used.
*   **Machine translation:** Generating more accurate and fluent translations by understanding the meaning of words in context.
*   **Text summarization:** Creating more concise and informative summaries by capturing the core meaning of the text.
*   **Sentence similarity:** Determining how similar two sentences are in meaning, even if they use different words.

The core idea is to leverage the contextual information captured in the embeddings to improve the performance of downstream NLP tasks.  Instead of directly training models from scratch for each task, we can use pre-trained contextual embedding models and fine-tune them on the specific task's data, which significantly reduces training time and improves performance, especially when labeled data is scarce.

## 2) Application scenario

Consider the word "bank".  In static word embeddings, the word "bank" would have a single vector representation that attempts to capture all its potential meanings. However, the word "bank" can have several different meanings:

*   **Financial institution:** "I deposited the check at the bank."
*   **River bank:** "We sat on the bank of the river."

Using static word embeddings, a model would struggle to distinguish between these two usages. However, with contextual embeddings, the model can analyze the surrounding words and understand the intended meaning. In the first sentence, words like "deposited" and "check" provide context that indicates "bank" refers to a financial institution. In the second sentence, words like "river" and "sat" provide context that indicates "bank" refers to the edge of a river.

Therefore, a contextual embedding model would generate different vector representations for "bank" in each of these sentences, reflecting the distinct meanings it has within each context.  This allows downstream tasks, such as sentiment analysis or part-of-speech tagging, to be performed with greater accuracy. For example, a sentiment analysis task would not incorrectly associate a negative sentiment with a sentence about a river bank simply because the word "bank" is sometimes used in negative contexts related to financial crises.

## 3) Python method (if possible)

We can use the `transformers` library in Python to work with contextual embeddings.  Here's an example using the BERT model:

```python
from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Input text
text = "I deposited the check at the bank."

# Tokenize the text
marked_text = "[CLS] " + text + " [SEP]"
tokenized_text = tokenizer.tokenize(marked_text)

# Map tokens to IDs
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])

# Get the model's output
with torch.no_grad():
    outputs = model(tokens_tensor)
    hidden_states = outputs[0]  # The last hidden-state is the contextual embedding. The shape is [batch_size, sequence_length, hidden_size].

# Get the embeddings for the word "bank"
bank_index = tokenized_text.index("bank")
bank_embedding = hidden_states[0, bank_index, :] # Shape: (hidden_size,) (e.g., 768 for bert-base-uncased)

print(f"Shape of bank embedding: {bank_embedding.shape}")
# Example: Shape of bank embedding: torch.Size([768])

# Now, let's do it with the second sentence
text2 = "We sat on the bank of the river."
marked_text2 = "[CLS] " + text2 + " [SEP]"
tokenized_text2 = tokenizer.tokenize(marked_text2)
indexed_tokens2 = tokenizer.convert_tokens_to_ids(tokenized_text2)
tokens_tensor2 = torch.tensor([indexed_tokens2])

with torch.no_grad():
    outputs2 = model(tokens_tensor2)
    hidden_states2 = outputs2[0]

bank_index2 = tokenized_text2.index("bank")
bank_embedding2 = hidden_states2[0, bank_index2, :]

# Compare the embeddings (e.g., using cosine similarity)
from torch.nn.functional import cosine_similarity
similarity = cosine_similarity(bank_embedding, bank_embedding2, dim=0)
print(f"Cosine similarity between the two 'bank' embeddings: {similarity.item()}")
# Example: Cosine similarity between the two 'bank' embeddings: 0.7769...  The embeddings are different, but still somewhat similar.
```

**Explanation:**

1.  **Load pre-trained model and tokenizer:** We load the BERT model and tokenizer.  The tokenizer is crucial for converting text into tokens that the model can understand.
2.  **Tokenize the text:** We tokenize the input text and add special tokens "[CLS]" and "[SEP]" that BERT uses.
3.  **Map tokens to IDs:**  The tokens are converted into numerical IDs, which are the actual input to the BERT model.
4.  **Convert to PyTorch tensors:**  The numerical IDs are converted into PyTorch tensors.
5.  **Get the model's output:**  We pass the tensor to the BERT model and obtain the hidden states, which contain the contextual embeddings. `torch.no_grad()` is used to disable gradient calculation during the forward pass, saving memory and computation.
6.  **Get the embedding for "bank":** We locate the index of the token "bank" and extract its corresponding embedding from the hidden states.
7.  **Compare the embeddings:** We repeat the process for the second sentence and then compute the cosine similarity between the two "bank" embeddings to demonstrate that they are different, reflecting the different contexts.

## 4) Follow-up question

What are some techniques to fine-tune contextual embedding models for specific downstream tasks, and what factors influence the choice of the fine-tuning approach? For example, should you fine-tune the entire model, or only specific layers? How does the size of your dataset affect your fine-tuning strategy?