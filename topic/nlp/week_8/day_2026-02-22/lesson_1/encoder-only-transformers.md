---
title: "Encoder-only Transformers"
date: "2026-02-22"
week: 8
lesson: 1
slug: "encoder-only-transformers"
---

# Topic: Encoder-only Transformers

## 1) Formal definition (what is it, and how can we use it?)

Encoder-only transformers are a type of transformer architecture that solely consists of the encoder portion of the original transformer model (as introduced in the "Attention is All You Need" paper). Unlike sequence-to-sequence models that have both an encoder and a decoder (like for machine translation), or decoder-only models (like GPT for text generation), encoder-only transformers process an input sequence and output a representation for *each* token in the input.

Specifically, the encoder consists of multiple stacked layers. Each layer typically contains two sub-layers:
*   **Multi-Head Self-Attention:** This sub-layer computes attention scores between each token in the input sequence, allowing the model to understand the relationships between different words in the context. This is critical for capturing long-range dependencies within the input.
*   **Feed Forward Network:** This sub-layer applies a feed-forward neural network to each token's representation independently.  This allows the model to learn non-linear relationships in the data.

Residual connections and layer normalization are used around each sub-layer to improve training stability and performance.

How can we use it?

The key strength of encoder-only transformers lies in their ability to create contextualized word embeddings. These embeddings can then be used for various downstream tasks, including:

*   **Text Classification:** Classifying a text document into predefined categories (e.g., sentiment analysis, spam detection). The encoded representation of the entire input sequence (often extracted via a special classification token like `[CLS]` in BERT) is fed into a classification layer.

*   **Named Entity Recognition (NER):** Identifying and classifying named entities in a text (e.g., person names, organizations, locations). The encoded representation of each token is fed into a classification layer to predict the entity type.

*   **Question Answering (extractive):** Given a question and a context passage, identifying the span of text within the context that answers the question. The encoder can be used to encode both the question and the context, and then an additional layer can predict the start and end positions of the answer within the context.

*   **Sentence Similarity:** Determining how similar two sentences are.  The encoded representations of the sentences can be compared using a similarity metric (e.g., cosine similarity).

Examples of models using the encoder-only architecture include BERT, RoBERTa, ALBERT, and ELECTRA.

## 2) Application scenario

Let's consider the scenario of **sentiment analysis** on customer reviews for an e-commerce website.  The goal is to automatically classify each review as either positive, negative, or neutral.

1.  **Input:** A customer review, e.g., "This product is amazing! The delivery was fast and the quality is excellent."

2.  **Encoding:** The review is passed through an encoder-only transformer model (e.g., a pre-trained BERT model fine-tuned on sentiment analysis data). The model outputs a contextualized embedding for each word in the review, as well as an embedding for the `[CLS]` token, which represents the overall sentiment of the review.

3.  **Classification:** The embedding for the `[CLS]` token is fed into a classification layer (e.g., a fully connected layer followed by a softmax activation) that predicts the probability of the review belonging to each sentiment class (positive, negative, neutral).

4.  **Output:** The model predicts the sentiment of the review based on the highest probability score. In this case, it would likely predict "positive."

This application showcases the power of encoder-only transformers in understanding the context of the input text and extracting relevant information for classification tasks. The pre-trained nature of models like BERT allows for effective transfer learning, requiring less training data to achieve high accuracy.

## 3) Python method (if possible)

This example uses the Hugging Face `transformers` library, a popular choice for working with pre-trained transformer models.

```python
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import torch

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased' # You can choose other BERT variants
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3) # 3 labels: positive, negative, neutral

# Input text
text = "This product is amazing! The delivery was fast and the quality is excellent."

# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True) # Pytorch tensors are returned.

# Make prediction
with torch.no_grad():  # Disable gradient calculation during inference
    outputs = model(**inputs)

# Get predicted class probabilities using softmax
probabilities = softmax(outputs.logits, dim=1)

# Get the predicted class
predicted_class = torch.argmax(probabilities, dim=1).item()

# Map class index to sentiment label
label_map = {0: "negative", 1: "neutral", 2: "positive"}
predicted_sentiment = label_map[predicted_class]

# Print the result
print(f"Text: {text}")
print(f"Predicted Sentiment: {predicted_sentiment}")
print(f"Probabilities: {probabilities.tolist()}")

```

Key improvements and explanations:

*   **`torch.no_grad()`:**  Crucially added `torch.no_grad()` to disable gradient calculation during inference. This significantly speeds up the prediction process and reduces memory usage.  It's extremely important for inference.

*   **Padding and Truncation:** The `tokenizer` now includes `padding=True` and `truncation=True`. Padding ensures all sequences have the same length, which is required by the BERT model.  Truncation prevents sequences exceeding the maximum length, which is also important.

*   **`softmax` function:**  Uses `torch.nn.functional.softmax` to get probabilities directly from the logits. This is the correct way to convert the model's raw output into probabilities. The `dim=1` argument is essential to ensure softmax is applied along the correct dimension (across the different classes).

*   **Tensor Handling:** The code now correctly handles PyTorch tensors. `torch.argmax` returns the *index* of the maximum value (the predicted class), and `item()` extracts the integer value from the tensor.

*   **Label Mapping:** A `label_map` is introduced to map the numerical class index (0, 1, 2) to human-readable sentiment labels (negative, neutral, positive).  This makes the output much more understandable.

*   **Complete and Runnable:** The code is now a fully functional example that can be run directly after installing the `transformers` library (`pip install transformers`).

*   **Clearer Comments:** Comments have been added to explain each step in more detail.

This revised example provides a much more accurate and usable demonstration of sentiment analysis with an encoder-only transformer.  It highlights important considerations for working with pre-trained models, such as disabling gradients and handling padding/truncation.  It also emphasizes the correct way to obtain probabilities and interpret the model's output.

## 4) Follow-up question

How do encoder-only transformers handle variable-length input sequences, especially considering that attention mechanisms have a quadratic complexity in terms of sequence length?  Are there specific techniques, such as attention masking or sequence packing, that are commonly used to optimize performance? Also, how does pre-training objective affect the resulting embedding quality?