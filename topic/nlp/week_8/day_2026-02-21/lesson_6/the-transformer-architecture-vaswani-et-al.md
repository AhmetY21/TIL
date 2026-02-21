---
title: "The Transformer Architecture (Vaswani et al.)"
date: "2026-02-21"
week: 8
lesson: 6
slug: "the-transformer-architecture-vaswani-et-al"
---

# Topic: The Transformer Architecture (Vaswani et al.)

## 1) Formal definition (what is it, and how can we use it?)

The Transformer architecture, introduced in the paper "Attention is All You Need" by Vaswani et al. (2017), is a novel neural network architecture based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. It's primarily used for sequence-to-sequence tasks like machine translation, text summarization, and text generation.

**Key Components:**

*   **Encoder:** Processes the input sequence and transforms it into a continuous representation. It consists of N identical layers. Each layer comprises two sub-layers:
    *   **Multi-Head Self-Attention:** Computes attention weights between different parts of the input sequence, capturing relationships within the input.
    *   **Feed Forward Network:** A position-wise fully connected feed-forward network applied to each position separately and identically.

*   **Decoder:** Generates the output sequence based on the encoder's output. It also consists of N identical layers. Each layer comprises three sub-layers:
    *   **Masked Multi-Head Self-Attention:** Similar to the encoder's attention, but prevents the decoder from attending to future tokens in the sequence during training.
    *   **Multi-Head Attention:** Attends to the output of the encoder, allowing the decoder to consider the entire input sequence when generating the output.
    *   **Feed Forward Network:** A position-wise fully connected feed-forward network.

*   **Attention Mechanism (Scaled Dot-Product Attention):** The core of the transformer.  It calculates attention weights based on three inputs:
    *   **Query (Q):** Represents the request for information.
    *   **Key (K):** Represents the data that the query is attending to.
    *   **Value (V):** The actual information being extracted.

    The attention weights are calculated as `Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V`, where `d_k` is the dimensionality of the keys. The scaling by `sqrt(d_k)` prevents the dot products from becoming too large, which can lead to small gradients after the softmax.

*   **Multi-Head Attention:** Allows the model to attend to information from different representation subspaces at different positions. It runs the attention mechanism multiple times in parallel with different learned linear projections of Q, K, and V, then concatenates the results.

*   **Positional Encoding:** Since the transformer lacks recurrence, it needs a mechanism to inject information about the position of tokens in the sequence. Positional encodings are added to the input embeddings. These can be learned or fixed functions (e.g., sine and cosine functions).

**How we can use it:**

1.  **Sequence-to-Sequence Tasks:** Machine translation, text summarization, question answering, code generation, etc.  The input sequence is fed to the encoder, and the decoder generates the output sequence.
2.  **Classification:** By feeding the input sequence through the encoder and using the final hidden state (or CLS token in some variations) for classification.
3.  **Pre-training and Fine-tuning:** Pre-training a large Transformer model on a massive dataset and then fine-tuning it for specific downstream tasks (e.g., BERT, GPT).
4.  **Feature Extraction:** Using the hidden states from different layers of the encoder or decoder as features for other models.

## 2) Application scenario

**Scenario: Machine Translation (English to French)**

Let's say we want to translate the English sentence "Hello, how are you?" into French.

1.  **Input:** The English sentence is tokenized and converted into numerical representations (word embeddings). Positional encodings are added to these embeddings.

2.  **Encoder:** The encoder processes the English sentence through multiple layers of self-attention and feed-forward networks. The self-attention mechanism allows the model to understand the relationships between words in the sentence. For example, it might learn that "how" is related to "are" and "you".

3.  **Decoder:** The decoder receives the encoder's output and generates the French translation one word at a time.
    *   The decoder's masked self-attention prevents it from "cheating" by looking at future words in the translation.
    *   The decoder's attention mechanism attends to the encoder's output, allowing it to focus on the relevant parts of the English sentence when generating each French word. For example, when generating the French word for "hello" ("Bonjour"), it will focus on the "Hello" part of the English sentence.

4.  **Output:** The decoder outputs the French translation "Bonjour, comment allez-vous?".

The Transformer excels in this scenario because it can capture long-range dependencies in the sentence, which is crucial for accurate translation. The attention mechanism allows the model to directly relate words that are far apart in the sequence.

## 3) Python method (if possible)

While implementing a full Transformer from scratch is complex, we can use existing libraries like Hugging Face Transformers to easily utilize pre-trained models.  Here's a simplified example using the `transformers` library to perform English to French translation using a pre-trained model from the Hugging Face model hub.

```python
from transformers import pipeline

# Load a pre-trained translation model (English to French)
translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")

# Input English sentence
english_sentence = "Hello, how are you?"

# Translate the sentence
french_translation = translator(english_sentence)

# Print the translation
print(french_translation)
```

**Explanation:**

1.  **Import `pipeline`:**  This utility from `transformers` simplifies using models for common tasks.
2.  **Load the pre-trained model:**  `pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")` loads a pre-trained English-to-French translation model. "Helsinki-NLP/opus-mt-en-fr" is the name of the model on the Hugging Face Model Hub. The `pipeline` function downloads the model and tokenizer if it's not already cached.
3.  **Translate the sentence:** `translator(english_sentence)` performs the translation using the loaded model.
4.  **Print the output:** The output will be a list containing a dictionary with the translated text. The dictionary key is `translation_text`.

This example demonstrates how easily we can leverage the power of the Transformer architecture using the `transformers` library without needing to implement the complex attention mechanisms ourselves.

## 4) Follow-up question

How does the Transformer architecture handle variable-length input sequences and output sequences, especially since the attention mechanism relies on matrix operations with fixed-size tensors?  Are there specific techniques or padding strategies employed to address this? Explain the role of padding and masking in achieving variable sequence lengths in Transformers.