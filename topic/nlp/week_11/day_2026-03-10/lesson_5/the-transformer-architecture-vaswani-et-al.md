---
title: "The Transformer Architecture (Vaswani et al.)"
date: "2026-03-10"
week: 11
lesson: 5
slug: "the-transformer-architecture-vaswani-et-al"
---

# Topic: The Transformer Architecture (Vaswani et al.)

## 1) Formal definition (what is it, and how can we use it?)

The Transformer architecture, introduced in the paper "Attention is All You Need" by Vaswani et al. (2017), is a neural network architecture that relies entirely on attention mechanisms, eschewing recurrent neural networks (RNNs) and convolutional neural networks (CNNs) for sequence-to-sequence tasks like machine translation.  It is based on the encoder-decoder structure, but with key differences:

*   **Attention Mechanism:** At its heart lies the *self-attention* mechanism. Self-attention allows the model to attend to different parts of the input sequence when processing each word.  Instead of processing the sequence sequentially like RNNs, the Transformer processes the entire input in parallel, making it significantly faster to train.  This is achieved by calculating attention weights between all pairs of words in the input sequence. The attention weights indicate how much each word should influence the representation of other words.  Specifically, it uses scaled dot-product attention, where queries, keys, and values are computed and then used to determine the attention weights.

*   **Encoder:** The encoder consists of multiple identical layers. Each layer contains two sub-layers:
    *   **Multi-Head Self-Attention:** Computes multiple attention outputs using different learned linear projections of the input. This allows the model to capture different aspects of the relationships between words.
    *   **Feed-Forward Network:** A position-wise fully connected feed-forward network applied to each position separately and identically.

*   **Decoder:** The decoder also consists of multiple identical layers. Each layer contains three sub-layers:
    *   **Masked Multi-Head Self-Attention:**  Similar to the encoder's self-attention, but prevents the decoder from attending to future tokens in the output sequence during training (masking ensures the model only uses information from previously generated tokens to predict the next token).
    *   **Multi-Head Attention:** This sub-layer allows the decoder to attend to the output of the encoder.  Queries come from the previous decoder layer, and keys and values come from the encoder output.
    *   **Feed-Forward Network:**  Same as the encoder.

*   **Positional Encoding:** Since the Transformer doesn't use recurrence or convolution, it needs a way to understand the order of words in a sequence.  Positional encodings are added to the input embeddings to provide information about the position of each word.  These encodings are deterministic functions of the position, often using sine and cosine functions.

**How to use it:** The Transformer is used for sequence-to-sequence tasks.  The input sequence is fed to the encoder, which produces a contextualized representation of the input. The decoder then takes this representation as input and generates the output sequence one token at a time. During inference (prediction), the decoder's output is fed back as input to generate the next token in an autoregressive fashion.  This process continues until a special end-of-sequence token is generated.

## 2) Application scenario

A key application scenario for the Transformer architecture is **machine translation**. For example, translating English to French.

*   **Input:** An English sentence (e.g., "The cat sat on the mat.") is fed into the encoder.
*   **Encoder:** The encoder processes the sentence and creates a contextualized representation.
*   **Decoder:** The decoder takes the encoder's output and generates the French translation (e.g., "Le chat était assis sur le tapis.").  The decoder starts by generating the first word, then uses that word and the encoder's output to generate the second word, and so on, until it generates the end-of-sentence token.  The attention mechanism allows the decoder to focus on relevant parts of the English sentence when generating each French word. For instance, when translating "cat", the decoder might pay more attention to the corresponding "cat" in the English sentence.

Beyond machine translation, the Transformer architecture has been successfully applied to various other tasks, including:

*   **Text Summarization:** Generating a concise summary of a longer text.
*   **Question Answering:** Answering questions based on a given context.
*   **Text Generation:** Generating realistic and coherent text.
*   **Code Generation:** Generating code from natural language descriptions.
*   **Image Captioning:** Generating descriptions for images (with modifications to handle image features).

## 3) Python method (if possible)

While implementing a full Transformer from scratch is complex, using a library like Hugging Face's Transformers makes it accessible. Here's an example using a pre-trained Transformer model for text generation:

```python
from transformers import pipeline

# Load a pre-trained text generation model
generator = pipeline('text-generation', model='gpt2') # or 'distilgpt2', 'facebook/bart-large' etc.

# Generate text based on a prompt
prompt = "The quick brown fox"
generated_text = generator(prompt, max_length=50, num_return_sequences=1)

# Print the generated text
print(generated_text[0]['generated_text'])
```

**Explanation:**

1.  **`from transformers import pipeline`**: Imports the `pipeline` function from the `transformers` library, which provides a high-level interface for using pre-trained models.
2.  **`generator = pipeline('text-generation', model='gpt2')`**: Creates a text generation pipeline using the `gpt2` pre-trained model. Other models like `distilgpt2` (a smaller, faster version of GPT-2) or `facebook/bart-large` (for summarization or translation) can be used.
3.  **`prompt = "The quick brown fox"`**: Defines the starting text (prompt) for the generation.
4.  **`generated_text = generator(prompt, max_length=50, num_return_sequences=1)`**: Generates text based on the prompt.
    *   `max_length=50` specifies the maximum length of the generated text (including the prompt).
    *   `num_return_sequences=1` specifies that we want to generate only one sequence of text.
5.  **`print(generated_text[0]['generated_text'])`**: Prints the generated text. The output is a list of dictionaries, where each dictionary contains the generated text. We access the text using `generated_text[0]['generated_text']`.

This code uses a simplified interface. For finer-grained control and to implement specific parts of the Transformer architecture, you would work more directly with the model classes and layers provided by the `transformers` library (e.g., `AutoModelForSeq2SeqLM`, `AutoTokenizer`, `T5ForConditionalGeneration`, etc.) and libraries such as PyTorch or TensorFlow.

## 4) Follow-up question

How are positional encodings generated, and why are sine and cosine functions often used for this purpose instead of just simple numerical indices?