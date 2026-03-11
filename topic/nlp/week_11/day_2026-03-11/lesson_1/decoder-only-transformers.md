---
title: "Decoder-only Transformers"
date: "2026-03-11"
week: 11
lesson: 1
slug: "decoder-only-transformers"
---

# Topic: Decoder-only Transformers

## 1) Formal definition (what is it, and how can we use it?)

A decoder-only transformer is a type of transformer architecture that exclusively uses the decoder part of the original transformer model as introduced by Vaswani et al. in "Attention is All You Need".  Unlike encoder-decoder transformers (like the original architecture used for machine translation), decoder-only transformers do not have a separate encoder component.  They are primarily designed for *generative tasks* where the goal is to produce a sequence of tokens given a prompt or context.

**Key components and features:**

*   **Self-Attention (Masked):**  The core building block is self-attention, but with a crucial modification: *masking*. The attention mechanism is modified to prevent a token from attending to future tokens in the sequence. This is because during training and inference, the model should only use information from the tokens that come before it to predict the next token. This causality is essential for generative modeling. The masking is typically achieved by setting the attention weights for future tokens to negative infinity (or a large negative number) before applying the softmax function. This effectively prevents these tokens from influencing the attention scores.

*   **Causal Language Modeling:** Decoder-only transformers are particularly well-suited for causal language modeling, where the objective is to predict the next token in a sequence given the preceding tokens. The probability of a sequence `w_1, w_2, ..., w_n` is modeled as the product of conditional probabilities:

    `P(w_1, w_2, ..., w_n) = P(w_1) * P(w_2 | w_1) * P(w_3 | w_1, w_2) * ... * P(w_n | w_1, w_2, ..., w_{n-1})`

*   **Layers:** A decoder-only transformer typically consists of multiple stacked decoder layers. Each layer usually includes:
    *   Masked multi-head self-attention.
    *   Layer normalization (e.g., pre-normalization or post-normalization).
    *   Feed-forward network (typically a two-layer MLP with a GELU or ReLU activation).
    *   Residual connections (skip connections) around each sub-layer.

*   **Usage:** We use decoder-only transformers by first training them on a large corpus of text to learn the statistical patterns and relationships within the language.  Then, during inference, we can provide a prompt (initial sequence of tokens) and the model will generate subsequent tokens one by one, based on the learned probability distribution. The generated tokens are appended to the context, and the process is repeated until a desired length is reached, or a stopping condition is met.

In summary, decoder-only transformers are generative models specifically designed for tasks where the prediction of the next token depends only on the preceding tokens. They are powerful tools for language modeling and text generation.

## 2) Application scenario

Decoder-only transformers have found wide application in several areas:

*   **Text generation:** They can generate realistic and coherent text for various purposes, such as writing articles, stories, poems, or code. Examples include GPT-3, GPT-4, and other large language models.

*   **Code generation:** Models like Codex (built on GPT-3) can generate code from natural language descriptions.

*   **Summarization:** While traditionally encoder-decoder architectures were used for summarization, decoder-only models can also perform summarization, especially when fine-tuned on summarization datasets. The model is prompted with the text to be summarized, and it generates the summary.

*   **Question answering:** They can be used for open-ended question answering, where the answer is generated rather than retrieved from a knowledge base. The question is provided as a prompt, and the model generates the answer.

*   **Dialogue generation:**  They are suitable for building conversational AI agents (chatbots). The model takes the dialogue history as input and generates the next turn in the conversation.

*   **Translation (Zero-Shot or Few-Shot):** Although less common than encoder-decoder models for direct translation, decoder-only transformers can perform translation with few or no explicit examples, leveraging their general language understanding capabilities.

*   **Content Creation:** Creating social media posts, blog articles, or marketing copy.

## 3) Python method (if possible)

Here's a Python example using the `transformers` library by Hugging Face to load and use a pre-trained decoder-only transformer model (specifically, GPT-2):

```python
from transformers import pipeline, set_seed

# Load a pre-trained GPT-2 model
generator = pipeline('text-generation', model='gpt2')

# Set a seed for reproducibility
set_seed(42)

# Generate text from a prompt
prompt = "The quick brown fox"
generated_text = generator(prompt, max_length=50, num_return_sequences=1)

# Print the generated text
print(generated_text[0]['generated_text'])
```

**Explanation:**

1.  **`from transformers import pipeline, set_seed`**: This line imports the necessary functions from the `transformers` library.  `pipeline` is a high-level API for quickly using pre-trained models, and `set_seed` ensures consistent results during generation.
2.  **`generator = pipeline('text-generation', model='gpt2')`**: This creates a text generation pipeline using the pre-trained GPT-2 model.  The `pipeline` function automatically downloads the model and its associated tokenizer if they are not already present.
3.  **`set_seed(42)`**: This sets the random seed to 42 for reproducibility.  Without this, the generated text will be different each time you run the code.
4.  **`prompt = "The quick brown fox"`**: This defines the prompt that will be used to start the text generation.
5.  **`generated_text = generator(prompt, max_length=50, num_return_sequences=1)`**: This is where the text generation actually happens.
    *   `prompt`: The input prompt.
    *   `max_length`: The maximum length of the generated text (including the prompt).
    *   `num_return_sequences`: The number of different text sequences to generate.
6.  **`print(generated_text[0]['generated_text'])`**: This prints the generated text to the console.  The `pipeline` function returns a list of dictionaries, where each dictionary contains the generated text.

This example provides a basic demonstration of using a pre-trained decoder-only transformer for text generation. You can experiment with different prompts, model sizes (e.g., `gpt2-medium`, `gpt2-large`), and generation parameters to achieve different results. To fine-tune your own, you would typically use the `transformers.Trainer` class, providing a dataset formatted correctly for causal language modeling.

## 4) Follow-up question

How can techniques like "temperature" and "top-p sampling" be used to control the diversity and quality of the text generated by a decoder-only transformer? Explain the purpose of each technique.