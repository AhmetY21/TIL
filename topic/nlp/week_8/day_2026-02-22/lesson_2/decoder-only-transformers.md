---
title: "Decoder-only Transformers"
date: "2026-02-22"
week: 8
lesson: 2
slug: "decoder-only-transformers"
---

# Topic: Decoder-only Transformers

## 1) Formal definition (what is it, and how can we use it?)

Decoder-only Transformers are a type of Transformer architecture that solely utilizes the "decoder" block, omitting the "encoder" component.  In contrast to the original Transformer architecture (e.g., used in machine translation), which consists of both encoder and decoder stacks, decoder-only models process input sequences and generate output sequences auto-regressively.

**Components and Functionality:**

*   **Input Embedding:** The input sequence is first converted into embeddings, similar to the standard Transformer.
*   **Decoder Layers:** The core of the architecture is a stack of identical decoder layers. Each layer consists of:
    *   **Masked Multi-Head Self-Attention:** This is a crucial component.  It allows each token in the input sequence to attend to all *preceding* tokens, but prevents it from attending to future tokens during training.  This masking is essential for auto-regressive generation. The attention mechanism calculates attention scores between tokens and uses these scores to weight the token representations.
    *   **Feed-Forward Network:** A fully connected feed-forward network applied to each token individually, providing non-linearity and further processing.
    *   **Layer Normalization and Residual Connections:** Standard techniques to stabilize training and improve performance.

*   **Output Projection and Softmax:**  The output of the final decoder layer is projected to a vocabulary space, and a softmax function is applied to produce a probability distribution over the possible next tokens.

**How it works (Auto-regressive generation):**

Decoder-only Transformers generate sequences token by token.  They start with an initial input (often a start-of-sequence token). The model predicts the probability distribution for the next token, and a token is selected (either deterministically via argmax or stochastically using sampling).  This predicted token is then appended to the input sequence, and the model is run again to predict the *next* token. This process repeats until a stopping criterion is met (e.g., a maximum length is reached or an end-of-sequence token is generated).

**Use Cases:**

Decoder-only Transformers are particularly well-suited for:

*   **Language Modeling:** Predicting the next word in a sequence, given the preceding words. This is the primary application.
*   **Text Generation:** Generating coherent and contextually relevant text based on a prompt or initial input.
*   **Code Generation:** Generating code snippets based on natural language descriptions or existing code.
*   **Chatbots and Dialogue Systems:** Generating responses in a conversational setting.

## 2) Application scenario

Let's consider the application of **text generation**. Suppose we want to use a decoder-only Transformer to generate a short story based on a given prompt.

**Prompt:** "The old lighthouse keeper noticed something unusual on the horizon..."

**Scenario:**

1.  **Input:** The prompt is fed into the decoder-only Transformer as the initial input sequence.  The prompt is first tokenized and converted into embeddings.
2.  **Auto-regressive Generation:**
    *   The model processes the prompt and predicts the probability distribution for the next word. Let's say the highest probability is assigned to the word "a".
    *   "a" is appended to the prompt: "The old lighthouse keeper noticed something unusual on the horizon... a".
    *   The model processes the updated sequence and predicts the next word. Let's say the highest probability is assigned to the word "dark".
    *   "dark" is appended: "The old lighthouse keeper noticed something unusual on the horizon... a dark".
    *   This process continues, generating words like "shape", "approaching", "rapidly", etc.
3.  **Stopping Condition:** The generation continues until a predefined length is reached, or the model generates an end-of-sequence token.
4.  **Output:** The final generated text is: "The old lighthouse keeper noticed something unusual on the horizon... a dark shape approaching rapidly. He grabbed his binoculars, his heart pounding in his chest."

In this scenario, the decoder-only Transformer leverages its learned language model to generate a coherent and plausible continuation of the given prompt.  The masked self-attention ensures that each predicted word is based only on the preceding words, maintaining the auto-regressive generation process.

## 3) Python method (if possible)

Using the `transformers` library by Hugging Face, we can easily use a pre-trained decoder-only Transformer model for text generation.  Here's an example using the `GPT2LMHeadModel`:

```python
from transformers import pipeline, set_seed

# Initialize the text generation pipeline with GPT-2
generator = pipeline('text-generation', model='gpt2')

# Set a seed for reproducibility
set_seed(42)

# Prompt text
prompt = "The old lighthouse keeper noticed something unusual on the horizon..."

# Generate text
generated_text = generator(prompt, max_length=50, num_return_sequences=1)

# Print the generated text
print(generated_text[0]['generated_text'])
```

**Explanation:**

1.  **`from transformers import pipeline, set_seed`**: Imports the necessary modules from the `transformers` library. `pipeline` provides a high-level interface for using pre-trained models, and `set_seed` ensures reproducible results.
2.  **`generator = pipeline('text-generation', model='gpt2')`**: Creates a text generation pipeline using the pre-trained GPT-2 model.  `'gpt2'` specifies the model to use. You could replace this with other decoder-only models like `'openai-gpt'`, `'distilgpt2'`, `'gpt2-medium'`, `'gpt2-large'`, `'gpt2-xl'`, or custom fine-tuned models.
3.  **`set_seed(42)`**: Sets a random seed for the generation process.  This makes the results reproducible.  Without this, the model will generate different text each time you run the code.
4.  **`prompt = "The old lighthouse keeper noticed something unusual on the horizon..."`**: Defines the prompt text that will be used as the starting point for text generation.
5.  **`generated_text = generator(prompt, max_length=50, num_return_sequences=1)`**:  This is the core part of the code.  It calls the `generator` to generate text based on the prompt.
    *   `prompt`: The input text.
    *   `max_length=50`: Specifies the maximum length of the generated text (including the prompt).
    *   `num_return_sequences=1`: Specifies that we want to generate only one sequence.
6.  **`print(generated_text[0]['generated_text'])`**: Prints the generated text. The `generator` returns a list of dictionaries, where each dictionary contains the generated text.

**Important Considerations:**

*   **Model Selection:** The quality of the generated text depends heavily on the chosen model. Larger models (e.g., `gpt2-xl`) generally produce better results but require more computational resources.
*   **Decoding Strategies:**  The `pipeline` uses a default decoding strategy (usually greedy decoding or beam search). You can customize this further by providing arguments to the `generator` function, such as `do_sample=True` for sampling-based generation, `top_k` and `top_p` for nucleus sampling, `temperature` for controlling randomness, and `repetition_penalty` to prevent the model from repeating phrases.

## 4) Follow-up question

How can we fine-tune a decoder-only Transformer model on a specific dataset to improve its performance for a particular task, such as generating code in a specific programming language or writing stories in a particular style?  What are the key considerations and techniques involved in the fine-tuning process?