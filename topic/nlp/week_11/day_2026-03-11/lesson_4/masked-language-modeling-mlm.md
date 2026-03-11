---
title: "Masked Language Modeling (MLM)"
date: "2026-03-11"
week: 11
lesson: 4
slug: "masked-language-modeling-mlm"
---

# Topic: Masked Language Modeling (MLM)

## 1) Formal definition (what is it, and how can we use it?)

Masked Language Modeling (MLM) is a training task where the model is trained to predict randomly masked words in a sentence. Specifically, during training, a percentage (typically 15%) of the words in the input sequence are randomly selected and replaced with a special "[MASK]" token. The model's objective is to predict the original, unmasked words based on the context provided by the surrounding words.

Formally, given a sequence of tokens `x = [x_1, x_2, ..., x_n]`, MLM selects a random subset of indices `M ⊆ {1, 2, ..., n}`. The input to the model becomes `x' = [x'_1, x'_2, ..., x'_n]`, where `x'_i = x_i` if `i ∉ M` and `x'_i = [MASK]` if `i ∈ M`. The model then tries to maximize the conditional probability `P(x_M | x')`, where `x_M` represents the original tokens at the masked positions.

We can use MLM to:

*   **Pre-train contextualized word embeddings:** MLM forces the model to learn bidirectional contextual representations of words, which are crucial for understanding the meaning of words based on their context. This pre-training can then be used to initialize the model for downstream tasks.
*   **Fine-tune for various NLP tasks:** After pre-training with MLM, the model can be fine-tuned on specific downstream tasks like text classification, question answering, and named entity recognition. The knowledge gained during MLM pre-training helps the model learn these tasks more efficiently.
*   **Improve model robustness:** By masking words during training, MLM makes the model more robust to missing or corrupted input data.

## 2) Application scenario

**Scenario:** Sentiment analysis of customer reviews.

**Use of MLM:**

1.  **Pre-training:** We pre-train a BERT-like model on a large corpus of text data (e.g., Wikipedia, books). During pre-training, the model learns to predict masked words, building a general understanding of language.
2.  **Fine-tuning:**  We fine-tune the pre-trained model on a dataset of customer reviews, where each review is labeled with a sentiment (e.g., positive, negative, neutral). During fine-tuning, the model learns to map the input review text to the correct sentiment label. The knowledge from the pre-training using MLM enables the fine-tuning to be performed much more efficiently and results in better overall model performance.

Because MLM allows the model to learn bidirectional contextual representations of words, it's particularly helpful in this scenario. The model can understand the nuances of language in the reviews and determine the sentiment even if some words are ambiguous or missing (due to potential typos or slang).

## 3) Python method (if possible)

```python
from transformers import BertTokenizer, BertForMaskedLM, pipeline

# Load pre-trained tokenizer and model
model_name = "bert-base-uncased"  # Or any other BERT variant
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)


def mask_and_predict(text, mask_index):
    """Masks a word in the text and predicts the masked word."""
    words = text.split()
    original_word = words[mask_index]
    words[mask_index] = "[MASK]"
    masked_text = " ".join(words)

    # Tokenize the input
    input_ids = tokenizer.encode(masked_text, return_tensors="pt")

    # Predict the masked word
    with torch.no_grad():
        outputs = model(input_ids)
        predictions = outputs.logits

    # Get the predicted token
    predicted_token_index = torch.argmax(predictions[0, mask_index + 1]).item() # +1 offset for [CLS] token
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_token_index])[0]

    return original_word, predicted_token, masked_text


# Example usage
import torch
text = "The quick brown fox jumps over the lazy dog."
mask_index = 2 # Mask the word "brown"

original_word, predicted_word, masked_text = mask_and_predict(text, mask_index)

print(f"Original text: {text}")
print(f"Masked text: {masked_text}")
print(f"Original word: {original_word}")
print(f"Predicted word: {predicted_word}")


# Using pipeline (easier but less control)
unmasker = pipeline('fill-mask', model=model_name)
result = unmasker(text.replace(text.split()[mask_index], "[MASK]"))
print(f"\nUsing pipeline: {result[0]['token_str']}")
```

This code first loads a pre-trained BERT model and tokenizer.  Then, the `mask_and_predict` function takes a text string and the index of the word to mask. It replaces the specified word with the "[MASK]" token, feeds the masked text to the BERT model, and predicts the most likely word to fill the mask. Finally, it returns the original word, predicted word, and masked text. The code also shows how to accomplish the same task using the `pipeline` function, which provides a higher-level, easier-to-use interface. The output shows both methods and makes it clear how they relate.  You will need to install the `transformers` library: `pip install transformers`.

## 4) Follow-up question

How does the percentage of masked tokens during MLM pre-training (e.g., 15% in BERT) affect the model's performance and learning process? What are the trade-offs involved in choosing a different masking percentage? For example, how would the training change if the percentage were lowered to 5% or raised to 30%?