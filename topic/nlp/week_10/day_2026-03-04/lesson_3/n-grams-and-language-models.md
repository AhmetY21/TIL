---
title: "N-Grams and Language Models"
date: "2026-03-04"
week: 10
lesson: 3
slug: "n-grams-and-language-models"
---

# Topic: N-Grams and Language Models

## 1) Formal definition (what is it, and how can we use it?)

**N-Grams:** An n-gram is a contiguous sequence of *n* items from a given sample of text or speech. The items can be phonemes, syllables, letters, words, or base pairs, depending on the application. Essentially, it's a sliding window of size *n* that moves across a piece of text. For example, given the sentence "The quick brown fox", the 2-grams (bigrams) would be "The quick", "quick brown", and "brown fox". 1-grams are called unigrams, 3-grams are called trigrams, and so on.

**Language Models:** A language model (LM) is a probabilistic model that assigns a probability to a sequence of words (or more generally, tokens). In simpler terms, given a sequence of words, the language model predicts the probability of the *next* word in the sequence.  Mathematically, a language model attempts to estimate P(w1, w2, ..., wn), the probability of the sequence of words w1 to wn occurring.

**N-Gram Language Models:** An n-gram language model is a type of language model that uses n-grams to predict the probability of the next word.  It approximates the probability of a word given its preceding words by only considering the preceding *n-1* words. This is based on the Markov assumption, which states that the probability of the next word depends only on the preceding *n-1* words.  For example, in a trigram language model, the probability of the word "fox" given the sequence "The quick brown" is approximated as P(fox | quick brown). The formula for the probability calculation using an n-gram language model is:

P(w_i | w_{i-n+1}, ..., w_{i-1}) ≈ Count(w_{i-n+1}, ..., w_{i-1}, w_i) / Count(w_{i-n+1}, ..., w_{i-1})

Where:

*   w_i is the i-th word in the sequence.
*   Count(w_{i-n+1}, ..., w_{i-1}, w_i) is the number of times the n-gram (w_{i-n+1}, ..., w_{i-1}, w_i) appears in the training corpus.
*   Count(w_{i-n+1}, ..., w_{i-1}) is the number of times the (n-1)-gram (w_{i-n+1}, ..., w_{i-1}) appears in the training corpus.

**How can we use it?** N-gram language models can be used for:

*   **Text generation:** Generating new text that resembles the training data.
*   **Speech recognition:** Predicting the most likely sequence of words given an audio input.
*   **Machine translation:** Choosing the most fluent translation of a sentence.
*   **Spelling correction:** Suggesting corrections for misspelled words.
*   **Autocomplete/Suggestion:** Suggesting the next word or phrase as a user types.
*   **Sentiment Analysis**: Used as features for training sentiment classifiers.
*   **Plagiarism Detection:** Identifying similarities between documents based on shared n-grams.

## 2) Application scenario

Consider a mobile keyboard app that suggests the next word as a user types.  Let's say the user has typed "I want to".  The keyboard app can use a trigram language model trained on a large corpus of text to predict the most likely next words.

The trigram language model would look up sequences that start with "I want to" in its training data and calculate the probabilities of different words following that sequence. For example, if the model has seen "I want to eat" more often than "I want to sleep" and "I want to run", it would suggest "eat" as the most probable next word. This helps users type faster and more accurately by providing contextually relevant suggestions. Smoothing techniques (explained later) are vital to prevent zero probabilities when the model encounters unseen word combinations.

## 3) Python method (if possible)

Here's a Python example using NLTK to create and use an n-gram language model:

```python
import nltk
from nltk.util import ngrams
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline

# Sample training text
text = """
The quick brown fox jumps over the lazy dog.
The dog is very lazy.
The quick rabbit also jumps.
"""

# Tokenize the text
tokenized_text = [list(map(str.lower, nltk.word_tokenize(sent)))
                  for sent in nltk.sent_tokenize(text)]

# Create padded n-grams (for smoother probability estimates)
n = 3  # Trigram model
train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)


# Train the n-gram language model
model = MLE(n)
model.fit(train_data, padded_sents)


# Example of calculating the probability of a word
print(f"Probability of 'fox' given 'brown' and 'jumps': {model.score('fox', ['brown', 'jumps'])}") # 0.0
print(f"Probability of 'dog' given 'lazy' and '.': {model.score('dog', ['lazy', '.'])}") #1.0


# Example of generating text (basic)
def generate_sentence(model, num_words=10, random_seed=42):
    """
    Generates a sentence using the n-gram language model.
    """
    import random
    random.seed(random_seed)
    text = [model.vocab.start_symbol()]
    for i in range(num_words):
        next_word_candidates = []
        context = tuple(text[-(model.order-1):])  # Get n-1 previous words as context
        for word in model.vocab:
            prob = model.score(word, context)
            if prob > 0:
                next_word_candidates.append((word,prob))
        if not next_word_candidates:
             next_word = model.vocab.end_symbol()
        else:

            next_word = random.choices([w for w, p in next_word_candidates], weights=[p for w,p in next_word_candidates], k=1)[0] # Weight probability
        text.append(next_word)
        if next_word == model.vocab.end_symbol():
            break
    return ' '.join(text[1:-1])


print(f"Generated sentence: {generate_sentence(model)}")
```

**Explanation:**

1.  **Tokenization:** The text is tokenized into sentences and then words, converting all words to lowercase for consistency.
2.  **Padded N-grams:** `padded_everygram_pipeline` adds padding symbols ('<s>' for sentence start, '</s>' for sentence end) to each sentence and generates n-grams. This is crucial for better probability estimation, especially at the beginning and end of sentences.
3.  **MLE (Maximum Likelihood Estimation):** `MLE(n)` creates an n-gram language model using maximum likelihood estimation. The `fit()` method trains the model on the padded n-gram data.
4.  **Probability Calculation:** `model.score(word, context)` calculates the probability of a given word given a specific context (the preceding n-1 words).
5.  **Text Generation:** The `generate_sentence` function shows a rudimentary way to generate text based on the probabilities learned by the model.  It starts with a start symbol and iteratively predicts the next word based on the context, using a weighted random choice based on the probabilities from the language model, until it reaches the end symbol.
**Important Notes:**
* The use of smoothing methods (e.g., Add-k smoothing, Good-Turing smoothing, Kneser-Ney smoothing) is essential in practice to handle unseen n-grams and avoid zero probabilities.  NLTK provides implementations of some smoothing techniques.  This example uses padding which helps somewhat.
* This is a simplified example. Real-world language models are much more complex and trained on significantly larger datasets.
* NLTK's `lm` module provides more sophisticated features for building and evaluating language models, including support for different smoothing techniques.

## 4) Follow-up question

How do smoothing techniques, such as Laplace smoothing (Add-1 smoothing) and Kneser-Ney smoothing, address the problem of zero probabilities in n-gram language models, and what are the trade-offs between them in terms of model performance and computational cost?