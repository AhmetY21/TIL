---
title: "N-Grams and Language Models"
date: "2026-02-14"
week: 7
lesson: 2
slug: "n-grams-and-language-models"
---

# Topic: N-Grams and Language Models

## 1) Formal definition (what is it, and how can we use it?)

**N-Grams:** An N-gram is a contiguous sequence of *n* items from a given sample of text or speech. These items can be characters, syllables, words, or base pairs, depending on the application. When the items are words, we typically call it a word n-gram.

Examples:

*   **Unigram (1-gram):** "the", "quick", "brown", "fox"
*   **Bigram (2-gram):** "the quick", "quick brown", "brown fox"
*   **Trigram (3-gram):** "the quick brown", "quick brown fox"

**Language Model (LM):** A language model is a probability distribution over sequences of words. It attempts to assign a probability to a sentence (or any sequence of words). Specifically, it estimates the probability of a word appearing given the words that precede it. N-gram language models approximate this probability by considering only the *n-1* preceding words.

Formally, we want to estimate P(w<sub>i</sub> | w<sub>1</sub>, w<sub>2</sub>, ..., w<sub>i-1</sub>), the probability of word w<sub>i</sub> given the preceding words. An n-gram language model approximates this as:

P(w<sub>i</sub> | w<sub>1</sub>, w<sub>2</sub>, ..., w<sub>i-1</sub>) ≈ P(w<sub>i</sub> | w<sub>i-n+1</sub>, w<sub>i-n+2</sub>, ..., w<sub>i-1</sub>)

For example, using a trigram language model:

P(fox | the quick brown) ≈ P(fox | quick brown)

We use these probabilities to:

*   **Predict the next word:** Given a sequence of words, predict the most likely word to follow.
*   **Generate text:** Generate new text by sampling words based on the probabilities learned from the training data.
*   **Score sentences:** Assign a probability score to a sentence, indicating its likelihood in the language.
*   **Machine Translation:** Determine which translation is most likely to be correct and fluent in the target language.
*   **Speech Recognition:** Help to refine the choices in speech-to-text transcription.
*   **Spell Checking:** Suggest corrections for misspelled words, based on the words around them.

## 2) Application scenario

Let's consider a scenario in a text-based chatbot designed to assist users in booking flights. The chatbot has already received the user input "I want to fly from".  Using an n-gram language model, the chatbot can predict the next word, providing useful suggestions to the user.

If the chatbot uses a trigram model trained on a large corpus of travel-related text, it might have learned the following probabilities:

*   P("London" | "fly from") = 0.25
*   P("New York" | "fly from") = 0.20
*   P("Paris" | "fly from") = 0.15
*   P("Chicago" | "fly from") = 0.10
*   P("anywhere" | "fly from") = 0.05

Based on these probabilities, the chatbot can suggest "London", "New York", "Paris", "Chicago", or "anywhere" to the user, ranked in order of likelihood. This dramatically improves the user experience by providing relevant and helpful suggestions, reducing the effort required to input their desired origin airport. This improves the user experience.

## 3) Python method (if possible)

Here's a simple Python example using the NLTK library to generate n-grams and calculate their frequencies. This does *not* create a full language model with probabilities, but shows how to get the raw counts needed for one:

```python
import nltk
from nltk.util import ngrams
from collections import Counter

def generate_ngrams(text, n):
    """Generates n-grams from a string of text."""
    tokens = nltk.word_tokenize(text) # Tokenize the text
    n_grams = ngrams(tokens, n)
    return n_grams

def calculate_ngram_frequencies(text, n):
    """Calculates the frequencies of n-grams in a text."""
    n_grams = generate_ngrams(text, n)
    ngram_counts = Counter(n_grams)
    return ngram_counts

# Example usage
text = "The quick brown fox jumps over the lazy dog. The quick brown rabbit also jumps."
n = 2  # Bigrams

ngram_counts = calculate_ngram_frequencies(text, n)

print(f"{n}-gram counts:")
for ngram, count in ngram_counts.items():
    print(f"{ngram}: {count}")


# Simple example to show how to derive probabilities (not complete smoothing etc)
total_ngrams = sum(ngram_counts.values())

print("\nBigram Probabilities (simplified example, no smoothing):")
for ngram, count in ngram_counts.items():
    probability = count / total_ngrams
    print(f"P({ngram}) = {probability:.4f}")
```

This code snippet first tokenizes the input text, then generates n-grams of the specified size. Finally, it calculates the frequency of each n-gram and prints the results. A small section is added at the end to demonstrate how probabilities are derived from the counts. This is a simplified example; real language models usually employ smoothing techniques to handle unseen n-grams.

## 4) Follow-up question

How do smoothing techniques (e.g., Laplace smoothing, Good-Turing smoothing, Kneser-Ney smoothing) address the problem of zero probabilities for unseen n-grams in language models, and why are these techniques important for the practical performance of n-gram models?