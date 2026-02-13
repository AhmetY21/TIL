---
title: "N-Grams and Language Models"
date: "2026-02-13"
week: 7
lesson: 6
slug: "n-grams-and-language-models"
---

# Topic: N-Grams and Language Models

## 1) Formal definition (what is it, and how can we use it?)

**N-grams:** An n-gram is a contiguous sequence of *n* items from a given sample of text or speech. The items can be characters, syllables, words, or even phrases.  For example, in the sentence "The quick brown fox jumps over the lazy dog.", the 2-grams (or bigrams) would be: "The quick", "quick brown", "brown fox", "fox jumps", "jumps over", "over the", "the lazy", "lazy dog". The 3-grams (or trigrams) would be: "The quick brown", "quick brown fox", "brown fox jumps", "fox jumps over", "jumps over the", "over the lazy", "the lazy dog". A 1-gram is also known as a unigram.

**Language Models:** A language model (LM) is a probabilistic model that assigns a probability to a sequence of words (or, more generally, any sequence of tokens). In simpler terms, given a sequence of words, a language model tries to predict the next word. LMs are used to assess the "likelihood" or "grammaticality" of a sentence. The higher the probability, the more likely the sentence is to occur (according to the model's training data).

**How N-grams are used in Language Models:**  N-gram language models are a type of statistical language model that uses the concept of n-grams to estimate the probability of a word occurring given the preceding *n-1* words. The core idea is that the probability of a word depends only on the *n-1* preceding words.  This is known as the Markov assumption.  Mathematically, this is expressed as:

P(w<sub>i</sub> | w<sub>1</sub>, w<sub>2</sub>, ..., w<sub>i-1</sub>) ≈ P(w<sub>i</sub> | w<sub>i-n+1</sub>, w<sub>i-n+2</sub>, ..., w<sub>i-1</sub>)

Where:

*   P(w<sub>i</sub> | w<sub>1</sub>, w<sub>2</sub>, ..., w<sub>i-1</sub>) is the probability of the word w<sub>i</sub> given the preceding words.
*   P(w<sub>i</sub> | w<sub>i-n+1</sub>, w<sub>i-n+2</sub>, ..., w<sub>i-1</sub>) is the approximation used by the n-gram model, which considers only the n-1 preceding words.

This probability is typically estimated from a large corpus of text by counting the occurrences of n-grams and normalizing.

**Use:** N-gram language models are used for:

*   **Speech Recognition:**  To improve the accuracy of transcription by predicting the most likely sequence of words.
*   **Machine Translation:** To evaluate the fluency of translated sentences.
*   **Text Generation:** To generate realistic text, such as chatbot responses or article headlines.
*   **Spelling Correction:**  To identify and correct misspelled words based on context.
*   **Grammar Checking:**  To detect grammatical errors.
*   **Information Retrieval:**  To rank search results based on relevance.
*   **Text classification:** Can be features used in text classification algorithms.
## 2) Application scenario

**Application scenario: Autocompletion in search engines or text editors.**

Imagine you are typing "I went to the" in a search engine.  An n-gram language model, trained on a large corpus of web text, can predict the most likely next words. If the model uses trigrams, it considers the n-gram "went to the".  Based on its training data, it might predict that "I went to the beach", "I went to the store", "I went to the park" are the most likely completions. The search engine then displays these suggestions to the user, speeding up the search process and improving user experience. The predictions are determined by calculating the probabilities: P(beach | went to the), P(store | went to the), P(park | went to the), etc. and suggesting the options with the highest probabilities.

## 3) Python method (if possible)

```python
import nltk
from nltk.util import ngrams
from collections import Counter

def create_ngrams(text, n):
    """
    Generates n-grams from a given text.

    Args:
        text (str): The input text.
        n (int): The size of the n-gram.

    Returns:
        list: A list of n-grams.
    """
    tokens = nltk.word_tokenize(text) #Tokenize into words
    n_grams = list(ngrams(tokens, n))
    return n_grams

def calculate_ngram_probabilities(corpus, n):
  """
  Calculates probabilities of n-grams in a corpus using maximum likelihood estimation.

  Args:
      corpus (str): The corpus of text.
      n (int): The size of the n-gram.

  Returns:
      dict: A dictionary where keys are n-grams (tuples) and values are their probabilities.
  """
  all_ngrams = create_ngrams(corpus, n)
  ngram_counts = Counter(all_ngrams)

  if n == 1:
      total_count = len(all_ngrams) #Total number of unigrams.
  else:
      #Use n-1 grams to normalize, this is the context.
      all_context_ngrams = create_ngrams(corpus, n-1)
      context_ngram_counts = Counter(all_context_ngrams)
      total_count = len(all_context_ngrams) #Total number of n-1 grams
  ngram_probabilities = {}

  for ngram, count in ngram_counts.items():
      if n == 1:
          ngram_probabilities[ngram] = count / total_count
      else:
        # Get the count of the n-1 gram that precedes the current ngram
        context_ngram = ngram[:-1] # get the first n-1 tokens.
        context_count = context_ngram_counts[context_ngram]
        ngram_probabilities[ngram] = count / context_count

  return ngram_probabilities

# Example usage:
text = "The quick brown fox jumps over the lazy dog. The quick brown cat sleeps."
bigrams = create_ngrams(text, 2)
print("Bigrams:", bigrams)

ngram_probs = calculate_ngram_probabilities(text, 2)
print("\nBigram Probabilities (example):")
print(f"P(('brown', 'fox')): {ngram_probs.get(('brown', 'fox'), 0):.4f}") # Access prob with .get() in case ngram doesn't exist.
print(f"P(('quick', 'brown')): {ngram_probs.get(('quick', 'brown'), 0):.4f}")
print(f"P(('lazy', 'dog')): {ngram_probs.get(('lazy', 'dog'), 0):.4f}")

unigram_probs = calculate_ngram_probabilities(text, 1)
print("\nUnigram Probabilities (example):")
print(f"P(('the',)): {unigram_probs.get(('the',), 0):.4f}")
print(f"P(('quick',)): {unigram_probs.get(('quick',), 0):.4f}")
```

**Explanation:**

1.  **`create_ngrams(text, n)`:**
    *   Takes text and n as input.
    *   Uses `nltk.word_tokenize` to split the text into words.
    *   Uses `nltk.util.ngrams` to generate n-grams from the tokens.
    *   Returns a list of n-grams (each n-gram is a tuple).

2.  **`calculate_ngram_probabilities(corpus, n)`:**
    * Takes corpus of text and n as input.
    * Calls `create_ngrams` to generate all n-grams in the corpus.
    * Uses `Counter` to count the occurrences of each n-gram.
    * If n == 1, just calculate the probability of each unigram by dividing it's count by the total number of unigrams.
    * If n > 1, this function implements Maximum Likelihood Estimation (MLE) for estimating the probability of an n-gram. It divides the count of a specific n-gram by the count of its n-1 gram context. The probabilities are stored in a dictionary where the keys are n-grams (tuples), and the values are their calculated probabilities.

**Important Considerations:**

*   **Smoothing:**  The above code does not implement smoothing techniques (e.g., Laplace smoothing, Kneser-Ney smoothing).  Without smoothing, n-grams that do not appear in the training data will have a probability of zero. Smoothing techniques help to avoid zero probabilities and improve the model's generalization ability.
*   **Out-of-Vocabulary (OOV) words:**  The model will not be able to handle words that are not in the training vocabulary.  Techniques for handling OOV words include using a special `<UNK>` token to represent unknown words.
*   **Corpus Size:** The larger the training corpus, the better the model will be at capturing the nuances of the language.

## 4) Follow-up question

How do different smoothing techniques (e.g., Laplace smoothing, Kneser-Ney smoothing) affect the performance of an n-gram language model, and when would you choose one smoothing technique over another?