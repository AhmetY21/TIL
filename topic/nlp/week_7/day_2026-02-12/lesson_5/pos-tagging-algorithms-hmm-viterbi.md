---
title: "POS Tagging Algorithms (HMM, Viterbi)"
date: "2026-02-12"
week: 7
lesson: 5
slug: "pos-tagging-algorithms-hmm-viterbi"
---

# Topic: POS Tagging Algorithms (HMM, Viterbi)

## 1) Formal definition (what is it, and how can we use it?)

**POS Tagging:** Part-of-Speech (POS) tagging, also known as grammatical tagging or word-category disambiguation, is the process of assigning a part-of-speech (e.g., noun, verb, adjective, adverb) to each word in a text. This assignment is based on both the word's definition and its context within the sentence.

**Hidden Markov Model (HMM) for POS Tagging:** An HMM is a probabilistic sequence model used to label each unit in a sequence of observations. In the context of POS tagging, the observed sequence is the sequence of words in a sentence, and the hidden sequence is the sequence of POS tags corresponding to those words.  The HMM model consists of:

*   **States:** The set of POS tags (e.g., Noun, Verb, Adjective).
*   **Observations:** The sequence of words in the sentence.
*   **Transition Probabilities:** The probability of transitioning from one POS tag to another POS tag (e.g., P(Verb | Noun) - the probability of a verb following a noun).  Formally, P(tag_i | tag_{i-1}).
*   **Emission Probabilities (Observation Likelihoods):** The probability of a word being assigned a particular POS tag (e.g., P(run | Verb) - the probability of the word "run" being a verb). Formally, P(word_i | tag_i).
*   **Initial Probabilities:**  The probability of starting a sentence with a specific POS tag (e.g., P(Noun) - the probability that the first word is a noun).

**Viterbi Algorithm:** The Viterbi algorithm is a dynamic programming algorithm used to find the most likely sequence of hidden states (POS tags in our case) given a sequence of observed events (words) and an HMM. It efficiently computes the path with the highest probability through all possible state sequences. It uses these calculations:

1.  **Initialization:** Calculate the probability of each starting state.
2.  **Recursion:** Calculate the maximum probability of reaching each state at each time step, considering all possible previous states.  This involves multiplying the transition probability from the previous state to the current state, the emission probability of the current observation given the current state, and the maximum probability of reaching the previous state.
3.  **Termination:**  Find the state with the highest probability at the end of the sequence.
4.  **Backtracking:** Trace back through the sequence of states that led to the highest probability, thus reconstructing the most likely sequence of POS tags.

**How can we use it?**

We can use HMMs and the Viterbi algorithm to:

*   Automatically assign POS tags to text, which is crucial for many NLP tasks.
*   Improve the accuracy of downstream NLP applications like machine translation, information retrieval, and question answering.
*   Perform text analysis, such as identifying the dominant grammatical structures in a document or comparing the writing styles of different authors.
*   Assist in speech recognition by helping to distinguish between words that sound alike but have different POS tags and meanings.

## 2) Application scenario

**Scenario:** Imagine we want to build a system that extracts information about companies and their products from news articles.  To do this effectively, we need to identify the relationships between different words in a sentence.  For example, we might want to find sentences where a company name (a proper noun) is followed by a verb describing its actions.

**Application:** We can use an HMM with the Viterbi algorithm to perform POS tagging on the news articles. By accurately tagging the words in each sentence, we can then easily identify proper nouns (company names) and verbs, and look for specific patterns like "<Company Name> <Verb>".  This allows us to automatically extract information about company activities from the text.

**Example:**

Consider the sentence: "Apple announced a new iPhone."

1.  **Input:** The sequence of words: "Apple announced a new iPhone ."
2.  **HMM & Viterbi:** The HMM (trained on a large corpus of tagged text) and the Viterbi algorithm would analyze the sentence.
3.  **Output:** The following POS tags:

    *   Apple: NNP (Proper Noun, singular)
    *   announced: VBD (Verb, past tense)
    *   a: DT (Determiner)
    *   new: JJ (Adjective)
    *   iPhone: NN (Noun, singular)
    *   .: . (Punctuation)

Now, based on these tags, our information extraction system can identify "Apple" as a company and "announced" as a verb related to its activity, allowing it to extract the information that Apple announced something.

## 3) Python method (if possible)

While implementing a full HMM and Viterbi algorithm from scratch can be involved, we can use existing libraries like `nltk` or `spaCy` to perform POS tagging.  Here's an example using `nltk` with a pre-trained tagger:

```python
import nltk

# Download necessary resources (if not already downloaded)
try:
    nltk.data.find("taggers/averaged_perceptron_tagger")
except LookupError:
    nltk.download("averaged_perceptron_tagger")
try:
    nltk.data.find("corpora/brown")
except LookupError:
    nltk.download("brown")

from nltk.tokenize import word_tokenize

def pos_tag_sentence(sentence):
  """
  Performs POS tagging on a given sentence using nltk's averaged_perceptron_tagger.

  Args:
    sentence: The sentence to tag (string).

  Returns:
    A list of tuples, where each tuple contains a word and its POS tag.
  """
  try:
      tokens = word_tokenize(sentence)  # Tokenize the sentence
      tagged_words = nltk.pos_tag(tokens)  # POS tag the tokens
      return tagged_words
  except LookupError:
        print("Error: Required NLTK resources not found.  Please run nltk.download('punkt') if this occurs for the first time")
        return None




# Example usage
sentence = "Apple announced a new iPhone."
tagged_sentence = pos_tag_sentence(sentence)

if tagged_sentence:
    print(tagged_sentence) # Output: [('Apple', 'NNP'), ('announced', 'VBD'), ('a', 'DT'), ('new', 'JJ'), ('iPhone', 'NN'), ('.', '.')]

```

**Explanation:**

1.  **Import nltk:** Imports the NLTK library.
2.  **Download Resources:** Downloads the necessary NLTK resources if they are not already present. `nltk.download()` provides access to various datasets and models. The `averaged_perceptron_tagger` is a pre-trained tagger that uses an averaged perceptron algorithm.  `punkt` is required for tokenization.
3.  **Tokenization:** Uses `word_tokenize` to split the sentence into individual words (tokens).
4.  **POS Tagging:** The `nltk.pos_tag()` function then applies the tagger to the tokens and returns a list of tuples, where each tuple contains a word and its corresponding POS tag.
5.  **Output:** Prints the tagged sentence, showing the POS tag assigned to each word.

This example demonstrates how to use a pre-trained tagger in `nltk`. You can also train your own HMM-based POS tagger using NLTK, but that requires significantly more code and a labeled training corpus.

## 4) Follow-up question

How would you handle unknown words (words not seen during training) when using an HMM for POS tagging? What techniques can be used to improve the accuracy of the POS tagger for these out-of-vocabulary words?