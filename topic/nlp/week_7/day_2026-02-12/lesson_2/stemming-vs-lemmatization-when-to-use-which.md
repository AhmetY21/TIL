---
title: "Stemming vs. Lemmatization: When to use which?"
date: "2026-02-12"
week: 7
lesson: 2
slug: "stemming-vs-lemmatization-when-to-use-which"
---

# Topic: Stemming vs. Lemmatization: When to use which?

## 1) Formal definition (what is it, and how can we use it?)

Stemming and lemmatization are text normalization techniques used in Natural Language Processing (NLP) to reduce words to their base or root form. This helps to group together words with similar meanings, even if they have different inflections or derivations. This is crucial for tasks like information retrieval, text classification, and sentiment analysis, as it reduces the dimensionality of the data and improves the ability of algorithms to generalize.

*   **Stemming:** Stemming is a heuristic process that chops off the ends of words in the hope of achieving this goal correctly most of the time. It's a simpler and faster process compared to lemmatization. Stemming algorithms often follow rules to remove common prefixes and suffixes, regardless of the word's context or meaning. This can sometimes result in stems that are not actual words.

*   **Lemmatization:** Lemmatization, on the other hand, is a more sophisticated approach that uses vocabulary and morphological analysis to find the base or dictionary form of a word, known as the lemma. It takes into account the context of the word and aims to return a valid word that represents the root meaning. This often involves looking up the word in a dictionary or using grammatical rules. Because it tries to find the dictionary form, lemmatization can provide a more accurate representation of the intended meaning.

We use these techniques to:

*   **Reduce vocabulary size:** By grouping variations of the same word, the number of unique tokens in a text corpus is reduced, improving efficiency.
*   **Improve search accuracy:** Searching for "running" might also return results containing "run" when stemming or lemmatization is applied.
*   **Improve text classification:** By reducing noise from variations in word forms, the signal for classification tasks can be strengthened.
*   **Standardize data:** Ensures consistent representation of words across different documents or data sources.

## 2) Application scenario

Here's a breakdown of when you might prefer one over the other:

*   **Stemming:**
    *   **Information Retrieval:** When speed is more important than accuracy (e.g., quick search in a large database), stemming can be a good choice. The "good enough" stemming often provides adequate results without a significant performance hit.
    *   **Sentiment Analysis (basic):** If you are only interested in the general sentiment (positive, negative, neutral), stemming can often provide sufficient information.
    *   **Resource-constrained environments:** Stemming requires less computational power and memory, making it suitable for resource-limited devices.

*   **Lemmatization:**
    *   **Question Answering Systems:** When the meaning of the word is crucial for understanding the question and finding the correct answer.
    *   **Chatbots/Dialogue Systems:** Accurate understanding of user input is critical, and lemmatization helps preserve the intended meaning.
    *   **Text Summarization:** When preserving the original meaning and fluency of the text is important, lemmatization is preferred.
    *   **High-accuracy Information Retrieval:** If retrieving only highly relevant results is paramount, lemmatization will likely outperform stemming.
    *   **Tasks that need human readability:** Since lemmatization usually provides a proper dictionary word as the root, it is better for cases where the output will be presented directly to a human.

For example:

*   Consider the sentence "The cats are running quickly."

    *   Stemming might reduce "cats" to "cat", "running" to "run", and "quickly" to "quickli".
    *   Lemmatization might reduce "cats" to "cat", "running" to "run", and "quickly" to "quick".

In this simple example, the results seem similar. However, consider "better". Stemming might not change it at all, while lemmatization would reduce it to "good".

## 3) Python method (if possible)

```python
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt') # Required for word_tokenize
nltk.download('wordnet') # Required for WordNetLemmatizer
nltk.download('omw-1.4') # Required for WordNetLemmatizer

# Sample text
text = "The cats are running quickly, better than ever before."

# Tokenize the text
tokens = word_tokenize(text)

# Stemming
porter_stemmer = PorterStemmer()
stemmed_words = [porter_stemmer.stem(word) for word in tokens]

print("Stemmed words:", stemmed_words)

# Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()
lemmatized_words = [wordnet_lemmatizer.lemmatize(word) for word in tokens]

print("Lemmatized words:", lemmatized_words)

# Lemmatization with POS tag
lemmatized_words_pos = [wordnet_lemmatizer.lemmatize(word, pos='v') for word in tokens] # 'v' indicates verb
print("Lemmatized words with POS:", lemmatized_words_pos)


# Example: Demonstrating POS tagging importance

text2 = "This meeting is a matter of concern. The meetings are important."

tokens2 = word_tokenize(text2)

print("Lemmatization without POS tags on tokens2:",[wordnet_lemmatizer.lemmatize(word) for word in tokens2]) #meeting is not changed
print("Lemmatization with POS tags on tokens2:",[wordnet_lemmatizer.lemmatize(word, pos='v') for word in tokens2]) # changes meeting to meet when treated as a verb.



```

Key points about the Python code:

*   **`nltk` library:**  We use the Natural Language Toolkit (nltk) library, a popular Python library for NLP tasks.
*   **`PorterStemmer`:** An implementation of the Porter stemming algorithm.
*   **`WordNetLemmatizer`:** A lemmatizer based on WordNet, a lexical database for English.
*   **`word_tokenize`:** Tokenizes the text into individual words.
*   **POS Tagging and Lemmatization:** Notice the use of `pos='v'` in the lemmatization.  Lemmatization often benefits greatly from Part-of-Speech (POS) tagging to determine the correct lemma based on the word's grammatical role. In the second example in the code, it is clear how much of a difference POS-tagging makes for lemmatization to work properly.
*   **Download requirements:** The nltk library requires downloading data before running, hence the calls to `nltk.download`.

## 4) Follow-up question

How do stemming and lemmatization handle out-of-vocabulary words, and what are the potential implications for downstream NLP tasks?