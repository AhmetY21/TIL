---
title: "Stemming vs. Lemmatization: When to use which?"
date: "2026-03-03"
week: 10
lesson: 1
slug: "stemming-vs-lemmatization-when-to-use-which"
---

# Topic: Stemming vs. Lemmatization: When to use which?

## 1) Formal definition (what is it, and how can we use it?)

Stemming and lemmatization are text normalization techniques used in Natural Language Processing (NLP) to reduce words to their base or root form. They both aim to consolidate variations of a word (e.g., "running", "ran", "runs" all become variations of "run") to improve the accuracy and efficiency of text analysis tasks.

*   **Stemming:** A heuristic process that chops off the ends of words in the hope of achieving this goal correctly most of the time. It's a simpler and faster method, but it can sometimes result in stems that are not actual words (over-stemming) or fail to reduce related words to the same stem (under-stemming). The goal is *computational efficiency* and not necessarily linguistic accuracy.

*   **Lemmatization:** A more sophisticated process that considers the morphological analysis of words.  It uses vocabulary and morphological analysis to find the base or dictionary form of a word, which is known as the lemma.  The lemma is always a valid word.  This involves using knowledge of the word's part-of-speech (POS) and context to determine the correct base form.  The goal is *linguistic accuracy* and correct representation.

**How can we use them?**

Both stemming and lemmatization can be used to:

*   **Reduce vocabulary size:** By reducing words to their root forms, you effectively decrease the number of unique tokens in your text data. This is useful for reducing the dimensionality of feature vectors in machine learning models.
*   **Improve search results:**  Users often use different forms of a word when searching.  Stemming/Lemmatization allows matching across these forms.
*   **Improve text classification/clustering:** By treating related words as the same, the models can focus on the core meaning of the text rather than surface-level variations.
*   **Improve topic modeling:** Similarly, grouping variations of words allows topic models to identify underlying themes more effectively.

## 2) Application scenario

Consider these scenarios:

*   **Stemming:**  Imagine you are building a search engine that needs to quickly return relevant results for user queries.  Speed is crucial.  You might choose stemming to rapidly process a large amount of text and index it.  High accuracy isn't *as* critical as speed. For example, a search for "connecting" might return documents containing "connect", "connection", or "connected". The fact that stemming might sometimes result in a non-word stem is acceptable for this use case.
*   **Lemmatization:**  Suppose you are building a chatbot that needs to understand the nuances of user input.  You want to accurately interpret the meaning of their requests. In this case, you'd prefer lemmatization. For example, correctly identifying the lemma "be" for forms like "is", "are", and "was" is important for accurately understanding the user's intent. Consider a sentence like, "The children were playing". Lemmatization will convert "were" to "be", providing useful information about the sentence's structure and meaning that simple stemming wouldn't. Also, consider medical records - accuracy is paramont, so you'd want lemmatization over stemming.

In summary:

*   **Stemming:**  Use when speed is paramount and accuracy is less critical (e.g., information retrieval, search engines with large indices, some text categorization tasks).
*   **Lemmatization:**  Use when accuracy and understanding the true meaning of the text are crucial (e.g., chatbots, sentiment analysis, question answering, detailed text analysis, medical NLP).

## 3) Python method (if possible)

```python
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources (run this once)
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')  # Required for POS tagging in lemmatization

# Example sentence
sentence = "The cats were running quickly through the city. They are better runners than mice."

# Tokenize the sentence
tokens = word_tokenize(sentence)

# Stemming using PorterStemmer
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in tokens]
print("Stemmed words:", stemmed_words)

# Lemmatization using WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# A helper function to convert NLTK's POS tags to WordNet's POS tags
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return nltk.corpus.wordnet.NOUN  # Default to noun

# POS tag the tokens
pos_tags = nltk.pos_tag(tokens)

# Lemmatize with POS tags
lemmatized_words = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in pos_tags]
print("Lemmatized words:", lemmatized_words)
```

## 4) Follow-up question

Are there situations where it makes sense to apply stemming *after* lemmatization, or vice versa, to improve the normalization results? If so, what would such situations look like, and what would be the expected benefit?