---
title: "Part-of-Speech (POS) Tagging Concepts"
date: "2026-02-12"
week: 7
lesson: 4
slug: "part-of-speech-pos-tagging-concepts"
---

# Topic: Part-of-Speech (POS) Tagging Concepts

## 1) Formal definition (what is it, and how can we use it?)

Part-of-speech (POS) tagging, also known as grammatical tagging or word-category disambiguation, is the process of assigning a part-of-speech (e.g., noun, verb, adjective, adverb) to each word in a text corpus.  The part of speech indicates how a word functions grammatically and semantically within a sentence.  It's a fundamental task in natural language processing and is a crucial intermediate step for many higher-level NLP tasks.

Formally, given a sentence *S* = *w<sub>1</sub>, w<sub>2</sub>, ..., w<sub>n</sub>*, where *w<sub>i</sub>* represents the *i*-th word in the sentence, POS tagging aims to find the optimal sequence of tags *T* = *t<sub>1</sub>, t<sub>2</sub>, ..., t<sub>n</sub>*, where *t<sub>i</sub>* represents the part-of-speech tag for the *i*-th word *w<sub>i</sub>*. The "optimality" is usually determined by maximizing a probability score, often learned from a tagged corpus.

We can use POS tagging for:

*   **Named Entity Recognition (NER):** Identifying nouns is a crucial first step. POS tagging helps to differentiate between person, location, and organization names.
*   **Information Retrieval:**  Improving search accuracy by understanding the grammatical role of words.  For example, searching for a "fast car" might benefit from knowing that "fast" is an adjective describing "car," a noun.
*   **Sentiment Analysis:** Determining the polarity of adjectives and adverbs.  The POS tag identifies that a word is indeed an adjective or adverb contributing to the sentiment.
*   **Machine Translation:**  Understanding the grammatical structure of the source language aids in translating it accurately into the target language.
*   **Parsing:**  Building parse trees that represent the grammatical structure of sentences. POS tags serve as the terminals of the parse tree.
*   **Text-to-Speech (TTS):**  Knowing the POS of a word can help in pronunciation (e.g., the word "lead" has different pronunciations as a noun vs. a verb).
*   **Text Summarization:** Identifying important nouns and verbs.

## 2) Application scenario

Imagine you're building a chatbot that helps users find information about movies. A user might ask: "Show me action movies directed by Christopher Nolan." To accurately process this request, the chatbot needs to understand the grammatical structure of the sentence.

*   **"Show"**: Verb (commanding action)
*   **"me"**: Pronoun (object of the verb)
*   **"action"**: Adjective (describing the type of movie)
*   **"movies"**: Noun (the object being searched for)
*   **"directed"**: Verb (past participle, describing the movies)
*   **"by"**: Preposition
*   **"Christopher Nolan"**: Proper Noun (director's name)

With this POS information, the chatbot can correctly interpret the query. It knows "action" modifies "movies," indicating the genre.  It knows "directed by Christopher Nolan" is a phrase describing the movies.  Without POS tagging, the chatbot might incorrectly interpret "action" as a verb, leading to irrelevant search results.

## 3) Python method (if possible)

We can use the `spaCy` library for POS tagging in Python.

```python
import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Example sentence
text = "The quick brown fox jumps over the lazy dog."

# Process the text
doc = nlp(text)

# Iterate through the tokens and print the word and its POS tag
for token in doc:
    print(token.text, token.pos_, token.tag_)

# The 'pos_' attribute gives the coarse-grained POS tag
# The 'tag_' attribute gives the fine-grained POS tag (more specific)
```

This code will output something like this:

```
The DET DT
quick ADJ JJ
brown ADJ JJ
fox NOUN NN
jumps VERB VBZ
over ADP IN
the DET DT
lazy ADJ JJ
dog NOUN NN
. PUNCT .
```

*   `DET`: Determiner
*   `ADJ`: Adjective
*   `NOUN`: Noun
*   `VERB`: Verb
*   `ADP`: Adposition (preposition or postposition)
*   `PUNCT`: Punctuation

Other Python libraries for POS tagging include NLTK (which offers multiple taggers) and TextBlob (which is built on NLTK and provides a simpler interface).

## 4) Follow-up question

How do Hidden Markov Models (HMMs) relate to POS tagging? Explain the basic idea of how HMMs are used for this task.