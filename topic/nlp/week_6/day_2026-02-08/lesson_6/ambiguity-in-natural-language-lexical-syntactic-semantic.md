---
title: "Ambiguity in Natural Language: Lexical, Syntactic, Semantic"
date: "2026-02-08"
week: 6
lesson: 6
slug: "ambiguity-in-natural-language-lexical-syntactic-semantic"
---

# Topic: Ambiguity in Natural Language: Lexical, Syntactic, Semantic

## 1) Formal definition (what is it, and how can we use it?)

Ambiguity in Natural Language Processing (NLP) refers to the property of language where a single linguistic expression (word, phrase, sentence) can have multiple possible interpretations. Understanding and resolving ambiguity is crucial for NLP tasks to accurately process and understand human language. There are three primary types of ambiguity:

*   **Lexical Ambiguity:** This occurs when a word has multiple meanings. It's about a single word having different possible senses (definitions). We can use lexical disambiguation techniques to determine the appropriate sense based on context.

*   **Syntactic Ambiguity (Structural Ambiguity):** This occurs when a sentence can be parsed in multiple ways, each resulting in a different meaning.  It's about the grammatical structure of a sentence being interpreted in various forms. Understanding the syntactic structure is fundamental to resolving this type of ambiguity.

*   **Semantic Ambiguity:** This arises when a sentence has a single syntactic structure but different possible interpretations due to the interaction of word meanings. It often involves resolving pronoun references, quantifier scope, or dealing with general world knowledge. Semantic understanding of the entire sentence and surrounding context is crucial.

We use the understanding of these ambiguities to:

*   **Improve NLP model accuracy:** By explicitly modeling ambiguity, we can build more robust and reliable NLP systems.
*   **Design better parsing algorithms:** Parsing algorithms must be able to handle syntactic ambiguity effectively.
*   **Improve machine translation:** Accurate translation depends on correctly resolving ambiguities in the source language.
*   **Enhance information retrieval:** Understanding the different possible meanings of a query allows for more relevant search results.
*   **Develop more human-like AI:**  Humans naturally resolve ambiguity; making AI systems do the same makes them more natural.

## 2) Application scenario

**Lexical Ambiguity Example:** "I went to the bank to deposit money." and "I sat on the bank of the river." The word "bank" has two distinct meanings: a financial institution and the edge of a river. In a financial application, the former meaning is far more likely. For a chatbot providing information about local parks and waterways, the latter meaning would be relevant.

**Syntactic Ambiguity Example:** "I saw the man on the hill with a telescope." This sentence can be interpreted in two ways:

1.  I saw the man who was on the hill, and I had a telescope.
2.  I saw the man who was on the hill with the telescope.

This ambiguity impacts applications such as machine translation. The correct parse is crucial for accurately conveying the intended meaning in the target language.

**Semantic Ambiguity Example:** "The cat sat on the mat, it was comfortable." The pronoun "it" could refer to either "cat" or "mat."  Determining the correct reference requires understanding the relationships between entities and the context of the sentence.  In a dialogue system, the interpretation affects which noun to use in subsequent sentences to maintain coherence.

## 3) Python method (if possible)

While there isn't a single "resolve ambiguity" function in Python, several libraries and techniques can be used to address different types of ambiguity.

```python
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn

# Lexical Disambiguation using Word Sense Disambiguation (WSD) with Lesk Algorithm
sentence1 = "I went to the bank to deposit money."
sentence2 = "I sat on the bank of the river."

# Tokenize the sentences
tokens1 = nltk.word_tokenize(sentence1)
tokens2 = nltk.word_tokenize(sentence2)

# Use Lesk algorithm to disambiguate "bank" in each sentence
bank_sense1 = lesk(tokens1, 'bank')
bank_sense2 = lesk(tokens2, 'bank')

print(f"Sense of 'bank' in '{sentence1}': {bank_sense1}")
print(f"Sense of 'bank' in '{sentence2}': {bank_sense2}")

#Accessing the meaning from wordnet
if bank_sense1:
  print(f"Definition of '{bank_sense1}': {bank_sense1.definition()}")

if bank_sense2:
  print(f"Definition of '{bank_sense2}': {bank_sense2.definition()}")

# Syntactic parsing (Illustrative, doesn't fully resolve ambiguity)
from nltk import CFG
from nltk import ChartParser

grammar = CFG.fromstring("""
    S -> NP VP
    VP -> V NP PP
    VP -> V NP
    NP -> Det N
    NP -> Det N PP
    PP -> P NP
    V -> 'saw'
    Det -> 'the' | 'a'
    N -> 'man' | 'hill' | 'telescope'
    P -> 'on' | 'with'
""")

parser = ChartParser(grammar)
sentence = "I saw the man on the hill with a telescope".split()
trees = list(parser.parse(sentence))
print(f"Number of possible parses: {len(trees)}")
for tree in trees:
    print(tree)
```

**Explanation:**

*   **Lexical Disambiguation (Lesk Algorithm):**  The `nltk.wsd.lesk` function attempts to determine the correct sense of a word (like "bank") based on the surrounding context (the other words in the sentence). It utilizes WordNet, a lexical database, to find the sense with the highest overlap in its definition with the surrounding words.  It's a simplified example, and more sophisticated WSD techniques exist.
*   **Syntactic Parsing:** The code demonstrates creating a simple context-free grammar (CFG) and using a chart parser to find possible parse trees for a syntactically ambiguous sentence. Note that the `nltk.ChartParser` is an example and that while it can find all the possible parses, it *doesn't* choose the "correct" one. Resolving which parse is correct requires incorporating semantic information or using more advanced parsing techniques.

This example showcases simple methods. Modern NLP utilizes deep learning models (e.g., BERT, Transformers) trained on massive datasets to implicitly learn to resolve these ambiguities.

## 4) Follow-up question

How do deep learning models, particularly transformers, handle ambiguity in natural language compared to the traditional methods demonstrated in the Python example (Lesk algorithm and CFG parsing)?  What are the limitations of deep learning approaches in resolving ambiguity, and are there situations where traditional methods might still be preferable?