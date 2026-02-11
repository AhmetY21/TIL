---
title: "Ambiguity in Natural Language: Lexical, Syntactic, Semantic"
date: "2026-02-11"
week: 7
lesson: 1
slug: "ambiguity-in-natural-language-lexical-syntactic-semantic"
---

# Topic: Ambiguity in Natural Language: Lexical, Syntactic, Semantic

## 1) Formal definition (what is it, and how can we use it?)

Ambiguity in Natural Language Processing (NLP) refers to the uncertainty in the meaning of a linguistic unit (word, phrase, sentence, etc.). A single linguistic unit can have multiple interpretations, making it challenging for machines (and sometimes even humans!) to understand the intended meaning. Recognizing and resolving ambiguity is crucial for NLP tasks like machine translation, information retrieval, and text summarization.  We can use an understanding of ambiguity types to design algorithms that employ techniques such as context analysis, part-of-speech tagging, and semantic role labeling to disambiguate text. Specifically, let's define the three types:

*   **Lexical Ambiguity:** A word has multiple meanings. This occurs when a single word form (spelling and pronunciation) represents different concepts or senses. For example, the word "bank" can refer to a financial institution or the edge of a river.

*   **Syntactic Ambiguity:** The structure of a sentence is ambiguous, leading to multiple possible interpretations. This occurs when a sentence can be parsed in different ways, each yielding a different meaning.  For example, "I saw the man on the hill with a telescope" could mean I used a telescope to see the man, or the man on the hill had a telescope.

*   **Semantic Ambiguity:** The meaning of the sentence as a whole is ambiguous even when the individual words and the grammatical structure are clear. This often arises from the interaction of words and their relationships. A common example includes quantifier scope ambiguities (e.g., "Every student read a book" – did each student read a potentially different book, or did they all read the same book?).

## 2) Application scenario

Let's consider a machine translation scenario. Suppose we want to translate the English sentence "I saw her duck" into French.

*   **Lexical Ambiguity:** The word "duck" can be a noun (the bird) or a verb (to lower the head or body quickly to avoid being hit).
*   **Syntactic Ambiguity:** "Her duck" could mean I saw her pet duck (noun phrase), or I saw her lower her head (verb phrase).

If the machine translation system doesn't resolve these ambiguities, it might incorrectly translate "duck" as *canard* (the bird) when it should be translated as *se baisser* (to duck). The system needs to consider the context of the sentence to determine the correct meaning and produce an accurate translation. For example, if a previous sentence mentioned "the pond", it would be more likely that "duck" refers to the bird. If the previous sentence mentioned "a flying object", it would be more likely that "duck" means to lower the body. Therefore, a real world application might use surrounding sentences or paragraphs to provide the needed context and avoid incorrect interpretation of an ambiguous word or sentence.

## 3) Python method (if possible)

While directly "solving" ambiguity in a single, simple Python function isn't possible (it usually requires comprehensive NLP pipelines), we can demonstrate how to identify potential lexical ambiguities using the WordNet lexical database through the `nltk` library. This shows one basic component used in resolving ambiguity.

```python
import nltk
from nltk.corpus import wordnet

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


def show_synsets(word):
    """
    Demonstrates potential lexical ambiguity by showing the synsets (sets of synonyms)
    associated with a word.
    """
    synsets = wordnet.synsets(word)
    if synsets:
        print(f"Synsets for '{word}':")
        for synset in synsets:
            print(f"- {synset.name()}: {synset.definition()}")
    else:
        print(f"No synsets found for '{word}'.")


# Example usage
show_synsets("bank")
show_synsets("rock")
```

This code snippet downloads the WordNet database if you don't have it already, then defines a function `show_synsets` that takes a word as input and prints its synsets, along with their definitions, from WordNet. This illustrates the multiple meanings associated with potentially ambiguous words.

Note: This is a simplified example. Actual ambiguity resolution requires more sophisticated techniques such as part-of-speech tagging, dependency parsing, and semantic analysis.

## 4) Follow-up question

How do more advanced techniques like transformer models (e.g., BERT, RoBERTa) handle ambiguity compared to traditional NLP methods (like the one shown in the Python example)? How can they be fine-tuned specifically to improve ambiguity resolution in a particular domain or application?