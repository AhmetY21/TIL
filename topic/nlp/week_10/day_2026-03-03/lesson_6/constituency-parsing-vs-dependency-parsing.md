---
title: "Constituency Parsing vs Dependency Parsing"
date: "2026-03-03"
week: 10
lesson: 6
slug: "constituency-parsing-vs-dependency-parsing"
---

# Topic: Constituency Parsing vs Dependency Parsing

## 1) Formal definition (what is it, and how can we use it?)

**Constituency Parsing:**

*   **What it is:** Constituency parsing, also known as phrase structure parsing, aims to represent the syntactic structure of a sentence according to phrase structure grammar. It breaks down a sentence into its constituent parts, like noun phrases (NP), verb phrases (VP), and prepositional phrases (PP), and then further decomposes these phrases into smaller constituents, forming a tree-like structure. The root node of the tree represents the entire sentence (S), and the leaves represent the individual words. The internal nodes represent the phrase types.

*   **How can we use it:** Constituency parsing is useful for understanding the hierarchical organization of a sentence. It helps in tasks such as:
    *   **Semantic interpretation:** Understanding the relationships between different parts of a sentence, which aids in extracting meaning.
    *   **Question answering:** Identifying the subject, object, and verb phrases to better understand the question and find the answer.
    *   **Machine translation:** Improving the accuracy of translations by understanding the grammatical structure of the source language sentence.
    *   **Text summarization:** Identifying important phrases and clauses in a document.
    *   **Grammar checking:** Validating the grammatical correctness of a sentence.

**Dependency Parsing:**

*   **What it is:** Dependency parsing represents the syntactic structure of a sentence by establishing dependencies between words. It identifies the head word (governor) and the dependent words (modifiers) for each word in the sentence, forming a directed graph. The edges in the graph are labeled with dependency relations (e.g., subject, object, adjective modifier). The root of the graph is usually the main verb of the sentence. Unlike constituency parsing, dependency parsing focuses on the relations between words rather than building phrase structures.

*   **How can we use it:** Dependency parsing is useful for understanding the grammatical relationships between words. It helps in tasks such as:
    *   **Information extraction:** Identifying relationships between entities in a sentence, such as who did what to whom.
    *   **Semantic role labeling:** Identifying the semantic roles of words in a sentence, such as agent, patient, and instrument.
    *   **Machine translation:** Improving the accuracy of translations by understanding the dependencies between words in the source language sentence.
    *   **Sentiment analysis:** Understanding the modifiers and dependencies affecting the sentiment of a sentence.
    *   **Coreference resolution:** Identifying which words or phrases refer to the same entity.

## 2) Application scenario

**Constituency Parsing Scenario:**

Imagine you are building a grammar checker. You want to determine if a sentence is grammatically correct. Constituency parsing can help by breaking the sentence down into its phrase structure. If the resulting parse tree doesn't conform to your grammar rules, you know there's a grammatical error. For example, if the parser finds an NP with a missing determiner when the grammar requires one, you can flag that as an error.

**Dependency Parsing Scenario:**

Consider a scenario where you're extracting information about relationships between people and organizations from news articles. For example, you want to find instances of "CEO of Company". Dependency parsing can identify the relationships between words like "CEO" and "Company". It would identify "CEO" as the head and "Company" as a dependent with a dependency relation like "of" or "poss". This allows you to extract structured information about these relationships.

## 3) Python method (if possible)

```python
import spacy
from nltk.tree import Tree # For visualization of constituency parse trees (if using NLTK parser)

# Option 1: Using spaCy for Dependency Parsing
def dependency_parse_spacy(sentence):
    nlp = spacy.load("en_core_web_sm") # load small english model
    doc = nlp(sentence)
    for token in doc:
        print(f"{token.text} -- {token.dep_} --> {token.head.text}")

    # Visualize the dependency parse using displacy
    from spacy import displacy
    displacy.render(doc, style="dep", jupyter=True)


# Option 2: Using NLTK with a pre-trained constituency parser (e.g., Stanford Parser or Berkeley Parser) - Requires additional setup
#  Note: The following commented code requires the Stanford Parser or Berkeley Parser to be downloaded and configured.
# from nltk.parse.stanford import StanfordParser
#
# def constituency_parse_nltk(sentence):
#     # Replace with the path to your Stanford Parser or Berkeley Parser
#     # Example for Stanford Parser:
#     # parser = StanfordParser(path_to_jar='/path/to/stanford-parser.jar',
#     #                         path_to_models_jar='/path/to/stanford-parser-models.jar')
#
#     # Need to load parser and perform the parse.  I do not have a parse to include here since it requires external configuration.
#     # parsed_tree = list(parser.raw_parse(sentence))[0]
#     # parsed_tree.draw()  # Visualizes the tree in a separate window (if you have tkinter installed)
#     # print(parsed_tree)  # Print the tree in text format
#     print("NLTK Constituency Parsing requires Stanford Parser or Berkeley Parser setup.  See comments in the code.")




# Example usage for spaCy dependency parsing:
sentence = "The quick brown fox jumps over the lazy dog."
dependency_parse_spacy(sentence)


# Example of a possible NLTK Constituency Parse outcome (assuming setup is complete and parser is invoked as explained above):
# (S
#   (NP (DT The) (JJ quick) (JJ brown) (NN fox))
#   (VP (VBZ jumps)
#     (PP (IN over)
#       (NP (DT the) (JJ lazy) (NN dog))))
#   (. .))

# Example usage for NLTK constituency parsing: (after proper setup!)
#sentence = "The quick brown fox jumps over the lazy dog."
#constituency_parse_nltk(sentence)
```

## 4) Follow-up question

What are the relative strengths and weaknesses of Constituency Parsing and Dependency Parsing in the context of processing languages like Chinese, which have relatively free word order and different grammatical structures compared to English?