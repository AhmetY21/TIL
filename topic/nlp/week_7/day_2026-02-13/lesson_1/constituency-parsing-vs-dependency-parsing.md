---
title: "Constituency Parsing vs Dependency Parsing"
date: "2026-02-13"
week: 7
lesson: 1
slug: "constituency-parsing-vs-dependency-parsing"
---

# Topic: Constituency Parsing vs Dependency Parsing

## 1) Formal definition (what is it, and how can we use it?)

Both constituency parsing and dependency parsing are methods used in Natural Language Processing (NLP) to analyze the syntactic structure of sentences. They aim to represent the relationships between words and phrases in a sentence, but they do so in different ways:

*   **Constituency Parsing (Phrase Structure Parsing):**
    *   **What is it?** Constituency parsing breaks down a sentence into its constituent parts (phrases, clauses). It represents the grammatical structure using a tree-like structure called a *parse tree* or *phrase structure tree*. The tree shows how words group together to form phrases, and how these phrases group together to form larger phrases and ultimately the entire sentence. Each node in the tree represents a constituent (e.g., noun phrase (NP), verb phrase (VP), prepositional phrase (PP)). It follows formal grammar rules (often context-free grammars, CFGs).
    *   **How can we use it?** Constituency parsing helps understand the hierarchical structure of a sentence. This is useful for:
        *   **Grammar checking:** Identifying grammatical errors by verifying if the sentence conforms to the defined grammar rules.
        *   **Machine translation:** Understanding the structure of the source sentence to generate a grammatically correct translation.
        *   **Information extraction:** Identifying specific phrases or entities within a text based on their syntactic role.
        *   **Question answering:** Understanding the structure of the question to better identify potential answers in a text corpus.
        *   **Semantic analysis:** Linking syntactic structure to semantic meaning.

*   **Dependency Parsing:**
    *   **What is it?** Dependency parsing focuses on the relationships between individual words in a sentence. It represents the syntactic structure using a *dependency graph* or *dependency tree*. Each word is a node in the graph, and the edges represent *dependencies* between words.  The dependency relation is often labeled with grammatical roles like subject (SUBJ), object (OBJ), modifier (MOD), etc. The root of the tree is usually the main verb of the sentence.
    *   **How can we use it?** Dependency parsing is helpful for understanding the functional relationships between words in a sentence. This is useful for:
        *   **Relation extraction:** Identifying relationships between entities mentioned in the text by analyzing the dependency paths between them.
        *   **Semantic role labeling:** Assigning semantic roles (e.g., Agent, Patient, Instrument) to words in a sentence based on their dependencies.
        *   **Machine translation:** Using the dependency structure to align words and phrases across languages.
        *   **Information retrieval:** Improving search results by understanding the relationships between keywords in a query.
        *   **Text summarization:** Identifying the most important dependencies to extract the core meaning of the text.

**Key Differences Summarized:**

| Feature        | Constituency Parsing                 | Dependency Parsing                     |
|----------------|--------------------------------------|----------------------------------------|
| Representation | Phrase structure tree                | Dependency graph/tree                  |
| Focus          | Grouping words into constituents     | Relationships between individual words  |
| Grammar        | Formal grammar rules (e.g., CFGs)   | Dependency relations                    |
| Root Node      | S (Sentence)                         | Main verb                               |

## 2) Application scenario

*   **Constituency Parsing Scenario:**  Imagine building a sophisticated grammar checker. You want to ensure sentences adhere to complex grammatical rules.  You would use constituency parsing to build a parse tree, representing the sentence's phrase structure. Then, your program can traverse the tree, validating each node (phrase) against predefined grammar rules. For example, checking that a noun phrase (NP) contains at least a noun, and that a verb phrase (VP) has a verb.

*   **Dependency Parsing Scenario:** Consider a task where you want to extract biographical information from text, specifically, identifying "born in" relationships. Dependency parsing can help you find sentences where a person's name is the subject, "born" is the verb, and a location is linked to "born" via a prepositional modifier (e.g., "in"). The dependency parser would directly expose the relationships:  `Person <-SUBJ- born -MOD-> Location`.  This dependency structure helps to efficiently identify the desired relationships.

## 3) Python method (if possible)

We can use the `spaCy` and `NLTK` libraries in Python to perform both constituency and dependency parsing, though NLTK often relies on underlying parsers like the Stanford Parser for constituency.

```python
import spacy
from nltk.tree import Tree
# Need to download the model first.
# python -m spacy download en_core_web_sm

# Dependency Parsing using spaCy
nlp = spacy.load("en_core_web_sm")
doc = nlp("The quick brown fox jumps over the lazy dog.")

print("Dependency Parsing (spaCy):")
for token in doc:
    print(f"{token.text} --({token.dep_})--> {token.head.text}")

# Constituency Parsing using NLTK (requires a constituency parser model trained separately).  Using a very basic example as training a complete parser is beyond the scope here.
import nltk

# Create a simple grammar
grammar = nltk.CFG.fromstring("""
    S -> NP VP
    NP -> Det N
    VP -> V
    Det -> 'The'
    N -> 'dog'
    V -> 'barks'
""")

# Create a parser
parser = nltk.ChartParser(grammar)

# Parse a sentence
sentence = "The dog barks".split()
trees = list(parser.parse(sentence))

# Print the parse tree
print("\nConstituency Parsing (NLTK):")
if trees:
    for tree in trees:
        print(tree)
else:
    print("No parse tree found.")


# Visualize spacy dependency parse.
from spacy import displacy
displacy.serve(doc, style="dep") # view in web browser at http://127.0.0.1:5000

```

**Explanation:**

*   **spaCy (Dependency Parsing):** The `spaCy` code loads a pre-trained language model and processes a sentence.  It then iterates through each token (word) and prints the word, its dependency relation (`token.dep_`), and the head word it depends on (`token.head.text`).  The `displacy` library provides a visualization of the dependency tree in a web browser.
*   **NLTK (Constituency Parsing):** The `NLTK` code defines a simple context-free grammar (CFG) and uses it to parse a very simple sentence. It is just an example. A real-world constituency parser is much more complex and would require a large, pre-trained grammar and parser. You'd typically use the Stanford Parser or Berkeley Parser through NLTK, which require more setup and pre-trained models. The `nltk.ChartParser` is a basic parser for demonstration. The code prints the generated parse tree. If no parse tree is found based on the given grammar, it reports that.

**Important Notes:**

*   The NLTK example is very basic. For realistic constituency parsing, you would need to use a pre-trained parser like the Stanford Parser or Berkeley Parser and a more comprehensive grammar.  These often involve downloading additional JAR files and models.
*   Dependency parsing is often easier to work with in practice, especially when using pre-trained models like those provided by spaCy, because it directly provides the relationships between words without requiring complex grammar definitions.
*   The spaCy visualization requires a web server. It starts one up and serves the dependency graph to your browser.

## 4) Follow-up question

Given a specific task, such as building a question-answering system, how would you decide whether to use constituency parsing, dependency parsing, or a combination of both, and what are the key factors that would influence this decision?