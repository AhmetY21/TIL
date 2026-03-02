---
title: "Ambiguity in Natural Language: Lexical, Syntactic, Semantic"
date: "2026-03-02"
week: 10
lesson: 1
slug: "ambiguity-in-natural-language-lexical-syntactic-semantic"
---

# Topic: Ambiguity in Natural Language: Lexical, Syntactic, Semantic

## 1) Formal definition (what is it, and how can we use it?)

Ambiguity in natural language refers to the situation where a word, phrase, or sentence has multiple possible interpretations. This presents a significant challenge for natural language processing (NLP) systems, as these systems must correctly identify the intended meaning to perform tasks like machine translation, question answering, and text summarization effectively. There are three main types of ambiguity:

*   **Lexical Ambiguity:** Occurs when a single word has multiple meanings. This is also called semantic ambiguity. Examples include:
    *   "bank" can refer to a financial institution or the side of a river.
    *   "bat" can refer to a flying mammal or a piece of sporting equipment.

*   **Syntactic Ambiguity:** Occurs when a sentence can have multiple possible grammatical structures. This often arises due to the different ways words can be grouped and related to each other. Examples include:
    *   "I saw the man on the hill with a telescope." (Did I use the telescope, or was the man on the hill using it?)
    *   "Visiting relatives can be boring." (Is the act of visiting boring, or are the relatives boring?)

*   **Semantic Ambiguity:** This is a broader category closely related to Lexical Ambiguity but can also arise from the interaction of words and their meanings within a sentence. It occurs when the meaning of a sentence or phrase is unclear, often due to vague or undefined terms or relationships between them. This can be seen as an ambiguity stemming from how the words relate to each other to convey meaning.  While lexical ambiguity relates to individual words, semantic ambiguity looks at the ambiguity in the constructed meaning of the whole utterance. Examples include:
    *   "The pen is in the box." (Is it a writing pen, a pen for animals, or some other kind of pen?) While "pen" could have multiple meanings, the surrounding context will help narrow down possibilities. However, in other cases, the context is insufficient.
    *   "He ate the cake with ice cream." (Was the cake served with ice cream, or did he use ice cream as a tool to eat the cake?) This is closely related to syntactic ambiguity.

How can we use this understanding of ambiguity? By understanding the different types of ambiguity, we can design NLP systems that are better equipped to resolve them. This involves:

*   **Contextual analysis:** Examining the surrounding words and sentences to determine the most likely meaning.
*   **Part-of-speech (POS) tagging:** Identifying the grammatical role of each word in a sentence to help resolve syntactic ambiguity.
*   **Semantic analysis:** Using knowledge about the world and the relationships between concepts to determine the most plausible interpretation.
*   **Probabilistic models:** Assigning probabilities to different interpretations based on frequency and likelihood.
*   **Machine learning:** Training models on large datasets to learn patterns and relationships that can help resolve ambiguity.

## 2) Application scenario

Consider a machine translation system translating the sentence "I saw the man on the hill with a telescope" from English to another language.

*   **Without ambiguity resolution:** The system might blindly translate the sentence, potentially conveying the wrong meaning (e.g., translating as if *I* used the telescope when the intention was that the man on the hill had it).  This could lead to a nonsensical or incorrect translation in the target language.

*   **With ambiguity resolution:** The system would need to consider the possible syntactic structures and meanings. It might use contextual information from the surrounding text to determine the most likely interpretation. For instance, if the previous sentence mentioned that the man on the hill was an astronomer, it's more likely that *he* had the telescope.  The system might also use a dependency parser to analyze the grammatical relationships between the words and determine which interpretation is more likely based on common linguistic patterns. By correctly resolving the syntactic ambiguity, the machine translation system can produce a more accurate and meaningful translation in the target language.

Another example could be a search engine. If a user searches for "bank," the search engine needs to understand whether the user is interested in financial institutions or river banks. Understanding the context of the user's previous searches or their location can help resolve this lexical ambiguity and provide more relevant search results.

## 3) Python method (if possible)

While Python doesn't have a single built-in function to magically resolve all types of ambiguity, we can use libraries like NLTK (Natural Language Toolkit) and spaCy to perform tasks that aid in ambiguity resolution. Below is an example demonstrating how spaCy can be used for POS tagging and dependency parsing, which can help in identifying syntactic ambiguities. We will focus on identifying the possible grammatical relations:

```python
import spacy

nlp = spacy.load("en_core_web_sm") # Ensure you have this model: python -m spacy download en_core_web_sm

sentence = "I saw the man on the hill with a telescope"
doc = nlp(sentence)

for token in doc:
    print(token.text, token.pos_, token.dep_, token.head.text)

# This output will give you POS tags and dependency relations. You can analyze these
# to understand the possible syntactic structures. For instance, you'll see how "with a telescope"
# is attached to different parts of the sentence, leading to ambiguity. The visualization
# using displacy below helps demonstrate the dependency relations visually.

from spacy import displacy
displacy.serve(doc, style="dep") # This will display a visualization of the dependency tree in your browser.
```

Explanation:

1.  We load a spaCy model ("en\_core\_web\_sm"). This provides pre-trained models for various NLP tasks.
2.  We process the ambiguous sentence with the model.
3.  We iterate through each token (word) in the processed sentence and print its text, part-of-speech tag (POS), dependency relation (dep), and the text of its head (the word it depends on). The dependency relation and the head are key to identifying possible syntactic structures.
4.  We then use `displacy` to visualize the dependency parse tree, making it easier to see the possible grammatical relations. Running this code will open a web server showing a directed graph showing the relations between the words. You will need to use your browser to view the output.

While this doesn't *resolve* the ambiguity, it gives you the tools to analyze the possible syntactic structures, which is a crucial first step. More complex methods would involve statistical models trained on large datasets to choose the most likely parse.

## 4) Follow-up question

How can we leverage large language models (LLMs) like GPT-3 or BERT to improve ambiguity resolution in NLP tasks? Specifically, can you describe a method using prompt engineering to guide an LLM to identify and resolve ambiguities in a given sentence? Consider the sentence "The old man the boat."