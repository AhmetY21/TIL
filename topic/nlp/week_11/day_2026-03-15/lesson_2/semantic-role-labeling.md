---
title: "Semantic Role Labeling"
date: "2026-03-15"
week: 11
lesson: 2
slug: "semantic-role-labeling"
---

# Topic: Semantic Role Labeling

## 1) Formal definition (what is it, and how can we use it?)

Semantic Role Labeling (SRL) is a natural language processing (NLP) task that aims to identify the semantic roles of constituents in a sentence. In simpler terms, SRL seeks to answer the question of "who did what to whom, when, where, and why?" in a given sentence. It identifies predicates (usually verbs) and assigns semantic roles to the other constituents in the sentence, indicating their function in relation to the predicate.

**Formal definition:**

Given a sentence and a predicate (usually a verb), SRL identifies the semantic arguments of that predicate and classifies them into pre-defined semantic roles.  These roles describe the part each constituent plays in the event or state described by the predicate.

Common semantic roles include:

*   **ARG0 (Agent/Causer):** The entity performing the action.
*   **ARG1 (Patient/Theme):** The entity affected by the action.
*   **ARG2 (Instrument/Beneficiary):** An instrument used in the action or the recipient of the action.
*   **ARG3 (Beneficiary/Attribute):** Another beneficiary or an attribute of the ARG1.
*   **ARG4 (End State/Attribute):** The final state of the ARG1 or another attribute of the ARG1.
*   **ARG5 (Attribute):** Yet another attribute of the ARG1.
*   **ARGM-LOC (Location):** Where the action takes place.
*   **ARGM-TMP (Temporal):** When the action takes place.
*   **ARGM-MNR (Manner):** How the action is performed.
*   **ARGM-PRP (Purpose):** Why the action is performed.
*   **ARGM-ADV (Adverbial):** General adverbial modifier.
*   **ARGM-CAU (Cause):** The cause of the action.
*   **ARGM-DIR (Direction):** The direction of movement.

**How can we use it?**

SRL is useful for a variety of downstream NLP tasks, including:

*   **Information Extraction:** SRL helps in extracting structured information from unstructured text, making it easier to populate knowledge bases and databases.
*   **Question Answering:** By understanding the roles of entities in a sentence, question answering systems can better understand the query and retrieve relevant information.
*   **Text Summarization:** SRL can help identify the most important events and entities in a document, which can be used to create more informative summaries.
*   **Machine Translation:** SRL can help ensure that the semantic roles of entities are preserved during translation, leading to more accurate translations.
*   **Textual Entailment:** SRL can help determine whether one sentence entails another by comparing the semantic roles of entities in both sentences.

## 2) Application scenario

Consider the sentence: "John gave Mary the book yesterday in the library because she needed it for her research."

An SRL system would analyze this sentence and identify the predicate (the verb "gave") and its arguments. The SRL output might look something like this:

*   **Predicate:** gave
*   **ARG0 (Agent):** John
*   **ARG1 (Theme):** the book
*   **ARG2 (Beneficiary):** Mary
*   **ARGM-TMP (Temporal):** yesterday
*   **ARGM-LOC (Location):** in the library
*   **ARGM-PRP (Purpose):** because she needed it for her research

This structured representation of the sentence's meaning allows us to easily answer questions like:

*   Who gave something? (John)
*   What did John give? (the book)
*   To whom did John give the book? (Mary)
*   When did John give the book? (yesterday)
*   Where did John give the book? (in the library)
*   Why did John give the book? (because she needed it for her research)

This demonstrates how SRL can be used to extract meaningful information and answer complex questions about a sentence. This is useful in applications such as chatbot development (understanding user intents and extracting relevant information), document understanding, and knowledge base construction.

## 3) Python method (if possible)

While a complete SRL implementation from scratch is complex, you can use existing NLP libraries that provide SRL capabilities. One popular library is spaCy with the `llm-coref` extension which utilizes large language models (LLMs) to perform coreference resolution and semantic role labeling.  Other libraries, like AllenNLP, offer more fine-grained control but require more setup.

Here's a basic example using spaCy with the `llm-coref` extension:

```python
import spacy
from llm_coref.coreference import CoreferenceResolver
from llm_coref.semantic_role_labeling import SemanticRoleLabeler

# Load a spaCy model (you might need to download a larger model for better accuracy)
nlp = spacy.load("en_core_web_sm")  # or a larger model like "en_core_web_lg"

# Add Coreference Resolution pipe
coref = CoreferenceResolver(nlp)
nlp.add_pipe("coref", config={'model_name': 'google/flan-t5-base'}) # using google/flan-t5-base

# Add Semantic Role Labeling pipe
srl = SemanticRoleLabeler(nlp)
nlp.add_pipe("srl", config={'model_name': 'google/flan-t5-base'})

# Process the text
text = "John gave Mary the book yesterday in the library because she needed it for her research."
doc = nlp(text)

# Iterate through the sentences and their predicates/roles
for sentence in doc.sents:
    print(f"Sentence: {sentence.text}")
    for token in sentence:
        if token.pos_ == "VERB": # Only process verbs
            print(f"  Predicate: {token.text}")
            try:
                roles = token._.semantic_roles
                for role in roles:
                    print(f"    {role['role']}: {role['argument']}")
            except AttributeError:
                print("    No SRL roles found for this predicate.")
```

**Explanation:**

1.  **Import Libraries:** Imports `spacy`, `CoreferenceResolver`, and `SemanticRoleLabeler`
2.  **Load spaCy Model:** Loads a pre-trained spaCy model.  `en_core_web_sm` is a small model; you'll likely want a larger model (e.g., `en_core_web_lg`) for better accuracy.
3.  **Add Coreference Resolution Pipe:** Configures and adds the coreference resolution pipeline to the spaCy processing pipeline. The `model_name` parameter specifies the LLM to use for coreference resolution.
4.  **Add Semantic Role Labeling Pipe:** Configures and adds the semantic role labeling pipeline to the spaCy processing pipeline.  The `model_name` parameter specifies the LLM to use for SRL.
5.  **Process Text:**  Processes the input text using the spaCy pipeline.
6.  **Iterate and Print Results:** Iterates through the tokens in each sentence, identifies the verb, and extracts the semantic roles associated with that verb using `token._.semantic_roles`. The role name and the corresponding argument are printed. Error handling is included to gracefully handle cases where no SRL roles are found for a specific predicate.

**Note:**

*   You may need to install the necessary libraries: `pip install spacy llm-coref`
*   Download a suitable spaCy model: `python -m spacy download en_core_web_sm` (or `en_core_web_lg` or similar).
* The performance and accuracy will vary depending on the spaCy model and the complexity of the sentence. Larger models tend to perform better, but require more computational resources.
*   The `llm-coref` library relies on LLMs so may have some setup with API keys.
*  Remember to handle potential errors (e.g., if the SRL component fails to identify roles).
*  Consider other libraries, like AllenNLP, for more advanced SRL tasks. AllenNLP offers pre-trained models specifically for SRL. However, it has a steeper learning curve.

## 4) Follow-up question

How does SRL deal with sentences that have multiple predicates or embedded clauses?  Does it handle them independently, or does it consider the relationships between the predicates and their arguments across different parts of the sentence? What are some of the challenges in handling complex sentence structures for SRL?