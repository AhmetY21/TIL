---
title: "Named Entity Recognition (NER) Fundamentals"
date: "2026-03-03"
week: 10
lesson: 5
slug: "named-entity-recognition-ner-fundamentals"
---

# Topic: Named Entity Recognition (NER) Fundamentals

## 1) Formal definition (what is it, and how can we use it?)

Named Entity Recognition (NER), also known as entity identification, entity chunking, and entity extraction, is a subtask of information extraction that seeks to locate and classify named entities mentioned in unstructured text into pre-defined categories such as person names, organizations, locations, dates, quantities, monetary values, percentages, etc.

**What it is:** Essentially, NER is a process that involves:

*   **Identification:** Pinpointing spans of text that represent entities.
*   **Classification:** Categorizing those entities into predefined types.

**How we can use it:** NER is a crucial component in many NLP applications, including:

*   **Information Retrieval:** Improving search results by allowing users to search for specific types of entities. For example, searching for news articles about "Apple" specifically referencing the *company* and not the fruit.
*   **Question Answering:** Identifying key entities in the question to help find relevant answers. For example, in the question "Who is the CEO of Microsoft?", NER would identify "Microsoft" as an organization.
*   **Text Summarization:** Identifying important entities to include in a summary.
*   **Knowledge Graph Construction:** Extracting entities and relationships between them to build knowledge graphs.
*   **Customer Service:** Routing customer inquiries to the appropriate department based on the entities mentioned in their message (e.g., routing inquiries about "billing" to the billing department).
*   **Fake News Detection:** Analyzing entities and their relationships to assess the credibility of news articles. Discrepancies or unusual entity associations might be indicators of misinformation.

## 2) Application scenario

Let's consider a news article about a business acquisition:

"Apple is planning to acquire Shazam for $400 million. The deal is expected to close by December 2017.  Shazam is a British company headquartered in London."

Using NER, we can extract the following entities:

*   **Apple:** ORGANIZATION
*   **Shazam:** ORGANIZATION
*   **$400 million:** MONEY
*   **December 2017:** DATE
*   **British:** NORP (Nationality or religious or political groups)
*   **London:** GPE (Geopolitical entity)

This extracted information can then be used for various purposes, such as populating a database of corporate acquisitions, identifying the companies involved, the deal amount, and the timeline.  It can also be used to perform sentiment analysis related to the acquisition. Was the market reaction positive or negative, based on articles mentioning "Apple" and "Shazam"?

## 3) Python method (if possible)

We can use the `spaCy` library for NER in Python.  Here's a simple example:

```python
import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")  # Or a larger more accurate model like "en_core_web_lg"

text = "Apple is planning to acquire Shazam for $400 million. The deal is expected to close by December 2017. Shazam is a British company headquartered in London."

# Process the text with spaCy
doc = nlp(text)

# Iterate over the entities and print their text and label
for ent in doc.ents:
    print(ent.text, ent.label_)
```

This code will output:

```
Apple ORG
Shazam ORG
$400 million MONEY
December 2017 DATE
Shazam ORG
British NORP
London GPE
```

**Explanation:**

*   We import the `spacy` library.
*   We load a pre-trained English language model.  `en_core_web_sm` is a smaller, faster model. `en_core_web_lg` is a larger model that typically provides more accurate results. You might need to download the model first using `python -m spacy download en_core_web_sm` or `python -m spacy download en_core_web_lg`.
*   We define the text we want to analyze.
*   We process the text using the `nlp` object, which creates a `Doc` object.
*   We iterate through the `ents` attribute of the `Doc` object, which contains all the recognized entities.
*   For each entity, we print its text and its label (entity type).

## 4) Follow-up question

How does the choice of a pre-trained model (e.g., `en_core_web_sm` vs. `en_core_web_lg` in spaCy) affect the accuracy and performance of NER, and what factors should I consider when selecting a model for a specific task?  Furthermore, how can NER be improved for languages other than English or for specialized domains (e.g., medical or legal text) where existing pre-trained models may be insufficient?