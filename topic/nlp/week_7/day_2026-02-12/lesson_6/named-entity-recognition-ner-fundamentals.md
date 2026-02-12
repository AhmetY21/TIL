---
title: "Named Entity Recognition (NER) Fundamentals"
date: "2026-02-12"
week: 7
lesson: 6
slug: "named-entity-recognition-ner-fundamentals"
---

# Topic: Named Entity Recognition (NER) Fundamentals

## 1) Formal definition (what is it, and how can we use it?)

Named Entity Recognition (NER), also known as entity identification, entity chunking, and entity extraction, is a subtask of information extraction that seeks to locate and classify *named entities* in text into pre-defined categories. These categories can include person names, organizations, locations, dates, quantities, monetary values, percentages, etc.

Formally, NER can be viewed as a sequence labeling problem. Given a sequence of tokens (words) in a sentence, the task is to assign a label from a predefined set of categories to each token, or mark the token as not belonging to any named entity.

We can use NER for a wide range of applications:

*   **Information Retrieval/Search:** NER helps to index documents more effectively, allowing users to search for specific entities like "Apple Inc." instead of just "apple".

*   **Question Answering:**  NER can be used to identify the entities mentioned in a question, which helps to narrow down the scope of possible answers. For example, in the question "Who is the CEO of Microsoft?", NER would identify "Microsoft" as an organization.

*   **News Summarization:**  NER can extract the key entities from a news article, such as people, organizations, and locations, to create a concise summary.

*   **Customer Support:** Analyze customer feedback or chats to identify recurring issues related to specific products, services, or locations.

*   **Content Recommendation:** Suggest related content based on the entities mentioned in a piece of text the user is reading. If an article mentions "Barack Obama" and "United States," the system could recommend other articles related to these entities.

## 2) Application scenario

Consider the following sentence:

"Apple is planning to open a new store in London next year, according to Tim Cook."

Applying NER to this sentence would result in the following entities being identified and classified:

*   **Apple:** ORGANIZATION
*   **London:** GPE (Geo-Political Entity, often a city or country)
*   **next year:** DATE
*   **Tim Cook:** PERSON

This extracted information can then be used for various downstream tasks. For example, a news aggregator might use it to categorize the article under "Business", "Technology", and "Europe" or create a "Company expansion in Europe" news feed. A search engine could use it to improve search results when someone searches for "Apple London store opening".

## 3) Python method (if possible)

We can use the `spaCy` library in Python, which is widely used for NLP tasks, including NER.

```python
import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")  # Use a smaller model for demonstration

text = "Apple is planning to open a new store in London next year, according to Tim Cook."

# Process the text with spaCy
doc = nlp(text)

# Iterate through the entities and print their text and label
for ent in doc.ents:
    print(ent.text, ent.label_)

# You can also iterate through the tokens and check if they're part of an entity
for token in doc:
    if token.ent_type_:  # Check if the token has an entity type assigned
        print(token.text, token.ent_type_)
```

This code snippet first loads a spaCy English language model. Then, it processes the input text and iterates through the identified entities, printing both the entity text and its label. The code also shows how to iterate through the tokens and only print tokens which are part of an entity.  The `en_core_web_sm` model is a smaller, faster model. For more accurate results, consider using a larger model like `en_core_web_lg`.

## 4) Follow-up question

What are some common challenges and limitations associated with Named Entity Recognition, especially in scenarios involving ambiguity or context-dependent entity types? How can these challenges be addressed?