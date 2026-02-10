---
title: "NLP vs NLU vs NLG: Understanding the Differences"
date: "2026-02-10"
week: 7
lesson: 6
slug: "nlp-vs-nlu-vs-nlg-understanding-the-differences"
---

# Topic: NLP vs NLU vs NLG: Understanding the Differences

## 1) Formal definition (what is it, and how can we use it?)

*   **NLP (Natural Language Processing):** This is the overarching field encompassing the ability of computers to understand, interpret, and generate human language. It is a broad area that involves a range of techniques from statistical methods to deep learning. Think of it as the parent category. NLP enables machines to perform tasks like translation, text summarization, sentiment analysis, and question answering. We use NLP to allow machines to work with text and speech data in meaningful ways.

*   **NLU (Natural Language Understanding):** This is a subfield of NLP focusing specifically on *understanding* the meaning and intent behind human language. It's about interpreting the semantics, identifying entities, and extracting relationships within the text. NLU aims to convert raw text into a structured, machine-readable representation. We use NLU to determine what a user means when they say something, going beyond just recognizing the words themselves. For example, "Book a flight to London" is interpreted as the *intent* to book a flight, and "London" is identified as the *destination* entity.

*   **NLG (Natural Language Generation):** This is another subfield of NLP, focusing on the reverse process: taking structured data or information and turning it into natural language text that humans can understand. NLG aims to generate coherent, grammatically correct, and contextually appropriate text. We use NLG to automatically create reports, summaries, product descriptions, or even hold conversations with users.

In short:
* NLP = The big picture.
* NLU = Understanding what humans say.
* NLG = Writing like a human.

## 2) Application scenario

Let's consider a chatbot application designed to help users book flights:

*   **NLU:** When a user types "I want a flight from New York to Los Angeles next Tuesday," the NLU component is responsible for understanding the user's *intent* (booking a flight) and extracting the relevant *entities* (origin city: New York, destination city: Los Angeles, date: next Tuesday). The NLU system must correctly interpret these entities even if the user expresses the request in different ways ("Fly me from NYC to LA on Tuesday," etc.).

*   **NLP (including NLU):**  The entire flight booking system requires NLP at various stages. It might use NLP for sentiment analysis on customer reviews of airlines, or for topic modelling to understand common customer complaints related to flight delays.  NLP provides the tools and techniques that enables the full functionality of the bot.

*   **NLG:** After processing the user's request and finding available flights, the NLG component generates a response like: "I found 3 flights from New York to Los Angeles on Tuesday: Flight 1 departs at 8:00 AM for $300, Flight 2 departs at 12:00 PM for $350, Flight 3 departs at 4:00 PM for $400. Which flight would you like to book?". This generated text needs to be grammatically correct, informative, and tailored to the user's initial request.

## 3) Python method (if possible)

While there isn't a single Python method that embodies *all* of NLP, NLU, or NLG, here's an example using the `spaCy` library for NLU, showcasing entity recognition:

```python
import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

text = "I want to book a flight from New York to Los Angeles next Tuesday."

# Process the text with spaCy
doc = nlp(text)

# Extract entities
for ent in doc.ents:
    print(f"Entity: {ent.text}, Label: {ent.label_}")

# Code to generate a canned response, though basic it shows generation:
origin_city = ""
destination_city = ""
date = ""

for ent in doc.ents:
    if ent.label_ == "GPE" and origin_city == "":
        origin_city = ent.text
    elif ent.label_ == "GPE":
        destination_city = ent.text
    elif ent.label_ == "DATE":
        date = ent.text

if origin_city and destination_city and date:
    print(f"Okay, searching for flights from {origin_city} to {destination_city} on {date}.")
else:
    print("Could not understand the cities and date provided. Please try again")
```

**Explanation:**

1.  We load a pre-trained spaCy language model.  `en_core_web_sm` is a small English model.
2.  We process the input text using `nlp(text)`, which returns a `Doc` object.
3.  We iterate through the detected entities using `doc.ents`.
4.  For each entity, we print its text and its label (e.g., "New York" is a `GPE` - geopolitical entity).
5.  The code extracts the entities and assigns them to variables.
6.  The code then uses these variables to generate a simple response, demonstrating a basic NLG aspect by creating a text output based on structured information.

For more sophisticated NLG, libraries like `transformers` (specifically, models like GPT-2 or BART) are commonly used.  NLU is also enhanced with deep learning models using `transformers`.

## 4) Follow-up question

Given that NLU focuses on understanding intent, and NLG focuses on generating text, how does the evaluation of NLU systems differ from the evaluation of NLG systems? What metrics are typically used for each?