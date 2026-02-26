---
title: "Semantic Role Labeling"
date: "2026-02-26"
week: 9
lesson: 2
slug: "semantic-role-labeling"
---

# Topic: Semantic Role Labeling

## 1) Formal definition (what is it, and how can we use it?)

Semantic Role Labeling (SRL), also known as shallow semantic parsing, is the process of identifying and labeling the semantic roles of the constituents in a sentence with respect to a predicate. In simpler terms, SRL answers the question "Who did what to whom, where, when, and why?" for a given sentence.

More formally:

*   **Predicate:** A verb or noun that acts as the central event or state being described in the sentence.
*   **Arguments (or Semantic Roles):** The phrases that participate in the event or state denoted by the predicate. Common roles include:
    *   **Agent (A0):** The initiator of the action.
    *   **Patient (A1):** The entity that is acted upon.
    *   **Instrument (A2):** The means by which the action is performed.
    *   **Location (A3):** The place where the action occurs.
    *   **Goal (A4):** The endpoint or beneficiary of the action.
    *   **Temporal (AM-TMP):** The time of the action.
    *   **Locative (AM-LOC):** The location of the action.
    *   **Manner (AM-MNR):** The manner in which the action is performed.
    *   **Cause (AM-CAU):** The cause of the action.

SRL aims to assign these roles to the phrases in a sentence, given a particular predicate. The 'AM-' prefixes usually denote adjuncts or modifiers, providing additional information about the action.

**How can we use it?**

SRL has numerous applications:

*   **Information Extraction:** SRL can extract structured information from unstructured text, enabling automated knowledge base construction and population.
*   **Question Answering:** SRL helps understand the relationships between entities in a sentence, allowing for more accurate question answering.  For example, understanding "Who gave the book to Mary?" requires identifying the agent (giver), the patient (book), and the recipient (Mary).
*   **Text Summarization:** SRL can identify the most important events and entities in a text, which can be used to create more coherent and informative summaries.
*   **Machine Translation:** SRL can improve the accuracy of machine translation by ensuring that the semantic roles of constituents are preserved across languages.
*   **Textual Entailment:** SRL can determine whether one sentence logically follows from another by analyzing the semantic relationships expressed in each sentence.
*   **Dialogue Systems:**  SRL aids in understanding user utterances and constructing meaningful responses.

## 2) Application scenario

Consider the sentence: "John gave the book to Mary yesterday in the library."

An SRL system would identify "gave" as the predicate and then label the following arguments:

*   **John:** Agent (A0) - The one doing the giving
*   **the book:** Patient (A1) - The thing being given
*   **to Mary:** Goal (A4) - The recipient of the book
*   **yesterday:** Temporal (AM-TMP) - When the giving occurred
*   **in the library:** Locative (AM-LOC) - Where the giving occurred

This structured representation of the sentence's meaning allows a computer system to understand the relationships between the entities involved and answer questions like: "Who gave the book?", "What was given?", "Who received the book?", "When did the event happen?", and "Where did the event happen?".

## 3) Python method (if possible)

While no single "perfect" Python library handles all aspects of SRL from scratch, you can leverage libraries that build upon existing NLP tools and pre-trained models.  One popular and relatively straightforward approach uses the `allennlp` library, which leverages pre-trained models for various NLP tasks, including SRL.

```python
from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction

# Load the SRL model.  This downloads the model on the first run, so it might take a while.
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")

sentence = "John gave the book to Mary yesterday in the library."

# Predict the semantic roles.
prediction = predictor.predict(sentence=sentence)

# Print the results.
words = prediction['words']
verbs = prediction['verbs']
verb_tags = prediction['verb_tags']

print(f"Sentence: {' '.join(words)}\n")

for i, verb in enumerate(verbs):
    print(f"Predicate: {verb}")
    tags = verb_tags[i]
    for j, tag in enumerate(tags):
        if tag != 'O': # O means "outside" any argument
            print(f"  {words[j]}: {tag}")
    print("\n")

# Example output (may vary slightly due to model versions):
# Sentence: John gave the book to Mary yesterday in the library .
#
# Predicate: gave
#   John: B-ARG0
#   gave: B-V
#   the: B-ARG1
#   book: I-ARG1
#   to: B-ARG4
#   Mary: I-ARG4
#   yesterday: B-AM-TMP
#   in: B-AM-LOC
#   the: I-AM-LOC
#   library: I-AM-LOC
#   .: O
```

**Explanation:**

1.  **Import necessary libraries:** `Predictor` from `allennlp.predictors.predictor` and `allennlp_models.structured_prediction` to make sure the relevant models are installed.
2.  **Load the pre-trained SRL model:** `Predictor.from_path()` loads a pre-trained SRL model from AllenNLP. The URL points to a BERT-based SRL model. This step downloads the model if it's not already cached.
3.  **Define the sentence:** The sentence to be analyzed is defined.
4.  **Predict the semantic roles:** `predictor.predict(sentence=sentence)` performs SRL on the sentence and returns a dictionary containing the results.
5.  **Print the results:** The code iterates through the words and their corresponding semantic role tags, printing the predicate and its arguments.  The tags indicate the semantic role assigned to each word. `B-` indicates the beginning of an argument, `I-` indicates a continuation of an argument, and `O` indicates that the word is not part of any argument.  ARG0, ARG1, ARG4, AM-TMP, and AM-LOC correspond to Agent, Patient, Goal, Temporal, and Locative modifiers, respectively, as discussed in section 1.

This example demonstrates how to use AllenNLP to perform SRL. Other libraries like `spaCy` (using extensions or custom models) can also be used, but AllenNLP offers a dedicated pre-trained SRL model that is generally more accurate out-of-the-box.

## 4) Follow-up question

How does the performance of SRL systems vary depending on the complexity of the sentence structure, and what techniques are used to improve SRL accuracy on challenging sentences like those with nested clauses or complex verb phrases?