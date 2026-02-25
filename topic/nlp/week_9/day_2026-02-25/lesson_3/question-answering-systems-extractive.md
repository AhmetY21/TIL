---
title: "Question Answering Systems (Extractive)"
date: "2026-02-25"
week: 9
lesson: 3
slug: "question-answering-systems-extractive"
---

# Topic: Question Answering Systems (Extractive)

## 1) Formal definition (what is it, and how can we use it?)

Extractive Question Answering (QA) systems are a type of NLP system designed to answer questions by *extracting* a span of text directly from a provided context document. Unlike abstractive QA systems, which generate new answers based on the context, extractive systems identify and return a portion of the context that directly answers the question.

Formally, given a question *Q* and a context document *D*, an extractive QA system aims to find the start and end indices (s, e) within *D* such that the substring *D[s:e+1]* represents the answer to *Q*.

We can use these systems in scenarios where the answer to a question is explicitly present within a given document or set of documents. This is useful for tasks like:

*   **Information Retrieval:** Finding specific answers within a knowledge base.
*   **Document Summarization:** Identifying key sentences that answer common questions about a document.
*   **Customer Support:** Answering user questions from a knowledge base of FAQs and documentation.
*   **Search Engines:** Highlighting the exact answer within a webpage snippet.

## 2) Application scenario

Imagine a user asks the question: "When was Albert Einstein born?"

And the system is given the following context document:

"Albert Einstein was a German-born theoretical physicist. He developed the theory of relativity, one of the two pillars of modern physics. His work is also known for its influence on the philosophy of science. He is best known to the general public for his mass-energy equivalence formula E = mc2 (which has been dubbed "the world's most famous equation"). He received the 1921 Nobel Prize in Physics for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect, a pivotal step in the development of quantum theory. He was born **March 14, 1879**, in Ulm, Württemberg, Germany. He died April 18, 1955, in Princeton, New Jersey, U.S."

An extractive QA system would identify the text span "**March 14, 1879**" within the context document as the answer to the question and return that span.

## 3) Python method (if possible)

We can use the `transformers` library from Hugging Face to implement an extractive QA system using pre-trained models.  This example uses the pre-trained model "bert-large-uncased-whole-word-masking-finetuned-squad".

```python
from transformers import pipeline

def answer_question(question, context):
    """
    Answers a question given a context using a pre-trained extractive QA model.
    """
    nlp = pipeline('question-answering', model='bert-large-uncased-whole-word-masking-finetuned-squad')
    QA_input = {
        'question': question,
        'context': context
    }
    result = nlp(QA_input)
    return result['answer'], result['score']

if __name__ == '__main__':
    question = "When was Albert Einstein born?"
    context = "Albert Einstein was a German-born theoretical physicist. He developed the theory of relativity, one of the two pillars of modern physics. His work is also known for its influence on the philosophy of science. He is best known to the general public for his mass-energy equivalence formula E = mc2 (which has been dubbed \"the world's most famous equation\"). He received the 1921 Nobel Prize in Physics for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect, a pivotal step in the development of quantum theory. He was born March 14, 1879, in Ulm, Württemberg, Germany. He died April 18, 1955, in Princeton, New Jersey, U.S."

    answer, score = answer_question(question, context)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f"Confidence Score: {score}")
```

This code defines a function `answer_question` that takes a question and context as input. It utilizes the `pipeline` function from `transformers` to create a question-answering pipeline. The pipeline uses a pre-trained BERT model fine-tuned for the SQuAD dataset, a benchmark dataset for extractive QA. The function then passes the question and context to the pipeline, which returns the extracted answer and a confidence score indicating how certain the model is that the answer is correct.

## 4) Follow-up question

How do extractive QA systems handle situations where the answer is not explicitly present in the provided context? What are some approaches to address this limitation?