---
title: "LLM Hallucinations and Mitigation"
date: "2026-03-18"
week: 12
lesson: 2
slug: "llm-hallucinations-and-mitigation"
---

# Topic: LLM Hallucinations and Mitigation

## 1) Formal definition (what is it, and how can we use it?)

LLM (Large Language Model) hallucinations refer to instances where an LLM generates content that is factually incorrect, nonsensical, or irrelevant to the prompt, even if it sounds plausible. Essentially, the model is confidently presenting information that is not grounded in reality or its training data. This can range from fabricating details about events, people, or places, to creating entirely new "facts" or citing nonexistent sources.

**What is it?**  A key characteristic is the model's *confidence* in its incorrect statement. It's not just an error; it's an error delivered with conviction.  Hallucinations can manifest in different forms:

*   **Factual Hallucinations:** Presenting incorrect facts about the real world.
*   **Inferential Hallucinations:** Drawing incorrect conclusions or making unjustified assumptions based on the given context.
*   **Internal Hallucinations:** Contradicting information it previously presented in the same conversation or response.
*   **Input Hallucinations:** Misinterpreting or ignoring parts of the input context.

**How can we use it?**  While hallucinations are generally undesirable, understanding them allows us to:

*   **Develop better evaluation metrics:** Metrics that specifically target and measure hallucination rates in LLM outputs are crucial for model improvement.
*   **Design more robust prompting strategies:**  Crafting prompts that minimize ambiguity and encourage the model to stay grounded in verifiable information.  For instance, asking for sources or limiting the scope of the response.
*   **Implement mitigation techniques:**  Techniques like retrieval-augmented generation (RAG), fine-tuning with factual data, and incorporating verification steps can reduce the likelihood of hallucinations.
*   **Build more trustworthy LLM applications:**  By understanding the potential for hallucinations, we can design applications that are less susceptible to their negative consequences, such as providing inaccurate medical advice.

## 2) Application scenario

Imagine you are building a question-answering chatbot designed to provide historical information based on a large text corpus. A user asks: "What was the significance of the Battle of Alesia in the Punic Wars?".

Without proper safeguards, an LLM prone to hallucination might confidently answer: "The Battle of Alesia was a decisive naval battle in the Second Punic War where the Roman fleet, led by Scipio Africanus, defeated the Carthaginian navy commanded by Hannibal Barca. This victory allowed Rome to secure control of the Mediterranean and ultimately win the war."

This response contains a *factual hallucination*. The Battle of Alesia was a siege during the *Gallic Wars*, led by Julius Caesar. Hannibal Barca was not involved, and it was a land battle, not a naval one. The model conflates details and creates a completely fictional scenario.

This hallucination, if presented to a user seeking accurate historical information, could mislead them and erode trust in the chatbot. A real-world application, such as a research assistant or educational tool, could propagate misinformation.

## 3) Python method (if possible)

While there isn't a single Python function to *prevent* hallucinations directly, we can use Python to implement techniques for *detecting* and mitigating them through retrieval-augmented generation (RAG). This example shows RAG usage, which seeks to reduce hallucinations by grounding the LLM in external data.
```python
from transformers import pipeline
from datasets import load_dataset

# 1. Load a pre-trained question-answering model
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

# 2. Load a relevant dataset (e.g., Wikipedia articles)
# For demonstration purposes, we'll use a small subset. In a real application, you'd use a more comprehensive knowledge base.
dataset = load_dataset("wikipedia", "20220301.en", split="train[:100]") # Load the first 100 articles

# 3. Define a function to retrieve relevant context based on the question
def retrieve_context(question, dataset):
    """Retrieves the most relevant text snippet from the dataset based on the question."""
    # Simple keyword matching for demonstration.  More sophisticated methods exist (e.g., semantic search).
    keywords = question.split()
    relevant_articles = []
    for article in dataset:
        if any(keyword in article["text"] for keyword in keywords):
            relevant_articles.append(article["text"])

    # Return the first relevant article.  In practice, use a ranking function to select the best.
    if relevant_articles:
        return relevant_articles[0]
    else:
        return "No relevant context found in the dataset."

# 4. Define a function to generate an answer using the retrieved context
def answer_question_with_context(question, context, qa_model):
    """Answers the question using the provided context and question-answering model."""
    result = qa_model(question=question, context=context)
    return result["answer"]

# 5. Example usage
question = "What is the capital of France?"
context = retrieve_context(question, dataset)
answer = answer_question_with_context(question, context, qa_model)

print(f"Question: {question}")
print(f"Answer: {answer}")
print(f"Context: {context[:200]}...")  # Print the first 200 characters of the context
```

**Explanation:**

1.  **Loading Resources**: The code loads a pre-trained question-answering model (RoBERTa) and a subset of Wikipedia articles.  In a real-world scenario, you'd use a larger, more targeted dataset for the knowledge base.
2.  **Context Retrieval**: The `retrieve_context` function uses a simple keyword-based search to find relevant articles in the dataset.  More sophisticated methods like semantic search (using sentence embeddings) would significantly improve the accuracy of context retrieval.
3.  **Question Answering**: The `answer_question_with_context` function utilizes the question-answering model to generate an answer based on the retrieved context.  This grounds the answer in the available information, reducing the likelihood of hallucination.
4.  **RAG principle:** The code demonstrates Retrieval-Augmented Generation. The question is first used to retrieve relevant information from an external knowledge source. Then, the question and the retrieved information are provided to the LLM to generate an answer. This significantly reduces hallucinations because the LLM is grounded in external knowledge rather than solely relying on its pre-trained knowledge.

**Limitations:**

*   This RAG approach is not foolproof. If the retrieved context is incorrect or incomplete, the answer may still be flawed.
*   Keyword-based retrieval is a basic method. Semantic search with vector databases (like FAISS or ChromaDB) offers significantly better context retrieval.
*   The code doesn't include steps to verify the answer's accuracy beyond relying on the retrieved context.

## 4) Follow-up question

Beyond RAG, what are some other promising techniques or research directions aimed at reducing LLM hallucinations, particularly those focused on improving the model's internal knowledge representation or reasoning capabilities? Specifically, can you elaborate on techniques like self-consistency, fine-tuning with factual datasets, or methods for improving the interpretability of LLM decision-making processes?