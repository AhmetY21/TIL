---
title: "Automatic Text Summarization (Extractive)"
date: "2026-02-25"
week: 9
lesson: 1
slug: "automatic-text-summarization-extractive"
---

# Topic: Automatic Text Summarization (Extractive)

## 1) Formal definition (what is it, and how can we use it?)

Extractive text summarization is a type of automatic text summarization where the summary is formed by directly selecting and concatenating sentences from the original source text.  It identifies the most important sentences or phrases within the document and extracts them to create a condensed version.  It doesn't involve rewriting or paraphrasing the original content; instead, it relies on scoring and ranking existing segments based on various features.

We can use extractive summarization to:

*   **Quickly grasp the main points of a document:**  Instead of reading an entire article or report, you can read the extracted summary to get a general understanding.
*   **Identify relevant information:**  It helps to pinpoint specific sentences or phrases that are most relevant to a user's query.
*   **Create headlines or abstracts:** The extracted sentences can be used to generate concise headlines or abstracts for larger documents.
*   **Pre-process text for other NLP tasks:**  Reducing the length of text can improve the efficiency of subsequent NLP tasks like topic modeling or sentiment analysis.
*   **Document clustering and indexing:** Useful for quickly characterizing documents in large collections.

## 2) Application scenario

Imagine you're a news aggregator site.  You want to display short summaries of news articles so that users can quickly scan headlines and decide which articles to read further. Extractive summarization is ideal here. You can automatically generate a summary by selecting the most important sentences from each article and displaying them below the headline.  This allows users to quickly assess the relevance of the article without having to open it.  The summary provides a good idea of the article's key points without introducing any new information or requiring human intervention for rewriting.

## 3) Python method (if possible)
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def extractive_summarization(text, num_sentences=3):
    """
    Performs extractive text summarization using TF-IDF and cosine similarity.

    Args:
      text: The input text (string).
      num_sentences: The desired number of sentences in the summary.

    Returns:
      A string containing the extracted summary.
    """

    sentences = text.split(".") # Simple sentence splitting (can be improved)
    sentences = [s.strip() for s in sentences if s] #Remove empty strings
    if not sentences:
      return ""

    vectorizer = TfidfVectorizer()
    sentence_vectors = vectorizer.fit_transform(sentences)

    similarity_matrix = cosine_similarity(sentence_vectors)

    sentence_scores = np.sum(similarity_matrix, axis=1)

    ranked_sentences = sorted(((sentence_scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    summary_sentences = [s for _, s in ranked_sentences[:num_sentences]]

    summary = ". ".join(summary_sentences)
    return summary

# Example usage:
text = """Automatic summarization is the process of shortening a text document with software, in order to create a summary with the major points of the original document. Technologies that can make a coherent summary take into account variables such as length, writing style and syntax. Automatic data summarization is part of machine learning and data mining. The two main types of summarization are extractive and abstractive. Extractive summarization chooses existing sentences from the text to form the summary. Abstractive summarization rewrites the text to create a new, shorter text using natural language generation techniques."""

summary = extractive_summarization(text, num_sentences=2)
print(summary)
```

## 4) Follow-up question

How can we improve the accuracy and quality of extractive summarization, especially considering the limitations of using TF-IDF and cosine similarity alone? What other features or techniques can be incorporated, and what are their trade-offs (e.g., computational complexity, data requirements)?