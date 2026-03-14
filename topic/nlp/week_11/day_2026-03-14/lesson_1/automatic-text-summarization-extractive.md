---
title: "Automatic Text Summarization (Extractive)"
date: "2026-03-14"
week: 11
lesson: 1
slug: "automatic-text-summarization-extractive"
---

# Topic: Automatic Text Summarization (Extractive)

## 1) Formal definition (what is it, and how can we use it?)

Extractive text summarization is a method of automatically creating a summary of a text by selecting and concatenating sentences or phrases directly from the original text.  It identifies the most important sentences or segments within the original document and combines them to form a shorter, representative summary.  Crucially, it *does not* paraphrase or rewrite content; it merely selects existing text.

**How can we use it?**

*   **Quick content overview:** Quickly understand the main points of a large document without reading the entire thing.
*   **News aggregation:** Generate concise headlines or summaries for news articles.
*   **Search result snippet generation:** Provide relevant context for search results.
*   **Document browsing:** Preview document content before deciding to read in detail.
*   **Content monitoring:** Track key topics discussed in a large collection of documents.
*   **Data analysis:**  Reducing the size of text datasets for faster analysis.

## 2) Application scenario

Imagine you're a researcher studying climate change. You come across a lengthy research paper detailing the impact of deforestation on global warming.  Instead of reading the entire paper, you can use extractive text summarization to quickly get the key findings. The summarizer would identify and extract the sentences that contain the most crucial information about deforestation's impact (e.g., sentences discussing specific emissions increases, temperature changes, or ecological consequences). The resulting summary provides a concise overview of the paper's core arguments, allowing you to determine if the paper warrants a more in-depth reading. This saves significant time and effort in your research process. Another application is summarising product reviews. Extracting salient sentences allows a potential customer to gauge the overall sentiment and key features without reading hundreds of individual reviews.

## 3) Python method (if possible)

Here's an example using the `nltk` library to perform a simple extractive summarization based on sentence scoring (term frequency-inverse document frequency - TF-IDF):

```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
import heapq
import re

def extractive_summarization(text, num_sentences=3):
    """
    Performs extractive summarization using TF-IDF and sentence scoring.

    Args:
        text (str): The text to summarize.
        num_sentences (int): The desired number of sentences in the summary.

    Returns:
        str: The generated summary.
    """
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

    # 1. Preprocessing
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    formatted_text = re.sub('[^a-zA-Z]', ' ', text )
    formatted_text = re.sub(r'\s+', ' ', formatted_text)

    sentence_list = sent_tokenize(text)

    stopwords_list = set(stopwords.words('english'))

    word_frequencies = {}
    for word in word_tokenize(formatted_text):
        if word not in stopwords_list:
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    maximum_frequency = max(word_frequencies.values())

    for word in word_frequencies:
        word_frequencies[word] = (word_frequencies[word]/maximum_frequency)

    # 2. Sentence Scoring
    sentence_scores = {}
    for sent in sentence_list:
        for word in word_tokenize(sent.lower()):
            if word in word_frequencies:
                if len(sent.split(' ')) < 30: #Optional: Consider only shorter sentences. Adjust value as required.
                    if sent not in sentence_scores:
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    # 3. Summary Generation
    best_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(best_sentences)
    return summary

# Example usage:
sample_text = """
Automatic summarization is the process of reducing a text document with a computer program in order to create a summary that retains the most important points of the original document. Technologies that can make a coherent summary take into account variables such as length, writing style and syntax. Automatic data summarization is part of machine learning and data mining. The two main categories of summarization are extractive and abstractive. Extractive methods extract sentences from the text. Abstractive techniques rewrite the text.
"""

summary = extractive_summarization(sample_text, num_sentences=2)
print(summary)
```

**Explanation:**

1.  **Preprocessing:** Cleans the text by removing brackets, extra spaces, and non-alphabetic characters.  Tokenizes the text into sentences. Removes stop words (common words like "the," "a," "is").
2.  **TF-IDF (Term Frequency-Inverse Document Frequency):** Calculates the frequency of each word in the document. Normalizes the frequencies by dividing by the maximum frequency to emphasize important terms.
3.  **Sentence Scoring:** Scores each sentence based on the sum of the normalized frequencies of the words it contains.  Optionally, sentences that are too long may be excluded from consideration.
4.  **Summary Generation:** Selects the `num_sentences` highest-scoring sentences and concatenates them to create the summary.

**Important Notes:**

*   This is a *very* basic example.  More sophisticated extractive summarization techniques exist, using more advanced natural language processing techniques.
*   The quality of the summary depends heavily on the input text and the scoring method used.
*   This method only selects complete sentences.  More advanced techniques might select phrases or clauses.

## 4) Follow-up question

How can more advanced NLP techniques like word embeddings (e.g., Word2Vec, GloVe, or BERT embeddings) be incorporated into extractive summarization to improve the identification of important sentences and to achieve better summary coherence?