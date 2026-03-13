---
title: "Automatic Text Summarization (Extractive)"
date: "2026-03-13"
week: 11
lesson: 6
slug: "automatic-text-summarization-extractive"
---

# Topic: Automatic Text Summarization (Extractive)

## 1) Formal definition (what is it, and how can we use it?)

Extractive text summarization is a method of creating a summary by directly selecting and concatenating important sentences or phrases from the original document.  The core idea is to identify the most salient pieces of information within the source text and assemble them to create a concise representation.  No new words or phrases are generated; the summary consists solely of excerpts from the original text.

We can use extractive summarization in scenarios where:

*   **Speed is important:** Extractive methods are generally faster than abstractive methods because they don't involve generating new text.
*   **Fidelity to the original text is crucial:**  Since the summary is composed of verbatim excerpts, it preserves the original phrasing and avoids potential factual inaccuracies introduced by paraphrasing.
*   **Large volumes of text need to be processed:**  Extractive summarization can quickly generate summaries for a large number of documents.
*   **There is a need to understand the overall topic of a lengthy document quickly:** Extractive summaries can highlight the most important aspects.

## 2) Application scenario

Imagine a news aggregator that collects articles from various sources.  To help users quickly scan the news, the aggregator uses extractive summarization to generate short summaries for each article.  A user can then read the summary to decide whether the full article is of interest, significantly reducing the time spent browsing news headlines.  This is more efficient than relying solely on article titles, which may not be fully informative. This application prioritizes speed and accuracy, as readers want a quick and truthful overview of the news story.

## 3) Python method (if possible)

Here's a Python example using the `sumy` library, a popular package for text summarization:

```python
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

LANGUAGE = "english"
SENTENCES_COUNT = 3  # Number of sentences in the summary

text = """
The Orbiter Discovery, after orbiting for 365 days, landed safely today.
Astronauts celebrated their successful mission.
During the mission, they repaired a satellite.
They also conducted several scientific experiments.
The mission was deemed a complete success by NASA.
"""

parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
stemmer = Stemmer(LANGUAGE)

summarizer = Summarizer(stemmer)
summarizer.stop_words = get_stop_words(LANGUAGE)

for sentence in summarizer(parser.document, SENTENCES_COUNT):
    print(sentence)
```

**Explanation:**

1.  **Import necessary modules:** Imports classes for parsing text, tokenizing sentences, stemming words, and using LSA summarization.
2.  **Define parameters:** Sets the language to English and specifies the desired number of sentences in the summary.
3.  **Provide input text:** `text` variable contains the document to be summarized.
4.  **Create a parser:** `PlaintextParser` converts the input text into a format suitable for `sumy`.
5.  **Initialize Stemmer:** `Stemmer` normalizes words to their root form (e.g., "running" to "run"). This helps in identifying similar sentences.
6.  **Initialize Summarizer:** `LsaSummarizer` uses Latent Semantic Analysis (LSA) to identify the most important sentences. You can replace `LsaSummarizer` with other summarizers provided by sumy (e.g., `LexRankSummarizer`, `TextRankSummarizer`).
7.  **Set Stop Words:** Stop words (e.g., "the", "a", "is") are common words that are generally not informative and are removed to improve performance.
8.  **Generate Summary:** The `summarizer(parser.document, SENTENCES_COUNT)` call generates the summary.
9.  **Print Summary:** The code iterates through the selected sentences and prints them.

**Output:**

```
The Orbiter Discovery, after orbiting for 365 days, landed safely today.
Astronauts celebrated their successful mission.
The mission was deemed a complete success by NASA.
```

## 4) Follow-up question

What are some of the limitations of extractive text summarization, and how do abstractive methods address these limitations?