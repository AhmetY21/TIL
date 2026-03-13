---
title: "Advanced Sentiment Analysis (Aspect-based)"
date: "2026-03-13"
week: 11
lesson: 4
slug: "advanced-sentiment-analysis-aspect-based"
---

# Topic: Advanced Sentiment Analysis (Aspect-based)

## 1) Formal definition (what is it, and how can we use it?)

Aspect-Based Sentiment Analysis (ABSA), also known as Feature-Based Sentiment Analysis, goes beyond simply identifying the overall sentiment of a text (positive, negative, neutral). Instead, it identifies the sentiment expressed toward *specific aspects* or *features* of a product, service, or entity mentioned in the text.

**What it is:** ABSA aims to:

*   **Identify aspects:** Determine which aspects or features are mentioned in the text. For example, in a restaurant review, aspects could be "food," "service," "ambiance," or "price."
*   **Determine sentiment polarity:**  Classify the sentiment (positive, negative, neutral, or sometimes more fine-grained emotions) expressed towards *each identified aspect*. For instance, the review might express positive sentiment towards the "food" but negative sentiment towards the "service."
*   **Resolve conflicts:**  Handle cases where different aspects within the same review receive different sentiment polarities. A single review can have both positive and negative sentiment, but ABSA clarifies *which* aspect is associated with each sentiment.

**How we can use it:** ABSA provides much more granular and actionable insights than traditional sentiment analysis. We can use it to:

*   **Identify areas for improvement:** Businesses can use ABSA to understand which aspects of their products or services customers are happy with and which areas need improvement.
*   **Improve product design:**  Understanding specific feature preferences can guide product development decisions.
*   **Monitor brand reputation:**  Track sentiment towards specific brand attributes over time.
*   **Personalize recommendations:** Recommend products or services based on a user's expressed sentiment towards specific features.
*   **Enhance customer service:**  Quickly identify customer pain points related to specific aspects of a product or service.

## 2) Application scenario

Imagine you are analyzing customer reviews for a new laptop model.

**Review Example:** "The screen is absolutely gorgeous and the battery life is amazing! However, the keyboard feels a bit cheap and the price is too high for what you get."

*   **Traditional Sentiment Analysis:** might classify the overall sentiment as "Neutral" or slightly "Positive" because it averages out the positive and negative comments. This misses crucial details.
*   **Aspect-Based Sentiment Analysis:** would identify the following:
    *   **Aspect:** "screen"
        *   **Sentiment:** "Positive"
    *   **Aspect:** "battery life"
        *   **Sentiment:** "Positive"
    *   **Aspect:** "keyboard"
        *   **Sentiment:** "Negative"
    *   **Aspect:** "price"
        *   **Sentiment:** "Negative"

By using ABSA, the laptop manufacturer can clearly see that while the screen and battery life are strong selling points, they need to improve the keyboard quality and potentially re-evaluate the pricing strategy. This actionable information is far more valuable than a simple overall sentiment score.

## 3) Python method (if possible)

While implementing a full ABSA system from scratch is complex, several Python libraries and techniques can be used.  One common approach involves combining rule-based methods with machine learning. Here's a simplified example using NLTK and a pre-trained sentiment analyzer (VADER) to illustrate the basic idea. Note that this example is extremely basic and would require significant expansion for real-world applications.

```python
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize

# Download necessary resources (run this once)
# nltk.download('vader_lexicon')
# nltk.download('punkt')

def aspect_based_sentiment_analysis(text, aspects):
    """
    Performs basic aspect-based sentiment analysis.

    Args:
        text: The text to analyze.
        aspects: A list of aspects to search for.

    Returns:
        A dictionary mapping aspects to their sentiment scores.
    """
    analyzer = SentimentIntensityAnalyzer()
    sentences = nltk.sent_tokenize(text)
    aspect_sentiments = {}

    for aspect in aspects:
        aspect_sentiments[aspect] = []

    for sentence in sentences:
        for aspect in aspects:
            if aspect in sentence.lower(): # Simple keyword matching
                vs = analyzer.polarity_scores(sentence)
                aspect_sentiments[aspect].append(vs['compound']) # Use compound score as sentiment

    # Aggregate sentiment scores for each aspect (e.g., average)
    result = {}
    for aspect, scores in aspect_sentiments.items():
        if scores:
            result[aspect] = sum(scores) / len(scores) #Average if aspect exists in sentence.
        else:
            result[aspect] = 0.0 #Aspect score is 0 if it is not in the sentence

    return result

# Example usage
text = "The screen is absolutely gorgeous and the battery life is amazing! However, the keyboard feels a bit cheap and the price is too high for what you get."
aspects = ["screen", "battery life", "keyboard", "price"]

sentiment_results = aspect_based_sentiment_analysis(text, aspects)
print(sentiment_results)

# Interpretation:
# Positive values indicate positive sentiment, negative values indicate negative sentiment, and 0 indicates neutral sentiment.  The closer the value is to 1 or -1, the stronger the sentiment.
```

**Explanation:**

1.  **SentimentIntensityAnalyzer:**  Uses NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner) for sentiment analysis. VADER is pre-trained to understand sentiments expressed in social media and general text.
2.  **Aspect Detection:** The code iterates through sentences and checks if each aspect (from the `aspects` list) is present in the sentence (case-insensitive). This is a very simple keyword-based approach.
3.  **Sentiment Scoring:** If an aspect is found, the sentiment score of the *entire sentence* is calculated using VADER.  A more sophisticated approach would focus on the specific phrases related to the aspect. The `compound` score is used as a general measure of sentiment.
4.  **Aggregation:** The code averages the sentiment scores for each aspect across all sentences where the aspect is mentioned.  This provides an overall sentiment score for each aspect.

**Limitations and Improvements:**

*   **Simple Keyword Matching:** This approach is naive and prone to errors. It doesn't understand context or handle synonyms. More sophisticated methods use dependency parsing, Named Entity Recognition (NER), and other NLP techniques to identify aspects more accurately.
*   **Sentence-Level Sentiment:**  Assigning the entire sentence's sentiment to an aspect can be inaccurate if the sentence discusses multiple aspects with different sentiments.  Aspect extraction and sentiment scoring should ideally be done at the phrase level.
*   **No Contextual Understanding:** The code doesn't understand contextual nuances or handle negation ("not good").
*   **Requires Pre-defined Aspects:** The code needs a predefined list of aspects. More advanced methods can automatically discover aspects.
*   **VADER is General Purpose:** VADER is a general-purpose sentiment analyzer. Fine-tuning or training a model on domain-specific data (e.g., laptop reviews) would significantly improve accuracy.

**Further Exploration:**

*   **Spacy:** Good for entity recognition and dependency parsing.
*   **Transformers (Hugging Face):** Powerful for fine-tuning pre-trained models for ABSA tasks. Libraries like `transformers` and `sentence-transformers` allow you to utilize state-of-the-art language models like BERT, RoBERTa, and others, and fine-tune them for aspect extraction and sentiment classification.
*   **CoreNLP:** Stanford's CoreNLP provides a range of NLP tools, including dependency parsing, which can be helpful for aspect extraction.
*   **SemEval ABSA Datasets:** Datasets like those from the SemEval workshops provide labeled data for training and evaluating ABSA models.

## 4) Follow-up question

How can I automatically identify aspects (instead of pre-defining them), and what are some common techniques for improving the accuracy of aspect extraction in ABSA?