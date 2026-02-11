---
title: "Text Normalization Techniques"
date: "2026-02-11"
week: 7
lesson: 4
slug: "text-normalization-techniques"
---

# Topic: Text Normalization Techniques

## 1) Formal definition (what is it, and how can we use it?)

Text normalization is the process of transforming text into a more uniform and consistent format. It's a crucial preprocessing step in Natural Language Processing (NLP) that aims to reduce variability in the text data without changing its core meaning. This uniformity makes it easier for NLP models to accurately process and analyze the text.

Essentially, text normalization boils down to applying a series of transformations to convert text into a standard form. These transformations handle issues like:

*   **Case variations:** Converting all text to lowercase or uppercase to ensure that "Hello" and "hello" are treated the same.
*   **Punctuation removal:** Eliminating punctuation marks that don't contribute significantly to the meaning of the text.
*   **Stop word removal:** Removing common words (e.g., "the," "a," "is") that often have little semantic value.
*   **Stemming:** Reducing words to their root or stem form (e.g., "running" -> "run").
*   **Lemmatization:** Reducing words to their dictionary form or lemma (e.g., "better" -> "good").
*   **Spelling correction:** Correcting typographical errors and spelling mistakes.
*   **Tokenization:** Breaking down text into individual words or tokens.
*   **Handling contractions:** Expanding contractions (e.g., "can't" -> "cannot").
*   **Removing special characters:** Removing characters beyond standard alphanumeric ones.
*   **Number and date formatting:** Converting numbers and dates to consistent formats.

We use text normalization to:

*   **Improve the accuracy of NLP models:** Consistent input data leads to better model performance.
*   **Reduce computational cost:** By reducing the vocabulary size, we can speed up processing and reduce memory usage.
*   **Simplify analysis:** Standardized text is easier to analyze and interpret.
*   **Increase efficiency:** Standardized text enhances processing speed.

## 2) Application scenario

Consider a sentiment analysis task where we want to classify movie reviews as either positive or negative. The reviews may contain a wide variety of text variations. For example:

*   "This movie was AMAZING!!!"
*   "The acting was awful :("
*   "It was good, I guess."
*   "aWful movie!! bad bad bad"

Without text normalization, the model might treat "AMAZING," "amazing," and "Amazing" as different words, potentially diluting the model's ability to identify the strong positive sentiment associated with the term. Similarly, the model might not recognize that "awful" and "aWful" are the same word, or it might be distracted by the punctuation.  Stop words like "was" or "the" add no value to sentiment.

By applying text normalization techniques like lowercasing, punctuation removal, stop word removal, and potentially stemming/lemmatization, we can transform these reviews into a more consistent format, allowing the sentiment analysis model to focus on the core sentiment-bearing words and achieve better accuracy.

## 3) Python method (if possible)

```python
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re

# Download necessary NLTK resources (only needed once)
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')


def normalize_text(text):
    """
    Normalizes text by lowercasing, removing punctuation,
    removing stop words, stemming, and lemmatizing.
    """

    # Lowercasing
    text = text.lower()

    # Removing punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenization
    tokens = nltk.word_tokenize(text)

    # Removing stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in stemmed_tokens] #lemmatizing after stemming often works best


    # Rejoin tokens into a single string
    normalized_text = ' '.join(lemmatized_tokens)

    return normalized_text

# Example usage
text = "This movie was AMAZING!!! The acting was awful :("
normalized_text = normalize_text(text)
print(f"Original text: {text}")
print(f"Normalized text: {normalized_text}")


# More advanced normalization (using more regular expressions)
def normalize_text_advanced(text):
    """Normalizes text with more aggressive techniques"""
    text = text.lower()  # Lowercase
    text = re.sub(r"http\S+|www\S+|@\S+", "", text)  # Remove URLs and mentions
    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove punctuation and special characters, keeping only letters and spaces
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    normalized_text = ' '.join(lemmatized_tokens)

    return normalized_text

text2 = "This movie was AMAZING!!! Check out the website at https://www.example.com.  The acting was awful :(  Visit our site at <a href='...'>link</a>. @john, what did you think?"
normalized_text2 = normalize_text_advanced(text2)
print(f"\nOriginal text: {text2}")
print(f"Advanced Normalized text: {normalized_text2}")
```

## 4) Follow-up question

How do you determine the optimal combination of text normalization techniques for a specific NLP task? Are there any established metrics or guidelines to help guide this process, and how do those decisions differ between different types of NLP problems (e.g., text classification vs. machine translation)?