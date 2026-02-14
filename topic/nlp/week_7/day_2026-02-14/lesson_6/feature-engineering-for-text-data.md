---
title: "Feature Engineering for Text Data"
date: "2026-02-14"
week: 7
lesson: 6
slug: "feature-engineering-for-text-data"
---

# Topic: Feature Engineering for Text Data

## 1) Formal definition (what is it, and how can we use it?)

Feature engineering for text data involves transforming raw textual data into numerical or categorical representations that machine learning models can understand and utilize.  Essentially, it's the process of creating meaningful, informative features from text that capture its underlying structure, semantics, and context.  Raw text is inherently unstructured and complex, making it unsuitable for direct input into most machine learning algorithms.

We use feature engineering in NLP to:

*   **Represent textual information numerically:**  Machine learning models predominantly work with numerical data. Feature engineering converts text into numbers or vectors, allowing models to perform tasks like classification, regression, or clustering.
*   **Capture semantic meaning:**  Moving beyond simple word counts, feature engineering can capture relationships between words, sentiments expressed in the text, and the overall topic or theme.
*   **Improve model performance:**  Well-engineered features can significantly boost the accuracy and efficiency of machine learning models by providing them with relevant and discriminative information.
*   **Reduce dimensionality:** Raw text often has a very high dimensionality (i.e., a large vocabulary). Feature engineering techniques can reduce this dimensionality while retaining the most important information.
*   **Handle different text properties:** Different feature engineering methods are suitable for handling various text properties, such as character-level information, word-level information, document-level information, and sequential information.

Common feature engineering techniques include:

*   **Bag of Words (BoW):** Representing a document by the frequency of its words, ignoring grammar and word order.
*   **TF-IDF (Term Frequency-Inverse Document Frequency):** Weighing words based on their frequency in a document and their rarity across the entire corpus.
*   **N-grams:**  Sequences of *n* words that capture some context.
*   **Word Embeddings (Word2Vec, GloVe, FastText):** Dense vector representations of words that capture semantic relationships.
*   **Character-level features:**  Using the frequency or presence of specific characters or character sequences.
*   **Sentiment scores:** Numerical representation of the sentiment (positive, negative, neutral) expressed in the text.
*   **Part-of-Speech (POS) tagging:** Identifying the grammatical role of each word (e.g., noun, verb, adjective) and using these tags as features.
*   **Document length and complexity:**  Number of words, sentences, average word length, etc.

## 2) Application scenario

Let's consider a spam email detection scenario. We want to build a machine learning model that can classify emails as either "spam" or "not spam" (ham).

**Without Feature Engineering:**

We could theoretically feed the raw text of the emails directly into a deep learning model (like a Transformer), but this would require a significant amount of data and computational resources.  Even then, simpler models might perform better with properly engineered features.

**With Feature Engineering:**

We can use feature engineering to extract meaningful features from the email text:

1.  **TF-IDF:** Calculate the TF-IDF scores for each word in each email.  Words like "free," "discount," and "urgent" are likely to have higher TF-IDF scores in spam emails.
2.  **N-grams:**  Consider bi-grams (pairs of words) like "limited time," which are common in spam messages.
3.  **Sentiment Analysis:** Calculate a sentiment score for each email. Spam emails might use manipulative language that results in a higher negative sentiment score.
4.  **Email Metadata:** Extract features from the email headers, such as the sender's domain, whether the email contains attachments, and the number of recipients.
5.  **Presence of URLS:** The number of URLS and whether they are obfuscated (e.g., shortened links).
6.  **Punctuation Frequency:** Spam emails often contain a higher frequency of exclamation marks and other excessive punctuation.

These features can then be used as input to a simpler machine learning model like a Logistic Regression or Support Vector Machine. The feature engineering process would help the model to better discriminate between spam and non-spam emails, leading to improved accuracy and faster training times.

## 3) Python method (if possible)

Here's an example using scikit-learn to create TF-IDF features:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Sample email data (replace with your actual data)
emails = [
    "This is a normal email.",
    "Get free money now!",
    "Urgent: Claim your prize!",
    "Meeting scheduled for tomorrow."
]

# Create a TfidfVectorizer object
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=2)
# stop_words removes common words like "the", "a", "is"
# max_df ignores terms that appear in more than 80% of the documents
# min_df ignores terms that appear in less than 2 documents.

# Fit the vectorizer to the email data and transform the data
tfidf_matrix = vectorizer.fit_transform(emails)

# Convert the TF-IDF matrix to a pandas DataFrame for easier inspection
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

print(tfidf_df)
```

This code snippet performs the following:

1.  **Imports necessary libraries:** `TfidfVectorizer` from scikit-learn and `pandas` for data handling.
2.  **Defines sample email data:**  A list of strings representing emails.  You would replace this with your actual data.
3.  **Creates a `TfidfVectorizer`:** Configures the vectorizer to remove common English stop words, ignore words that appear in more than 80% of the documents, and ignore words that appear in less than 2 documents.
4.  **Fits and transforms the data:**  `fit_transform` learns the vocabulary from the email data and transforms each email into a TF-IDF vector.
5.  **Converts to a DataFrame:**  The TF-IDF matrix is converted into a pandas DataFrame for easier visualization and analysis. The columns represent the terms (words) in the vocabulary, and the rows represent the emails. The values in the DataFrame are the TF-IDF scores for each word in each email.

## 4) Follow-up question

Given the advancements in pre-trained language models like BERT and GPT, how does the importance of traditional feature engineering techniques like TF-IDF compare to fine-tuning these pre-trained models for text classification tasks? When would you choose one approach over the other?