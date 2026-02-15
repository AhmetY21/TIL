---
title: "Naive Bayes Classifier for Text"
date: "2026-02-15"
week: 7
lesson: 1
slug: "naive-bayes-classifier-for-text"
---

# Topic: Naive Bayes Classifier for Text

## 1) Formal definition (what is it, and how can we use it?)

The Naive Bayes classifier is a probabilistic machine learning algorithm used for classification tasks.  "Naive" refers to the simplifying assumption of *feature independence*, meaning it assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature. In the context of text classification, this means it assumes the occurrence of a word in a document is independent of the occurrence of other words in the same document, given the class.  This assumption is almost always violated in real-world text, but surprisingly, the Naive Bayes classifier often performs well.

Specifically, for text classification, we use the Bayes' Theorem:

P(c|d) = [P(d|c) * P(c)] / P(d)

Where:

*   P(c|d) is the *posterior probability* of a document *d* belonging to class *c*. This is what we want to calculate.
*   P(d|c) is the *likelihood* of observing the document *d* given that it belongs to class *c*.  This is where the "naive" assumption comes in.  We approximate this by multiplying the probabilities of each word in the document occurring given the class: P(d|c) ≈ P(word1|c) * P(word2|c) * ... * P(wordN|c)
*   P(c) is the *prior probability* of class *c*, which is the proportion of documents in the training data belonging to class *c*.
*   P(d) is the *evidence* or probability of observing the document *d*. Since we only care about comparing the probabilities of different classes for the same document, P(d) acts as a normalizing constant and can often be ignored in the classification decision.

To classify a new document, we calculate P(c|d) for each possible class *c* and assign the document to the class with the highest probability.

There are several types of Naive Bayes classifiers tailored for different feature types:

*   **Multinomial Naive Bayes:** Suitable for discrete features, like word counts (term frequency) in text documents.  It models the probability of a document belonging to a class based on the frequencies of words in the document. This is commonly used in text classification.
*   **Bernoulli Naive Bayes:** Suitable for binary features, such as word presence/absence (boolean feature). It models the probability of a document belonging to a class based on whether or not certain words are present.
*   **Gaussian Naive Bayes:** Suitable for continuous features, and is not typically used directly for text classification as text data is usually discrete. However, you might use it for text features that have been transformed into continuous values via some other method, like word embeddings averaged to form a document vector.

## 2) Application scenario

Naive Bayes is commonly used in the following application scenarios involving text:

*   **Spam Filtering:** Classifying emails as spam or not spam. The features are typically the words present in the email.
*   **Sentiment Analysis:** Determining the sentiment of a piece of text (e.g., positive, negative, neutral). Reviews, tweets, and social media posts are common inputs.
*   **Topic Classification:** Categorizing documents into predefined topics or categories (e.g., sports, politics, technology).  This can be used for news articles, scientific papers, and customer support tickets.
*   **Author Identification:** Determining the author of a document based on their writing style.
*   **Language Detection:** Identifying the language of a given text.

Naive Bayes is a good choice for these scenarios when:

*   You have a large dataset.
*   Interpretability is important.
*   Speed and simplicity are prioritized.
*   The feature independence assumption, while not perfectly met, is not grossly violated.
*   You want a baseline model to compare against more complex models.

## 3) Python method (if possible)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Sample data (replace with your actual data)
documents = [
    "This is a positive review. I loved the product!",
    "This is a negative review. I hated it.",
    "Another positive review. Great experience!",
    "A terrible experience. Very disappointing.",
    "The movie was great!  I really enjoyed it.",
    "A boring movie.  I fell asleep."
]
labels = ['positive', 'negative', 'positive', 'negative', 'positive', 'negative']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(documents, labels, test_size=0.25, random_state=42)

# Create a pipeline: TF-IDF vectorizer -> Multinomial Naive Bayes
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")

# Example prediction on new data
new_document = ["This movie was absolutely fantastic!"]
predicted_class = model.predict(new_document)[0]
print(f"Predicted class for new document: {predicted_class}")
```

This code uses `sklearn` to:

1.  **Vectorize the text:** `TfidfVectorizer` converts the text documents into numerical feature vectors using TF-IDF (Term Frequency-Inverse Document Frequency).  This is a common technique for representing text data numerically. TF-IDF weighs words based on their frequency in a document and their inverse document frequency across the entire corpus, thus highlighting important terms.
2.  **Create a Naive Bayes model:** `MultinomialNB` is used because we are dealing with word frequencies.
3.  **Create a pipeline:**  `make_pipeline` chains the vectorizer and classifier together.  This simplifies the workflow.
4.  **Train the model:** The model is trained using the training data.
5.  **Predict and Evaluate:** The model predicts labels for the test data and the accuracy and classification report are printed.
6.  **Predict new data:** Shows how to classify a new, unseen document.

## 4) Follow-up question

What are some common techniques to improve the performance of a Naive Bayes classifier for text, especially considering the naive independence assumption?