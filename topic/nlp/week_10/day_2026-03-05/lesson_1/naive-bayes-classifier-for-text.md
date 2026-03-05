---
title: "Naive Bayes Classifier for Text"
date: "2026-03-05"
week: 10
lesson: 1
slug: "naive-bayes-classifier-for-text"
---

# Topic: Naive Bayes Classifier for Text

## 1) Formal definition (what is it, and how can we use it?)

The Naive Bayes classifier is a probabilistic machine learning algorithm used for classification tasks. In the context of text, it leverages Bayes' theorem with a "naive" assumption of feature independence to determine the probability of a given text document belonging to a specific category or class.

**Bayes' Theorem:**

P(c|d) = [P(d|c) * P(c)] / P(d)

Where:

*   P(c|d) is the **posterior probability** of the document 'd' belonging to class 'c'. This is what we want to calculate.
*   P(d|c) is the **likelihood** of observing document 'd' given that it belongs to class 'c'. This is estimated from the training data.
*   P(c) is the **prior probability** of class 'c'.  This is the probability of a document belonging to class 'c' before seeing the document itself (estimated from the frequency of the class in the training data).
*   P(d) is the **probability of the document** 'd'. This acts as a normalizing constant and is often ignored since it's the same for all classes being compared.

**The "Naive" Assumption:**

The "naive" part comes from the assumption that the features (words) in a document are conditionally independent given the class. In reality, this is almost never true.  For example, the word "terrible" is highly correlated with the word "movie" when talking about film reviews. Despite this unrealistic assumption, Naive Bayes often performs surprisingly well in text classification.

**How we use it:**

1.  **Training:** Given a labeled training dataset (documents with known categories), the algorithm calculates the prior probabilities P(c) for each class and the likelihoods P(d|c) for each word given each class. Typically, this is done by counting the frequency of words within each class and applying smoothing (e.g., Laplace smoothing) to avoid zero probabilities for unseen words.  The document 'd' is often represented as a vector of word frequencies (Bag-of-Words).

2.  **Classification:** Given a new, unseen document, the algorithm calculates the posterior probability P(c|d) for each class using the learned prior probabilities and likelihoods. The document is then assigned to the class with the highest posterior probability.

Different variants of Naive Bayes exist based on the assumed distribution of features, such as:

*   **Multinomial Naive Bayes:**  Suitable for discrete features, such as word counts or term frequencies.  Commonly used in text classification.
*   **Bernoulli Naive Bayes:** Suitable for binary features, such as the presence or absence of a word.
*   **Gaussian Naive Bayes:** Suitable for continuous features, assuming they follow a Gaussian distribution. Less common for text.

## 2) Application scenario

Naive Bayes classifiers are well-suited for various text classification tasks, including:

*   **Spam detection:** Classifying emails as spam or not spam based on the words they contain.
*   **Sentiment analysis:** Determining the sentiment (positive, negative, neutral) expressed in a piece of text, such as a product review or a tweet.
*   **Topic classification:** Categorizing news articles or research papers into different topics (e.g., sports, politics, technology).
*   **Author identification:** Identifying the author of a document based on their writing style and vocabulary.
*   **Language detection:** Determining the language of a given text.

Naive Bayes is particularly useful when:

*   The dataset is relatively large.
*   High accuracy is not strictly required (it's often a good baseline model).
*   Interpretability is desired (the learned probabilities can provide insights into the features that contribute to each class).
*   Speed is important (it's a relatively fast algorithm to train and classify).

## 3) Python method (if possible)

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Sample data (replace with your actual data)
documents = [
    "This is a positive review. I loved the movie!",
    "The movie was terrible. I hated it.",
    "This is another positive review. Excellent!",
    "The product is bad. I do not recommend it.",
    "I really enjoyed the service.  Great experience.",
    "The service was awful. Very disappointing."
]
labels = ['positive', 'negative', 'positive', 'negative', 'positive', 'negative']

# 1. Convert text to numerical data using CountVectorizer (Bag-of-Words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)  # Learn vocabulary and transform documents

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# 3. Train a Multinomial Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# 4. Make predictions on the test set
y_pred = classifier.predict(X_test)

# 5. Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))

# Example of predicting a new document
new_document = ["This is a great product!"]
new_document_vectorized = vectorizer.transform(new_document) # IMPORTANT: use transform, not fit_transform
prediction = classifier.predict(new_document_vectorized)
print(f"Prediction for '{new_document[0]}': {prediction[0]}")
```

Explanation:

*   **CountVectorizer:** Converts the text documents into a matrix of token counts. `fit_transform` learns the vocabulary from the training data and transforms the documents into a numerical representation. It creates a sparse matrix where each row represents a document and each column represents a word in the vocabulary. The values in the matrix are the counts of each word in each document.  Importantly, when predicting on new data, you should only use `.transform()` to convert the new documents to a numerical format *using the vocabulary learned during training*.  Using `.fit_transform()` again will create a *new* vocabulary and lead to errors or poor performance.
*   **train\_test\_split:** Splits the data into training and testing sets to evaluate the model's performance on unseen data.
*   **MultinomialNB:**  Instantiates and trains the Multinomial Naive Bayes classifier.
*   **fit(X\_train, y\_train):**  Trains the classifier using the training data.
*   **predict(X\_test):**  Predicts the labels for the test data.
*   **accuracy\_score & classification\_report:**  Evaluate the model's performance.  The `classification_report` provides precision, recall, and F1-score for each class.
*   **Example Prediction:** Shows how to predict the class of a new document.

## 4) Follow-up question

How does Laplace smoothing (or other forms of smoothing) impact the performance of a Naive Bayes classifier for text data, and why is it important to use? What are some potential drawbacks of applying too much smoothing?