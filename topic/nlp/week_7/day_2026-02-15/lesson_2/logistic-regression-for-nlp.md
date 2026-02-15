---
title: "Logistic Regression for NLP"
date: "2026-02-15"
week: 7
lesson: 2
slug: "logistic-regression-for-nlp"
---

# Topic: Logistic Regression for NLP

## 1) Formal definition (what is it, and how can we use it?)

Logistic Regression, despite its name, is actually a *classification* algorithm used in NLP, not a regression algorithm in the traditional sense (predicting a continuous value). It's used to predict the probability of a categorical outcome based on a set of predictor variables (features).

**What is it?**

*   It's a linear model that transforms a linear combination of input features using the *logistic sigmoid function* (also known as the sigmoid function or the logistic function) to produce a probability value between 0 and 1.  The sigmoid function is defined as:

    `sigmoid(x) = 1 / (1 + exp(-x))`

*   The linear combination of features is calculated as:

    `z = b + w1*x1 + w2*x2 + ... + wn*xn`

    where:
    *   `b` is the bias (intercept) term.
    *   `w1, w2, ..., wn` are the weights associated with the features.
    *   `x1, x2, ..., xn` are the values of the input features.

*   This `z` value is then passed through the sigmoid function:

    `p = sigmoid(z) = 1 / (1 + exp(-z))`

    `p` represents the predicted probability of the input belonging to the positive class (typically class 1).

*   A decision threshold (usually 0.5) is used to classify the instance. If `p >= 0.5`, the instance is classified as belonging to the positive class; otherwise, it's classified as belonging to the negative class (typically class 0).

**How can we use it in NLP?**

Logistic regression can be used in a variety of NLP tasks by treating text or words as features. Examples include:

*   **Sentiment Analysis:**  Predict whether a piece of text expresses positive or negative sentiment.  Features could include word frequencies (bag-of-words), presence of certain keywords, or n-grams.
*   **Spam Detection:** Classify emails as spam or not spam.  Features could be the presence of certain words ("free", "discount"), the sender's address, or the structure of the email.
*   **Part-of-Speech (POS) Tagging:** Classify a word in a sentence based on its POS (noun, verb, adjective, etc.). Features could include the word itself, its surrounding words, and its morphological features.
*   **Text Categorization:** Assign a category or topic to a document. Features could be word frequencies, TF-IDF scores, or document embeddings.

## 2) Application scenario

Let's consider a **sentiment analysis** scenario. We want to build a system that can automatically determine whether a movie review is positive or negative.

1.  **Data Collection:**  Gather a labeled dataset of movie reviews, where each review is labeled as either "positive" or "negative".
2.  **Feature Extraction:**  Convert the text of each review into numerical features.  A simple approach is to use the "bag-of-words" model. This involves:
    *   Creating a vocabulary of all unique words in the corpus.
    *   Representing each review as a vector where each element corresponds to the frequency of a word from the vocabulary in that review.  For example, if the vocabulary is ["good", "bad", "movie", "acting"], the review "good movie good" would be represented as [2, 0, 1, 0].
3.  **Training the Model:** Train a logistic regression model using the extracted features and the corresponding sentiment labels. The model learns the weights associated with each word that best predict the sentiment.
4.  **Prediction:** Given a new movie review, extract its features (using the same bag-of-words approach), and use the trained logistic regression model to predict its sentiment (positive or negative).

## 3) Python method (if possible)

We can use scikit-learn's `LogisticRegression` class.

```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample data (replace with your actual data)
reviews = [
    "This movie was great!",
    "The acting was terrible.",
    "I really enjoyed the film.",
    "It was a complete waste of time.",
    "Absolutely amazing, I loved it!",
    "The plot was boring and predictable."
]
labels = [1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative

# 1. Data Splitting
X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=42)

# 2. Feature Extraction (Bag of Words)
vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)  # Note: Use transform on the test set

# 3. Train the Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_vectors, y_train)

# 4. Make Predictions
y_pred = model.predict(X_test_vectors)

# 5. Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Example Prediction
new_review = ["This was an ok movie."]
new_review_vector = vectorizer.transform(new_review)
prediction = model.predict(new_review_vector)[0]  # Predict a single review
if prediction == 1:
    print("Sentiment: Positive")
else:
    print("Sentiment: Negative")
```

## 4) Follow-up question

How does the performance of logistic regression in NLP compare to more complex models like deep learning models (e.g., transformers) for tasks like sentiment analysis?  Under what circumstances might logistic regression still be a better choice?