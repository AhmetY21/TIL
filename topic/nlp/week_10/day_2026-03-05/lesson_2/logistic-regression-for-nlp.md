---
title: "Logistic Regression for NLP"
date: "2026-03-05"
week: 10
lesson: 2
slug: "logistic-regression-for-nlp"
---

# Topic: Logistic Regression for NLP

## 1) Formal definition (what is it, and how can we use it?)

Logistic Regression, despite its name, is a **linear classifier** used for **binary or multi-class classification problems** in Natural Language Processing (NLP). It models the probability of a binary outcome (e.g., spam or not spam, positive or negative sentiment) or a categorical outcome (e.g., topic 1, topic 2, topic 3) based on a linear combination of input features.

**How it works:**

1.  **Input Features:**  The input is a set of features, typically derived from text data. These features can be:
    *   **Bag-of-Words (BoW):** Frequency of each word in the document.
    *   **TF-IDF (Term Frequency-Inverse Document Frequency):** Weights words based on their importance in a document relative to a corpus.
    *   **N-grams:** Sequences of N words (e.g., "very good", "not happy").
    *   **Word Embeddings (e.g., Word2Vec, GloVe, fastText):** Vector representations of words capturing semantic relationships.

2.  **Linear Combination:** The features are combined linearly with learned weights (coefficients) and a bias (intercept) term.  The equation is:

    `z = w1 * x1 + w2 * x2 + ... + wn * xn + b`

    where:
    *   `z` is the linear combination.
    *   `w1, w2, ..., wn` are the weights for each feature.
    *   `x1, x2, ..., xn` are the feature values.
    *   `b` is the bias.

3.  **Sigmoid Function (for binary classification):**  The linear combination `z` is then passed through a sigmoid (logistic) function:

    `p = 1 / (1 + e^(-z))`

    The sigmoid function outputs a probability `p` between 0 and 1.

4.  **Prediction:**  A threshold (usually 0.5) is used to classify the input. If `p >= 0.5`, the input is classified as belonging to class 1; otherwise, it's classified as class 0.

**Multi-class Logistic Regression (Softmax Regression):**

For multi-class problems (more than two classes), the softmax function is used instead of the sigmoid.  The softmax function outputs a probability distribution over all classes.

`p(y=i) = e^(zi) / sum(e^(zj))` for all classes `j`

The class with the highest probability is predicted as the output.

**Training:**

The weights `w` and bias `b` are learned during the training process using optimization algorithms like gradient descent to minimize a cost function, such as cross-entropy loss.

**In summary, Logistic Regression in NLP maps text features to probabilities of belonging to different categories, enabling text classification and other NLP tasks.**

## 2) Application scenario

*   **Sentiment Analysis:** Classifying movie reviews or product reviews as positive or negative.  Features could be word unigrams, bigrams, and/or TF-IDF scores.

*   **Spam Detection:** Identifying emails as spam or not spam. Features might include the presence of certain keywords ("free," "discount"), the sender's domain, and the email's structure.

*   **Topic Classification:** Categorizing news articles into different topics (e.g., sports, politics, technology). Features can be TF-IDF of the words appearing in the articles.

*   **Intent Detection:** Identifying the user's intention in a chatbot conversation (e.g., booking a flight, ordering food).  Features might include words and phrases related to booking, ordering, or other intents.

*   **Language Detection:** Identifying the language of a text document. Features could include the frequency of specific character n-grams.

## 3) Python method (if possible)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data (replace with your actual data)
documents = [
    "This is a positive review. I loved it!",
    "This movie was terrible. I hated it.",
    "The product is amazing. Highly recommended.",
    "This is a bad experience. Do not buy.",
    "Great service and fast delivery.",
    "Extremely disappointed with the quality."
]
labels = [1, 0, 1, 0, 1, 0]  # 1 for positive, 0 for negative

# 1. Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(documents)

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 3. Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Make predictions on the test set
predictions = model.predict(X_test)

# 5. Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# Example usage: predict sentiment of a new sentence
new_sentence = ["This is an okay product."]
new_features = vectorizer.transform(new_sentence)
new_prediction = model.predict(new_features)
print(f"Sentiment Prediction: {new_prediction}") # will predict either 0 or 1
```

## 4) Follow-up question

How does Logistic Regression compare to other linear classifiers like Support Vector Machines (SVMs) in terms of performance, training time, and suitability for different NLP tasks, especially when dealing with high-dimensional feature spaces? Also, what are the common techniques to mitigate overfitting when using Logistic Regression in NLP?