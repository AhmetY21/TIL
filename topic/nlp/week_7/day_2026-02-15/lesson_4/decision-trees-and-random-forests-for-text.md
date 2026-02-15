---
title: "Decision Trees and Random Forests for Text"
date: "2026-02-15"
week: 7
lesson: 4
slug: "decision-trees-and-random-forests-for-text"
---

# Topic: Decision Trees and Random Forests for Text

## 1) Formal definition (what is it, and how can we use it?)

**Decision Trees for Text:** A decision tree is a supervised learning algorithm that creates a tree-like structure to classify or predict a target variable based on input features. In the context of text, the input features are typically representations of the text, such as Term Frequency-Inverse Document Frequency (TF-IDF) values, word embeddings, or presence/absence of specific keywords.  The tree is built by recursively partitioning the data based on the feature that best splits the data according to the target variable (e.g., document category, sentiment).  Each internal node in the tree represents a test on a feature (e.g., "Is TF-IDF of 'cat' > 0.5?"), and each branch represents the outcome of the test.  Leaf nodes represent the predicted class or value.

**How we can use it:**
*   **Text Classification:**  Predict the category of a document (e.g., spam/not spam, sports/politics).
*   **Sentiment Analysis:** Determine the sentiment expressed in a piece of text (e.g., positive, negative, neutral).
*   **Topic Modeling (Less Common):** While not a primary topic modeling technique, decision trees can be used to assign documents to predefined topics based on keywords or feature combinations.

**Random Forests for Text:** A random forest is an ensemble learning method that builds multiple decision trees on different subsets of the training data and using a random subset of features. The final prediction is made by aggregating the predictions of all individual trees (e.g., through majority voting for classification or averaging for regression). This helps to reduce overfitting and improve the generalization performance compared to a single decision tree.

**How we can use it:** Essentially the same applications as decision trees, but often with better accuracy and robustness.

*   **Text Classification:** Predict the category of a document.
*   **Sentiment Analysis:** Determine the sentiment expressed in a piece of text.
*   **Spam Detection:** Classify emails or messages as spam or not spam.
*   **Information Retrieval:** Rank documents based on their relevance to a query.

## 2) Application scenario

Let's consider a **Sentiment Analysis** scenario for movie reviews.

**Goal:** Classify movie reviews as either "positive" or "negative".

**Data:** A dataset of movie reviews, where each review is labeled with its sentiment.

**Features:** We'll use TF-IDF to represent the text.  This means each review will be represented by a vector, where each element corresponds to the TF-IDF value of a word in the vocabulary.

**Process:**

1.  **Data Preparation:**  Load the movie review dataset and split it into training and testing sets.  Preprocess the text data by removing punctuation, stop words, and converting to lowercase.
2.  **Feature Extraction:** Calculate the TF-IDF values for each review in both training and testing sets using scikit-learn's `TfidfVectorizer`.
3.  **Model Training:** Train a Random Forest classifier on the training data and the extracted TF-IDF features.
4.  **Model Evaluation:** Evaluate the trained model on the testing data using metrics like accuracy, precision, recall, and F1-score.
5.  **Prediction:** Use the trained model to predict the sentiment of new, unseen movie reviews.

A single decision tree would likely overfit on this dataset. A random forest, by averaging the predictions of many trees trained on different subsets of the data and features, would generally perform better and generalize better to unseen reviews.

## 3) Python method (if possible)

```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Sample data (replace with your actual movie review dataset)
reviews = [
    "This movie was amazing! I loved it.",
    "The acting was terrible. I hated it.",
    "It was okay, nothing special.",
    "A fantastic film, highly recommended!",
    "I was bored and disappointed."
]
labels = ["positive", "negative", "neutral", "positive", "negative"]

# 1. Data Preparation
X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=42) # Corrected typo

# 2. Feature Extraction (TF-IDF)
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 3. Model Training (Random Forest)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42) # You can adjust n_estimators (number of trees)
rf_classifier.fit(X_train_tfidf, y_train)

# 4. Model Evaluation
y_pred = rf_classifier.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))

# 5. Prediction (example)
new_review = ["This movie was surprisingly good!"]
new_review_tfidf = vectorizer.transform(new_review)
prediction = rf_classifier.predict(new_review_tfidf)
print(f"Prediction for new review: {prediction}")
```

## 4) Follow-up question

How do techniques like stemming and lemmatization, which reduce words to their root form, affect the performance of decision trees and random forests when used in conjunction with TF-IDF for text classification?  Specifically, do these techniques consistently improve performance, or does their effectiveness depend on the specific dataset and task? Why?