---
title: "Naive Bayes Classifier for Text"
date: "2026-03-04"
week: 10
lesson: 6
slug: "naive-bayes-classifier-for-text"
---

# Topic: Naive Bayes Classifier for Text

## 1) Formal definition (what is it, and how can we use it?)

The Naive Bayes classifier is a probabilistic machine learning algorithm that's used for classification tasks, particularly effective and popular in text classification.  It's based on applying Bayes' theorem with a strong (naive) assumption of independence between the features.  In the context of text, the features are usually the words in a document. The "naive" part refers to the assumption that the presence of one word in a document is independent of the presence of any other word, given the class label.  This is, of course, almost never true in reality, but surprisingly, the classifier still performs well despite this simplification.

**Bayes' Theorem:**

The foundation of Naive Bayes is Bayes' theorem:

P(C|D) = [P(D|C) * P(C)] / P(D)

Where:

*   P(C|D):  The posterior probability of a class *C* given the document *D*.  This is what we want to calculate - the probability that a document belongs to a specific class.
*   P(D|C):  The likelihood probability of observing document *D* given that it belongs to class *C*. This represents how likely the document is to appear if it belongs to this class.
*   P(C):  The prior probability of class *C*.  This is the probability of a document belonging to class *C* before seeing the document itself.
*   P(D):  The evidence probability of document *D*. This represents the probability of seeing the document regardless of class.  Since it is a constant for a given document across all class comparisons, it is often ignored when comparing probabilities for different classes.

**Applying to Text:**

In text classification, we break down the document *D* into its individual words: D = (word1, word2, ..., wordN).  The naive assumption allows us to simplify P(D|C) as follows:

P(D|C) = P(word1|C) * P(word2|C) * ... * P(wordN|C)

We calculate P(word*i*|C) by counting the number of times word *i* appears in all documents belonging to class *C*, and dividing it by the total number of words in all documents belonging to class *C*.  Smoothing techniques (like Laplace smoothing) are often used to avoid zero probabilities when a word doesn't appear in a particular class during training.

**Usage:**

To classify a new document, we calculate P(C|D) for each possible class *C*. We choose the class with the highest posterior probability.  Therefore, we predict the class C* such that:

C* = argmax_C P(C|D)

We use the trained classifier with prior and conditional probabilities calculated during training to perform classification of new, unseen documents.

## 2) Application scenario

A very common application scenario is **spam email detection**.  We can train a Naive Bayes classifier on a dataset of emails labeled as "spam" or "not spam" ("ham"). The features would be the words in the email's subject and body. The classifier learns the probability of each word appearing in spam emails versus ham emails. When a new email arrives, the classifier calculates the probability that it's spam and the probability that it's ham, based on the words present in the email.  The email is classified as the category with the higher probability.

Other application scenarios include:

*   **Sentiment Analysis:**  Classifying text as positive, negative, or neutral.  For instance, analyzing product reviews.
*   **Document Categorization:** Assigning documents to predefined categories (e.g., sports, politics, technology).
*   **Topic Classification:** Determining the main topic or theme of a piece of text.
*   **Language Detection:** Identifying the language a text is written in.

## 3) Python method (if possible)
```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data (replace with your actual data)
documents = [
    "This is a positive document about happy cats.",
    "This is another positive document filled with joy and kittens.",
    "This is a negative document about sad dogs.",
    "This is a very negative document filled with sorrow and grief.",
    "This is a neutral document about the weather today.",
    "The weather is hot."
]
labels = ['positive', 'positive', 'negative', 'negative', 'neutral', 'neutral']

# 1. Feature Extraction: Convert text to numerical data using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)  # Learn vocabulary and transform documents

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

# 3. Train the Naive Bayes classifier (MultinomialNB is suitable for text)
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# 4. Make predictions on the test set
y_pred = classifier.predict(X_test)

# 5. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Example of predicting a new document
new_document = ["This is a happy day"]
new_X = vectorizer.transform(new_document) # Use the SAME vectorizer fitted on the training data
prediction = classifier.predict(new_X)
print(f"Prediction for new document: {prediction}")

```

**Explanation:**

1.  **`CountVectorizer`**: This converts the text documents into a matrix of token counts.  Each row represents a document, and each column represents a word from the vocabulary. The value in each cell is the number of times that word appears in that document. `fit_transform` both learns the vocabulary and transforms the data. Important: When predicting on new data, use the *same* vectorizer object that was fitted on the training data, so that it uses the same vocabulary.

2.  **`train_test_split`**: This splits the data into training and testing sets to evaluate the performance of the model.

3.  **`MultinomialNB`**: This is a specific type of Naive Bayes classifier that is well-suited for discrete data, such as word counts in text. The `fit` method trains the classifier on the training data.

4.  **`predict`**: This method uses the trained classifier to predict the labels of the test data.

5.  **`accuracy_score`**:  This calculates the accuracy of the model by comparing the predicted labels to the true labels.

## 4) Follow-up question

What are some strategies to improve the performance of a Naive Bayes text classifier, and what are the limitations of the Naive Bayes assumption of feature independence in text classification?