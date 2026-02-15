---
title: "Support Vector Machines (SVM) in Text Classification"
date: "2026-02-15"
week: 7
lesson: 3
slug: "support-vector-machines-svm-in-text-classification"
---

# Topic: Support Vector Machines (SVM) in Text Classification

## 1) Formal definition (what is it, and how can we use it?)

Support Vector Machines (SVMs) are supervised machine learning models used for classification and regression. In the context of text classification, an SVM aims to find the optimal hyperplane that separates different classes of text documents in a high-dimensional space. Each document is represented as a vector of features (e.g., TF-IDF scores of words). The hyperplane maximizes the margin between the closest data points of different classes (support vectors).

More formally:

*   **Input:** A set of labeled training data, where each document *x<sub>i</sub>* is associated with a class label *y<sub>i</sub>* (e.g., spam/not spam, positive/negative sentiment).  The document *x<sub>i</sub>* is represented as a feature vector.
*   **Goal:** Find a hyperplane *w<sup>T</sup>x + b = 0* that separates the documents into different classes with the largest margin.  *w* is the normal vector to the hyperplane, and *b* is the bias term.
*   **Margin:** The distance between the hyperplane and the closest data points from each class (support vectors).
*   **Optimization:**  The SVM algorithm solves an optimization problem to find the optimal *w* and *b* that maximize the margin while minimizing classification errors.  This often involves solving a quadratic programming problem.
*   **Kernel Trick:** SVMs can use different kernel functions (e.g., linear, polynomial, radial basis function (RBF)) to implicitly map the input data into a higher-dimensional space where it might be linearly separable. This is particularly useful when dealing with non-linear relationships between text features and classes.

We use SVMs for text classification by first representing text documents as feature vectors (e.g., using TF-IDF, word embeddings, or bag-of-words).  We then train an SVM model on the labeled training data.  Finally, we use the trained model to predict the class label of new, unseen text documents.

## 2) Application scenario

Consider the scenario of **spam detection**.  We want to classify incoming emails as either "spam" or "not spam" (ham).  We can represent each email as a feature vector, where each feature could be the frequency of certain words in the email (e.g., "free", "discount", "urgent").

1.  **Data Collection:** Collect a large dataset of labeled emails, where each email is tagged as either "spam" or "ham."
2.  **Feature Extraction:** Extract features from the emails, such as TF-IDF scores of words, presence of certain keywords, sender information, etc.
3.  **Model Training:** Train an SVM model on the labeled data, using a suitable kernel (e.g., linear or RBF).
4.  **Model Evaluation:** Evaluate the performance of the model on a held-out test set.
5.  **Deployment:** Deploy the trained model to filter incoming emails in real-time.

SVMs can also be applied in other text classification tasks such as:

*   **Sentiment analysis:** Classifying movie reviews, product reviews, or social media posts as positive, negative, or neutral.
*   **Topic categorization:** Assigning news articles or research papers to different topics or categories (e.g., sports, politics, technology).
*   **Authorship attribution:** Identifying the author of a text based on its writing style.
*   **Intent classification:** Determining the user's intent from a short text query (e.g., "book a flight to London").

## 3) Python method (if possible)

```python
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data (replace with your actual data)
documents = [
    "This is a positive review. I loved the product!",
    "The movie was terrible. I hated it.",
    "This is another positive comment. Excellent!",
    "The service was awful. I would not recommend it.",
    "The product is okay. Nothing special.",
]
labels = ["positive", "negative", "positive", "negative", "neutral"]

# 1. Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(documents)

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 3. Train the SVM model (linear kernel)
model = svm.SVC(kernel='linear') # You can try other kernels like 'rbf' or 'poly'
model.fit(X_train, y_train)

# 4. Make predictions on the test set
predictions = model.predict(X_test)

# 5. Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# Example of predicting on new data
new_document = ["This is a great product. Highly recommend!"]
new_features = vectorizer.transform(new_document) # Transform the new document using the fitted vectorizer
new_prediction = model.predict(new_features)
print(f"Prediction for new document: {new_prediction}")
```

## 4) Follow-up question

How does the choice of kernel function (e.g., linear, RBF, polynomial) affect the performance of an SVM in text classification, and how can you choose the best kernel for a particular task? Consider factors such as data dimensionality, non-linearity, and computational cost.