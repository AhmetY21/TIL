---
title: "Support Vector Machines (SVM) in Text Classification"
date: "2026-03-05"
week: 10
lesson: 3
slug: "support-vector-machines-svm-in-text-classification"
---

# Topic: Support Vector Machines (SVM) in Text Classification

## 1) Formal definition (what is it, and how can we use it?)

Support Vector Machines (SVMs) are a powerful and versatile machine learning algorithm that can be effectively used for text classification. At its core, an SVM aims to find the optimal hyperplane that separates different classes of data points with the largest possible margin.  In the context of text classification, each data point represents a document (e.g., a news article, a tweet, a review) and is represented by a feature vector. This feature vector captures the characteristics of the text, such as the frequency of certain words, the presence of specific phrases, or other relevant linguistic features.

Here's a breakdown of how SVM works for text classification:

*   **Feature Extraction:**  The raw text is transformed into a numerical feature vector. Common techniques include:
    *   **Bag of Words (BoW):** Represents a document as a collection of words and their frequencies.
    *   **Term Frequency-Inverse Document Frequency (TF-IDF):**  Assigns weights to words based on their frequency in the document and their rarity across the entire corpus.
    *   **Word Embeddings (e.g., Word2Vec, GloVe, fastText):** Represent words as dense vectors, capturing semantic relationships between words.
*   **Model Training:** The SVM algorithm learns a hyperplane that best separates the text documents into different classes based on their feature vectors.  The algorithm identifies the "support vectors," which are the data points closest to the hyperplane. These support vectors play a crucial role in defining the decision boundary.
*   **Classification:**  When a new, unseen document needs to be classified, it is first transformed into a feature vector using the same feature extraction method. Then, the SVM model uses the learned hyperplane to determine which side of the plane the document's feature vector falls on, assigning it to the corresponding class.

The "margin" is the distance between the hyperplane and the closest data points from each class (the support vectors).  A larger margin generally leads to better generalization and robustness to noisy data.

SVMs often use kernel functions to map the input data into a higher-dimensional space, making it possible to find a linear hyperplane even when the data is not linearly separable in the original space. Common kernel functions include:

*   **Linear Kernel:** Suitable for linearly separable data.
*   **Polynomial Kernel:**  Can model non-linear relationships.
*   **Radial Basis Function (RBF) Kernel:**  A popular choice for complex, non-linear data.
*   **Sigmoid Kernel:** Similar to a neural network activation function.

The choice of kernel and its parameters (e.g., gamma for RBF) is crucial for SVM performance and often requires careful tuning.

## 2) Application scenario

SVMs are well-suited for various text classification tasks, including:

*   **Sentiment Analysis:** Classifying text as positive, negative, or neutral (e.g., analyzing customer reviews or social media posts).
*   **Spam Detection:** Identifying emails or messages as spam or not spam.
*   **Topic Categorization:** Assigning documents to predefined categories (e.g., classifying news articles into topics like sports, politics, or technology).
*   **Authorship Attribution:** Determining the author of a text based on their writing style.
*   **Language Detection:** Identifying the language of a text document.
*   **Intent Classification:** Determining the user's intent from a text query (e.g., in chatbot applications).

In a sentiment analysis scenario, you might collect a dataset of customer reviews labeled as either "positive" or "negative." You would then:

1.  Preprocess the text data (e.g., removing punctuation, converting to lowercase).
2.  Extract features using TF-IDF.
3.  Train an SVM model on the labeled data.
4.  Use the trained model to predict the sentiment of new, unseen customer reviews.

## 3) Python method (if possible)

```python
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data (replace with your actual data)
documents = [
    "This is a great movie. I loved it!",
    "The acting was terrible. I hated it.",
    "A decent film, but nothing special.",
    "Absolutely amazing, a must-see!",
    "Waste of time, avoid this movie.",
    "The product is excellent and works as intended.",
    "Very bad experience. Will never buy this again.",
    "Good quality product, highly recommended."
]
labels = ['positive', 'negative', 'neutral', 'positive', 'negative', 'positive', 'negative', 'positive']

# 1. Feature Extraction (TF-IDF)
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(documents)

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 3. Create and train the SVM model
#  Using a linear kernel for simplicity. RBF kernel is a good alternative, but requires gamma tuning
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, y_train)

# 4. Make predictions on the test set
predictions = classifier.predict(X_test)

# 5. Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Example of predicting sentiment of new text
new_document = ["This product is awesome!"]
new_features = vectorizer.transform(new_document)
new_prediction = classifier.predict(new_features)
print("Predicted sentiment for:", new_document[0], "is", new_prediction[0])
```

## 4) Follow-up question

How does the choice of kernel function and its parameters (e.g., gamma in RBF kernel) affect the performance of an SVM classifier for text data, and what strategies can be used to optimize these choices?