---
title: "Decision Trees and Random Forests for Text"
date: "2026-03-05"
week: 10
lesson: 4
slug: "decision-trees-and-random-forests-for-text"
---

# Topic: Decision Trees and Random Forests for Text

## 1) Formal definition (what is it, and how can we use it?)

**Decision Trees:**

A decision tree is a supervised learning algorithm that uses a tree-like structure to make decisions. In the context of text data, a decision tree aims to classify text documents into predefined categories or predict a numerical target based on the text content.  The tree consists of internal nodes representing tests on the features (e.g., presence/absence of a word, frequency of a word, etc.), branches representing the outcomes of those tests, and leaf nodes representing the class label or predicted value.

To build a decision tree for text, we first need to represent the text data numerically. Common techniques include:

*   **Bag of Words (BoW):**  Representing each document as a vector where each element counts the frequency of a particular word in the vocabulary.
*   **TF-IDF (Term Frequency-Inverse Document Frequency):**  Weighing words based on their frequency within a document and their rarity across the entire corpus.
*   **Word Embeddings (e.g., Word2Vec, GloVe):** Mapping words to dense vector representations that capture semantic relationships.

The decision tree algorithm (e.g., CART, ID3, C4.5) then iteratively selects the best feature and split point at each internal node to maximize information gain or minimize impurity.  Information gain measures the reduction in entropy (uncertainty) after splitting on a feature.  Impurity measures (e.g., Gini impurity, entropy) quantify the heterogeneity of class labels within a node.  The splitting process continues until a stopping criterion is met, such as reaching a maximum tree depth, minimum number of samples per leaf, or acceptable impurity level.

**How can we use it?**  Decision trees can be used for text classification (e.g., spam detection, sentiment analysis, topic categorization) and text regression (e.g., predicting the popularity of a news article based on its content).

**Random Forests:**

A random forest is an ensemble learning method that combines multiple decision trees to improve prediction accuracy and reduce overfitting. It operates by constructing a multitude of decision trees during training, each trained on a random subset of the data and a random subset of the features.

The randomness is introduced in two main ways:

*   **Bootstrap Aggregating (Bagging):**  Each tree is trained on a different bootstrap sample of the original data. A bootstrap sample is created by randomly sampling the original dataset with replacement, meaning some instances may be included multiple times in a single sample, while others are excluded.
*   **Random Subspace:** When splitting a node during the construction of each tree, the algorithm considers only a random subset of the available features.

To make a prediction, the random forest aggregates the predictions of all the individual trees. For classification, this is typically done by majority voting (the class predicted by the most trees wins). For regression, the predictions are usually averaged.

**How can we use it?**  Random forests can be used for the same text-based tasks as decision trees (classification, regression), but they typically achieve better performance due to their ensemble nature and reduced risk of overfitting.

## 2) Application scenario

**Application Scenario:** Sentiment Analysis of Movie Reviews

Suppose we have a dataset of movie reviews labeled as either "positive" or "negative".  We want to build a model that can automatically classify new movie reviews based on their sentiment.

1.  **Data Preparation:** We preprocess the reviews by removing stop words (e.g., "the", "a", "is"), punctuation, and converting all text to lowercase.
2.  **Feature Extraction:** We use TF-IDF to represent each review as a vector of term frequencies weighted by their inverse document frequencies.
3.  **Model Training:** We train a random forest model using the TF-IDF vectors as input features and the sentiment labels ("positive" or "negative") as the target variable.
4.  **Model Evaluation:**  We evaluate the model's performance on a held-out test set using metrics like accuracy, precision, recall, and F1-score.
5.  **Prediction:**  Given a new movie review, we preprocess it, extract its TF-IDF vector, and use the trained random forest to predict its sentiment.

In this scenario, the random forest can learn to identify words and phrases that are indicative of positive or negative sentiment. For example, words like "amazing", "excellent", and "enjoyable" might be strong indicators of positive sentiment, while words like "terrible", "awful", and "boring" might be strong indicators of negative sentiment. The random forest's ability to combine multiple decision trees helps it to capture more complex relationships between words and sentiment compared to a single decision tree.

## 3) Python method (if possible)
```python
import nltk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

# Sample data (replace with your actual data)
reviews = [
    "This movie was absolutely fantastic! The acting was superb, and the plot was engaging.",
    "I really enjoyed this film. The story was well-written and the characters were believable.",
    "The movie was okay. Nothing special, but not terrible either.",
    "This was the worst movie I have ever seen. The acting was awful, and the plot made no sense.",
    "I hated this film. It was boring and predictable."
]
labels = ['positive', 'positive', 'neutral', 'negative', 'negative']

# Preprocessing (minimal example - you'd likely do more)
def preprocess_text(text):
    text = text.lower()
    # tokenization with NLTK
    tokens = nltk.word_tokenize(text) # can customize this
    return " ".join(tokens)


processed_reviews = [preprocess_text(review) for review in reviews]


# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(processed_reviews)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)  # Adjust hyperparameters as needed
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
predictions = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_report(y_test, predictions))

# Example prediction on new data
new_review = "The special effects were great, but the story was lacking."
processed_new_review = preprocess_text(new_review)
new_review_features = vectorizer.transform([processed_new_review])  # Use transform, not fit_transform
prediction = rf_classifier.predict(new_review_features)[0]
print(f"Predicted sentiment for the new review: {prediction}")
```

## 4) Follow-up question

How can we handle imbalanced datasets (e.g., many more positive reviews than negative reviews) when using decision trees or random forests for text classification?  What techniques can be employed during preprocessing, model training, and evaluation to mitigate the impact of class imbalance?