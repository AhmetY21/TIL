---
title: "Confusion Matrix and Error Analysis"
date: "2026-02-16"
week: 8
lesson: 1
slug: "confusion-matrix-and-error-analysis"
---

# Topic: Confusion Matrix and Error Analysis

## 1) Formal definition (what is it, and how can we use it?)

A **confusion matrix** is a table that summarizes the performance of a classification model by showing the counts of correct and incorrect predictions. It visualizes the performance of an algorithm and highlights its strengths and weaknesses with respect to different classes.  It is often used when evaluating supervised learning problems.

Formally, for a binary classification problem, the confusion matrix is a 2x2 matrix:

|                    | Predicted Positive | Predicted Negative |
|--------------------|--------------------|--------------------|
| **Actual Positive**  | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

*   **True Positive (TP):** The model correctly predicted the positive class.
*   **True Negative (TN):** The model correctly predicted the negative class.
*   **False Positive (FP):** The model incorrectly predicted the positive class (Type I error). Also called a *Type I Error*.
*   **False Negative (FN):** The model incorrectly predicted the negative class (Type II error). Also called a *Type II Error*.

For multi-class classification, the confusion matrix becomes an NxN matrix, where N is the number of classes. Each cell (i, j) represents the number of instances that belong to the actual class i and were predicted as class j.

**How to use it:**

The confusion matrix provides insights into the following:

*   **Accuracy:** Overall proportion of correct predictions.  Calculated as (TP + TN) / (TP + TN + FP + FN).
*   **Precision:** Proportion of correctly predicted positives out of all predicted positives. Calculated as TP / (TP + FP). High precision means low false positive rate.
*   **Recall (Sensitivity):** Proportion of correctly predicted positives out of all actual positives. Calculated as TP / (TP + FN). High recall means low false negative rate.
*   **F1-Score:** The harmonic mean of precision and recall. Useful when you want to balance precision and recall. Calculated as 2 * (Precision * Recall) / (Precision + Recall).
*   **Error Analysis:**  By examining the cells with high FP or FN counts, we can identify classes that are frequently confused with each other. This helps us understand the types of errors the model is making and prioritize improvements.  It can also help you identify where more data is needed.

**Error Analysis** goes beyond just looking at the numbers in the confusion matrix. It involves qualitatively analyzing the specific examples that the model got wrong. This might involve looking at the input text, features used by the model, and the model's predicted probability for each class. Error analysis helps us understand *why* the model is making those errors. For example, are there systematic issues with how the model handles specific words, phrases, or sentence structures? Are there biases in the training data that are causing the model to perform poorly on certain subsets of the data?

## 2) Application scenario

**Scenario:** Sentiment Analysis of Movie Reviews

Let's say you've built a sentiment analysis model to classify movie reviews as either "Positive" or "Negative". After training the model, you want to evaluate its performance. You can use a confusion matrix to see how well it's classifying the reviews.

**Example:**

Suppose you have 1000 movie reviews in your test set, with 500 positive and 500 negative reviews. Your model makes the following predictions:

|                    | Predicted Positive | Predicted Negative |
|--------------------|--------------------|--------------------|
| **Actual Positive**  | 400 (TP)           | 100 (FN)           |
| **Actual Negative** | 50 (FP)            | 450 (TN)           |

From this confusion matrix, you can calculate:

*   **Accuracy:** (400 + 450) / 1000 = 0.85 (85%)
*   **Precision:** 400 / (400 + 50) = 0.89 (89%) for Positive class
*   **Recall:** 400 / (400 + 100) = 0.80 (80%) for Positive class
*   **F1-Score:** 2 * (0.89 * 0.80) / (0.89 + 0.80) = 0.84 for Positive class

**Error Analysis:**

Looking at the 100 False Negatives (reviews that were actually positive but predicted as negative), you might find that many of these reviews use sarcasm, irony, or subtle positive language that the model struggles to detect. Similarly, analyzing the 50 False Positives (reviews that were actually negative but predicted as positive) might reveal that they contain positive keywords that mislead the model despite the overall negative sentiment. You could also look for edge cases that are too difficult to classify with available data. Identifying these patterns guides you on how to improve the model, such as adding more features, improving the training data, or refining the model architecture. For example, one might add more negative examples or use contextual embeddings or more advanced sequence models to better handle sarcasm.

## 3) Python method (if possible)

We can use `scikit-learn` to generate a confusion matrix and calculate performance metrics.

```python
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Actual labels
y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])  # 1 = Positive, 0 = Negative

# Predicted labels
y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0])

# Generate the confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# Generate a classification report (includes precision, recall, f1-score, support)
report = classification_report(y_true, y_pred)
print("\nClassification Report:")
print(report)

# Example of Error Analysis - would usually involve looking at the actual data
#  associated with the errors

# Example:  Indices where model incorrectly predicted the positive label
fp_indices = np.where((y_true == 0) & (y_pred == 1))[0]
print(f"\nIndices of False Positives: {fp_indices}")

# Example: Indices where model incorrectly predicted the negative label
fn_indices = np.where((y_true == 1) & (y_pred == 0))[0]
print(f"Indices of False Negatives: {fn_indices}")

# Assuming we have the original text data in a list called 'data'
# You can then inspect the data corresponding to these indices:
# data = ["review text 1", "review text 2", ...]
# for index in fp_indices:
#   print(f"False Positive Review: {data[index]}")
# for index in fn_indices:
#   print(f"False Negative Review: {data[index]}")
```

This code first defines the true and predicted labels. Then, it uses `confusion_matrix` to compute the confusion matrix. The `classification_report` provides a more detailed summary of the model's performance. The added code at the end allows the specific indices of false positives and false negatives to be found, allowing for manual inspection of those items and enabling in depth error analysis.

## 4) Follow-up question

How can we use confusion matrices to compare the performance of different machine learning models on the same dataset, and what metrics should we prioritize based on the specific application (e.g., medical diagnosis vs. spam detection)?