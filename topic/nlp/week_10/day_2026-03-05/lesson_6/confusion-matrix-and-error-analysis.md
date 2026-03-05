---
title: "Confusion Matrix and Error Analysis"
date: "2026-03-05"
week: 10
lesson: 6
slug: "confusion-matrix-and-error-analysis"
---

# Topic: Confusion Matrix and Error Analysis

## 1) Formal definition (what is it, and how can we use it?)

A **confusion matrix** is a table that visualizes the performance of a classification model. It summarizes the results of the classification by showing the counts of correct and incorrect predictions, broken down by each class. The matrix's rows represent the actual (true) classes, and the columns represent the predicted classes (or vice versa, depending on the convention used).

Key terms associated with a confusion matrix:

*   **True Positive (TP):** The model correctly predicted the positive class.
*   **True Negative (TN):** The model correctly predicted the negative class.
*   **False Positive (FP):** The model incorrectly predicted the positive class (also known as Type I error).
*   **False Negative (FN):** The model incorrectly predicted the negative class (also known as Type II error).

**How we can use it:**

*   **Evaluate Model Performance:** A confusion matrix provides a more detailed evaluation of a classification model than simple accuracy. It allows us to see not only *how many* predictions were correct, but *which types* of errors the model is making.
*   **Identify Class-Specific Weaknesses:** We can use the matrix to identify classes for which the model performs poorly. For example, a high number of false negatives for a particular class indicates the model struggles to identify instances of that class.
*   **Calculate Performance Metrics:** From the confusion matrix, we can derive various performance metrics such as:
    *   **Accuracy:** (TP + TN) / (TP + TN + FP + FN) - Overall correctness.
    *   **Precision:** TP / (TP + FP) - Of all instances predicted as positive, how many were actually positive?
    *   **Recall (Sensitivity):** TP / (TP + FN) - Of all actual positive instances, how many were correctly predicted?
    *   **F1-score:** 2 * (Precision * Recall) / (Precision + Recall) - Harmonic mean of precision and recall.
    *   **Specificity:** TN / (TN + FP) - Of all actual negative instances, how many were correctly predicted as negative?
*   **Guide Model Improvement:** By analyzing the patterns of errors, we can get insights into how to improve the model. This may involve collecting more data for under-represented classes, adjusting the model's parameters, or trying a different modeling approach.
*   **Error Analysis:** Beyond the numbers in the matrix, **error analysis** is the process of examining the specific examples that the model misclassified. This often involves manually inspecting the input data and model predictions to understand the underlying reasons for the errors. Error analysis can uncover issues like data quality problems, ambiguity in the data, or limitations in the model's features.

## 2) Application scenario

Consider a sentiment analysis model designed to classify movie reviews as either "positive" or "negative". After training and testing the model on a dataset of movie reviews, you obtain the following confusion matrix:

```
                   Predicted Negative | Predicted Positive
Actual Negative     150                 | 20
Actual Positive      30                 | 100
```

From this confusion matrix, we can glean the following information:

*   The model correctly classified 150 negative reviews as negative (TN).
*   The model correctly classified 100 positive reviews as positive (TP).
*   The model incorrectly classified 20 negative reviews as positive (FP).
*   The model incorrectly classified 30 positive reviews as negative (FN).

Using this information:

* We can calculate metrics like precision, recall, and F1-score for each class (positive and negative). For example, precision for positive reviews is 100 / (100 + 20) = 0.833.
* We can identify that the model has more difficulty with positive reviews, as it incorrectly classifies them as negative 30 times, which suggests that the model might be biased toward identifying reviews as negative.
* **Error Analysis:** By manually reviewing the 30 false negative reviews, you might discover that they contain subtle sarcasm, nuanced language, or implicit positive sentiment that the model fails to recognize. Further analysis could reveal common phrases or words used in these reviews that confuse the model. This could prompt you to refine the model's features or training data to better handle these cases. Maybe adding context around certain phrases would improve accuracy.

## 3) Python method (if possible)

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Actual and predicted labels
y_true = np.array([0, 1, 0, 1, 0, 1, 0, 0, 1, 1]) # 0: Negative, 1: Positive
y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 0, 1, 1])

# Generate the confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

# Display the confusion matrix using ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"]) #Display Labels can be modified to fit use-case

disp.plot()
plt.title("Confusion Matrix for Sentiment Analysis")
plt.show()

# Calculating Metrics from the Confusion Matrix (example)
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]
TP = cm[1, 1]

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1_score:.2f}")
```

## 4) Follow-up question

How can you extend error analysis using techniques beyond manual inspection, such as looking at feature importance or using explainable AI (XAI) methods, to gain deeper insights into why a model is making specific errors in an NLP task like text classification?