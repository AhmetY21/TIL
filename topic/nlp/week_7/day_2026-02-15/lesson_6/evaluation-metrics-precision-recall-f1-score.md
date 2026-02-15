---
title: "Evaluation Metrics: Precision, Recall, F1-Score"
date: "2026-02-15"
week: 7
lesson: 6
slug: "evaluation-metrics-precision-recall-f1-score"
---

# Topic: Evaluation Metrics: Precision, Recall, F1-Score

## 1) Formal definition (what is it, and how can we use it?)

Precision, Recall, and F1-Score are evaluation metrics used to assess the performance of classification models, particularly in tasks like information retrieval, machine learning, and natural language processing. They are especially useful when dealing with imbalanced datasets where simply measuring accuracy can be misleading. These metrics focus on the performance of positive predictions (i.e., cases where the model predicts something belongs to a specific class, and we want to know how correct that prediction is).

Let's define the terms that form the basis of these metrics:

*   **True Positives (TP):** The number of instances correctly predicted as positive (belonging to the class).
*   **False Positives (FP):** The number of instances incorrectly predicted as positive (not belonging to the class, but predicted as belonging to it). Also known as Type I error.
*   **False Negatives (FN):** The number of instances incorrectly predicted as negative (belonging to the class, but predicted as *not* belonging to it). Also known as Type II error.
*   **True Negatives (TN):** The number of instances correctly predicted as negative (not belonging to the class).

Using these, we can define Precision, Recall, and F1-Score as follows:

*   **Precision:**  Measures the accuracy of positive predictions.  It answers the question: "Of all the instances the model *predicted* as positive, how many were actually positive?"
    *   Formula:  `Precision = TP / (TP + FP)`

*   **Recall:** Measures the ability of the model to find all the positive instances.  It answers the question: "Of all the instances that *actually* belong to the positive class, how many did the model correctly identify?"
    *   Formula: `Recall = TP / (TP + FN)`

*   **F1-Score:** The harmonic mean of precision and recall. It provides a single score that balances both precision and recall.  It's useful when you want to find a balance between the two.
    *   Formula: `F1-Score = 2 * (Precision * Recall) / (Precision + Recall)`

We use these metrics to:

*   **Evaluate model performance:** Compare different models and choose the one that performs best for a given task, especially with imbalanced data.
*   **Tune model parameters:** Adjust parameters to optimize for precision, recall, or a balance between the two, depending on the application.
*   **Understand model behavior:** Diagnose issues like high false positives or false negatives, which can inform improvements to the model or data.

## 2) Application scenario

Imagine you are building a spam email classifier.  The positive class is "spam", and the negative class is "not spam" (ham).

*   A *high precision* means that when your classifier flags an email as spam, it's very likely to be actual spam.  A high precision is important if you *really* don't want to accidentally classify a legitimate email as spam (because that's very annoying for the user).  In this case, a false positive is worse than a false negative.

*   A *high recall* means that your classifier is good at catching most of the spam emails. A high recall is important if you *really* want to filter out as much spam as possible, even if it means occasionally misclassifying a legitimate email as spam. In this case, a false negative is worse than a false positive.

*   The *F1-score* provides a balance between precision and recall. If you want a classifier that performs well on both fronts (i.e., catches most spam without misclassifying too many legitimate emails), the F1-score is a good metric to optimize.

Another scenario might be detecting a rare disease. In this case, a high recall is critical because missing a true case (false negative) is very detrimental.

## 3) Python method (if possible)

The `sklearn.metrics` module in scikit-learn provides functions for calculating precision, recall, and F1-score.

```python
from sklearn.metrics import precision_score, recall_score, f1_score

# Example:
y_true = [0, 1, 0, 0, 1, 1, 0]  # Actual labels (0 = negative, 1 = positive)
y_pred = [0, 1, 1, 0, 0, 1, 0]  # Predicted labels

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

# Example using pos_label parameter (useful if the positive class is not '1')
y_true_string = ['no', 'yes', 'no', 'no', 'yes', 'yes', 'no']
y_pred_string = ['no', 'yes', 'yes', 'no', 'no', 'yes', 'no']

precision_yes = precision_score(y_true_string, y_pred_string, pos_label='yes')
recall_yes = recall_score(y_true_string, y_pred_string, pos_label='yes')
f1_yes = f1_score(y_true_string, y_pred_string, pos_label='yes')

print(f"Precision (yes): {precision_yes}")
print(f"Recall (yes): {recall_yes}")
print(f"F1-Score (yes): {f1_yes}")


# Support for different averaging methods (useful for multi-class classification)
from sklearn.metrics import precision_score, recall_score, f1_score

y_true_multi = [0, 1, 2, 0, 1, 2]
y_pred_multi = [0, 2, 1, 0, 0, 1]

precision_micro = precision_score(y_true_multi, y_pred_multi, average='micro')
recall_micro = recall_score(y_true_multi, y_pred_multi, average='micro')
f1_micro = f1_score(y_true_multi, y_pred_multi, average='micro')

precision_macro = precision_score(y_true_multi, y_pred_multi, average='macro')
recall_macro = recall_score(y_true_multi, y_pred_multi, average='macro')
f1_macro = f1_score(y_true_multi, y_pred_multi, average='macro')

print(f"Micro-averaged Precision: {precision_micro}")
print(f"Micro-averaged Recall: {recall_micro}")
print(f"Micro-averaged F1-Score: {f1_micro}")

print(f"Macro-averaged Precision: {precision_macro}")
print(f"Macro-averaged Recall: {recall_macro}")
print(f"Macro-averaged F1-Score: {f1_macro}")
```

## 4) Follow-up question

How are precision and recall related to the Receiver Operating Characteristic (ROC) curve and Area Under the Curve (AUC) score, and when would you prefer to use ROC/AUC over precision/recall/F1-score, or vice-versa?