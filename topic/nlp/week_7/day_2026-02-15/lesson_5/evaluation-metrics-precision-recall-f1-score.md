---
title: "Evaluation Metrics: Precision, Recall, F1-Score"
date: "2026-02-15"
week: 7
lesson: 5
slug: "evaluation-metrics-precision-recall-f1-score"
---

# Topic: Evaluation Metrics: Precision, Recall, F1-Score

## 1) Formal definition (what is it, and how can we use it?)

These three metrics are fundamental for evaluating the performance of classification models, especially in scenarios where the class distribution is imbalanced. They focus on the accuracy of positive predictions and the model's ability to identify all actual positive instances.

*   **Precision:**  Precision measures the accuracy of the positive predictions made by the model. It answers the question: "Out of all the instances the model *predicted* as positive, how many were *actually* positive?".
    *   Formula:  `Precision = True Positives / (True Positives + False Positives)`
    *   A high precision indicates that the model is good at avoiding false positive errors.  It means that when the model predicts something is positive, it's likely to be correct.

*   **Recall (Sensitivity or True Positive Rate):** Recall measures the completeness of the positive predictions.  It answers the question: "Out of all the instances that were *actually* positive, how many did the model *correctly* identify as positive?".
    *   Formula: `Recall = True Positives / (True Positives + False Negatives)`
    *   A high recall indicates that the model is good at finding most of the positive instances.  It means the model doesn't miss many actual positive cases.

*   **F1-Score:** The F1-score is the harmonic mean of precision and recall. It provides a single score that balances both precision and recall.
    *   Formula: `F1-Score = 2 * (Precision * Recall) / (Precision + Recall)`
    *   The F1-score is useful when you want to find a balance between precision and recall.  It is especially helpful when the costs of false positives and false negatives are different but difficult to quantify precisely.
    *   Harmonic mean gives more weight to lower values. Therefore, a high F1-score ensures both precision and recall are reasonably high.

These metrics are used to understand how well a classification model is performing with respect to identifying positive cases, particularly important when dealing with imbalanced datasets.

## 2) Application scenario

Imagine you're building a spam email filter.

*   **Precision:** High precision would mean that when the filter flags an email as spam, it's very likely to actually *be* spam.  Fewer legitimate emails will be incorrectly classified as spam (false positives). This is important because you don't want to miss important emails.

*   **Recall:** High recall would mean that the filter catches *most* of the spam emails. Very few spam emails would slip through the filter and end up in your inbox (false negatives). This is important to avoid being bombarded with unwanted emails.

*   **F1-Score:** The F1-score helps balance these two concerns. If you prioritize precision (avoiding misclassifying legitimate emails as spam), you might sacrifice recall (letting some spam through). Conversely, prioritizing recall (catching all spam) might lead to lower precision (flagging legitimate emails as spam). The F1-score helps you choose a model that strikes a good balance between these two.  You want a high F1-score to minimize both missed spam and misclassified legitimate emails.

Other examples include: medical diagnosis (identifying diseases), fraud detection, and information retrieval (identifying relevant documents).

## 3) Python method (if possible)

```python
from sklearn.metrics import precision_score, recall_score, f1_score

# Example predictions and ground truth labels
y_true = [0, 1, 1, 0, 1, 0, 0, 1, 0, 1] # Actual labels (0: negative, 1: positive)
y_pred = [0, 1, 0, 0, 1, 1, 0, 0, 1, 1] # Predicted labels

# Calculate precision
precision = precision_score(y_true, y_pred)
print(f"Precision: {precision}")

# Calculate recall
recall = recall_score(y_true, y_pred)
print(f"Recall: {recall}")

# Calculate F1-score
f1 = f1_score(y_true, y_pred)
print(f"F1-Score: {f1}")


# Example using different averaging methods for multiclass/multilabel problems
y_true_multiclass = [0, 1, 2, 0, 1, 2]
y_pred_multiclass = [0, 2, 1, 0, 0, 2]

precision_micro = precision_score(y_true_multiclass, y_pred_multiclass, average='micro')
print(f"Micro-averaged Precision: {precision_micro}") #Global True positives / Global number of predicted positives

precision_macro = precision_score(y_true_multiclass, y_pred_multiclass, average='macro') #average precision of each class
print(f"Macro-averaged Precision: {precision_macro}")

precision_weighted = precision_score(y_true_multiclass, y_pred_multiclass, average='weighted') # weighted average precision of each class
print(f"Weighted-averaged Precision: {precision_weighted}")


from sklearn.metrics import classification_report

# Generate a comprehensive classification report
report = classification_report(y_true, y_pred)
print(report)
```

## 4) Follow-up question

How do these metrics (precision, recall, F1-score) relate to the concept of a "confusion matrix," and how can analyzing a confusion matrix provide further insights into model performance beyond just these three metrics?