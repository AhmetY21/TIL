---
title: "Evaluation Metrics: Precision, Recall, F1-Score"
date: "2026-03-05"
week: 10
lesson: 5
slug: "evaluation-metrics-precision-recall-f1-score"
---

# Topic: Evaluation Metrics: Precision, Recall, F1-Score

## 1) Formal definition (what is it, and how can we use it?)

Precision, Recall, and F1-score are evaluation metrics used to assess the performance of classification models, particularly in binary or multi-class classification problems. They focus on the accuracy of the model's positive predictions.  These metrics are especially important when dealing with imbalanced datasets where accuracy alone can be misleading.

**Definitions:**

*   **True Positive (TP):** The model correctly predicts the positive class.
*   **False Positive (FP):** The model incorrectly predicts the positive class when the actual class is negative.
*   **True Negative (TN):** The model correctly predicts the negative class.
*   **False Negative (FN):** The model incorrectly predicts the negative class when the actual class is positive.

Given these, we can define the metrics:

*   **Precision:**  Out of all the instances the model predicted as positive, what proportion were actually positive?  It measures the accuracy of positive predictions.

    `Precision = TP / (TP + FP)`

*   **Recall (Sensitivity or True Positive Rate):** Out of all the actual positive instances, what proportion did the model correctly identify?  It measures the ability of the model to find all the positive instances.

    `Recall = TP / (TP + FN)`

*   **F1-Score:** The harmonic mean of precision and recall.  It provides a single score that balances both precision and recall. It's particularly useful when you want a metric that considers both false positives and false negatives.

    `F1-Score = 2 * (Precision * Recall) / (Precision + Recall)`

**How to use them:**

These metrics help us understand the trade-offs between minimizing false positives (high precision) and minimizing false negatives (high recall).

*   **High Precision, Low Recall:** The model is very good at avoiding false positives but misses many actual positives. This might be acceptable if the cost of a false positive is very high.
*   **Low Precision, High Recall:** The model identifies most of the actual positives but has a high rate of false positives. This might be acceptable if the cost of a false negative is very high.
*   **High Precision, High Recall:** The ideal scenario; the model is accurate at predicting positives and identifies most of the actual positives.
*   **Low Precision, Low Recall:** The model performs poorly in both predicting positives accurately and identifying the actual positives.

The F1-score provides a single metric to balance these concerns, but the optimal choice of metric depends on the specific application. For example, in spam detection, high precision is crucial to avoid incorrectly classifying legitimate emails as spam. In medical diagnosis, high recall is vital to avoid missing cases of a disease.

## 2) Application scenario

Consider a model designed to detect fraudulent credit card transactions.

*   **Positive Class:** Fraudulent transaction
*   **Negative Class:** Legitimate transaction

In this scenario:

*   A **False Positive** (FP) means a legitimate transaction is flagged as fraudulent. This inconveniences the customer, who might have their card temporarily blocked.
*   A **False Negative** (FN) means a fraudulent transaction is missed. This results in financial loss.

The relative importance of precision and recall depends on the business priorities. If the bank prioritizes customer satisfaction and wants to minimize the inconvenience caused by false positives, they would aim for high precision. However, if the bank prioritizes minimizing financial loss and wants to catch as many fraudulent transactions as possible, they would aim for high recall. The F1-score could be used to balance these competing goals. A high F1-score indicates a good balance between correctly identifying fraudulent transactions and minimizing false alarms.

## 3) Python method (if possible)

```python
from sklearn.metrics import precision_score, recall_score, f1_score

# Example usage:
y_true = [0, 1, 1, 0, 1, 0]  # Actual class labels
y_pred = [0, 1, 0, 0, 0, 1]  # Predicted class labels

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

# Handling multi-class classification:
# You might need to specify the 'average' parameter. Common options:
# - 'micro': Calculate metrics globally by counting the total true positives, false negatives, and false positives.
# - 'macro': Calculate metrics for each label and find their unweighted mean. This does not take label imbalance into account.
# - 'weighted': Calculate metrics for each label and find their average, weighted by support (the number of true instances for each label). This alters 'macro' to account for label imbalance.
# - 'binary': Only report results for the class specified by pos_label. This is applicable only if labels specified are to be computed on and ignore other labels (i.e. a binary evaluation task).
# The default setting varies depending on scikit-learn version, but it's generally wise to specify it explicitly.

y_true_multi = [0, 1, 2, 0, 1, 2]
y_pred_multi = [0, 2, 1, 0, 0, 2]

precision_micro = precision_score(y_true_multi, y_pred_multi, average='micro')
recall_macro = recall_score(y_true_multi, y_pred_multi, average='macro')
f1_weighted = f1_score(y_true_multi, y_pred_multi, average='weighted')


print(f"Multi-class Precision (micro): {precision_micro}")
print(f"Multi-class Recall (macro): {recall_macro}")
print(f"Multi-class F1-Score (weighted): {f1_weighted}")

```

## 4) Follow-up question

How do these metrics relate to the Receiver Operating Characteristic (ROC) curve and the Area Under the Curve (AUC)?  When is it more appropriate to use ROC/AUC instead of precision, recall, and F1-score?