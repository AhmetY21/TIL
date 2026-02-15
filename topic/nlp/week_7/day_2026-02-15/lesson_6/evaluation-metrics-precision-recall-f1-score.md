---
title: "Evaluation Metrics: Precision, Recall, F1-Score"
date: "2026-02-15"
week: 7
lesson: 6
slug: "evaluation-metrics-precision-recall-f1-score"
---

# Topic: Evaluation Metrics: Precision, Recall, F1-Score

## 1) Formal definition (what is it, and how can we use it?)

Precision, Recall, and F1-Score are crucial evaluation metrics used in various Natural Language Processing (NLP) tasks, particularly in classification and information retrieval. They assess the performance of a model by comparing its predictions against the actual ground truth labels. Understanding them allows us to choose the best model for a specific problem based on its strengths and weaknesses.

Let's define the key terms first:

*   **True Positive (TP):** The model correctly predicted the positive class.
*   **True Negative (TN):** The model correctly predicted the negative class.
*   **False Positive (FP):** The model incorrectly predicted the positive class (also known as Type I error).
*   **False Negative (FN):** The model incorrectly predicted the negative class (also known as Type II error).

Now, we can define the metrics:

*   **Precision:**  Represents the accuracy of positive predictions.  It answers the question: "Of all the instances the model predicted as positive, how many were actually positive?"

    *   Formula:  `Precision = TP / (TP + FP)`

    *   High precision means the model makes very few false positive errors.  It's "precise" in its positive predictions.

*   **Recall (Sensitivity or True Positive Rate):**  Represents the ability of the model to find all the positive instances.  It answers the question: "Of all the actual positive instances, how many did the model correctly identify?"

    *   Formula: `Recall = TP / (TP + FN)`

    *   High recall means the model misses very few actual positive instances. It's good at "recalling" the positives.

*   **F1-Score:**  The harmonic mean of precision and recall. It provides a single score that balances both precision and recall. It's especially useful when you want to find a balance between the two metrics.

    *   Formula: `F1-Score = 2 * (Precision * Recall) / (Precision + Recall)`

    *   A high F1-Score indicates a good balance between precision and recall.  The harmonic mean gives more weight to lower values. Thus, a model with high precision and low recall (or vice-versa) will have a lower F1-Score than a model with relatively balanced precision and recall.

We use these metrics to:

*   **Evaluate model performance:**  Quantify how well a model is performing on a specific task.
*   **Compare different models:** Determine which model is best suited for a given task.
*   **Tune model parameters:**  Optimize model performance by adjusting parameters based on the evaluation metrics.
*   **Understand model biases:** Identify potential biases in the model's predictions based on imbalances in precision and recall.

## 2) Application scenario

Consider a spam email classifier.

*   **Scenario 1:** High Precision, Low Recall: The classifier is very conservative and only flags emails as spam if it's absolutely certain. It correctly identifies almost all emails it flags as spam (high precision), but it misses a lot of actual spam emails (low recall).  This is preferable if you absolutely cannot afford to mark a legitimate email as spam.

*   **Scenario 2:** Low Precision, High Recall: The classifier is very aggressive and flags any suspicious email as spam. It catches almost all spam emails (high recall), but it also incorrectly flags some legitimate emails as spam (low precision).  This is preferable if you want to ensure you catch all spam, even if it means some legitimate emails are filtered.

*   **Scenario 3:** Balanced Precision and Recall: The classifier strikes a balance between catching spam and avoiding false positives. This provides a generally good performance and is often the goal.

In a medical diagnosis scenario (e.g., detecting a rare disease), high recall is often more important than high precision. Missing a case of the disease (false negative) is much more dangerous than incorrectly diagnosing someone who is healthy (false positive). Therefore, you'd prioritize a model with high recall, even if it comes at the expense of lower precision.

## 3) Python method (if possible)

```python
from sklearn.metrics import precision_score, recall_score, f1_score

# Example usage:
y_true = [0, 1, 0, 0, 1, 1, 0, 1]  # Actual labels
y_pred = [0, 1, 1, 0, 0, 1, 0, 1]  # Predicted labels

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

# Calculating these metrics with custom labels requires specifying the `pos_label` parameter
# e.g. if 'yes' represents the positive class
y_true_str = ['no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'yes']
y_pred_str = ['no', 'yes', 'yes', 'no', 'no', 'yes', 'no', 'yes']

precision_str = precision_score(y_true_str, y_pred_str, pos_label='yes')
recall_str = recall_score(y_true_str, y_pred_str, pos_label='yes')
f1_str = f1_score(y_true_str, y_pred_str, pos_label='yes')

print(f"Precision (String Labels): {precision_str}")
print(f"Recall (String Labels): {recall_str}")
print(f"F1-Score (String Labels): {f1_str}")

```

## 4) Follow-up question

How do these metrics behave when dealing with imbalanced datasets (where one class has significantly more instances than the other)?  What are some strategies to address the challenges posed by imbalanced datasets in the context of these evaluation metrics?