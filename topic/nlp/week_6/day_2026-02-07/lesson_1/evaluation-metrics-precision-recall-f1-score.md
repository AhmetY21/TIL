## Topic: Evaluation Metrics: Precision, Recall, F1-Score

**1- Provide formal definition, what is it and how can we use it?**

Precision, Recall, and F1-Score are evaluation metrics used to assess the performance of classification models, especially in scenarios where class distribution is imbalanced. They are calculated based on the counts of True Positives (TP), False Positives (FP), and False Negatives (FN).

*   **True Positive (TP):** The model correctly predicts the positive class.
*   **False Positive (FP):** The model incorrectly predicts the positive class when it's actually negative. (Also known as Type I error).
*   **False Negative (FN):** The model incorrectly predicts the negative class when it's actually positive. (Also known as Type II error).
*   **True Negative (TN):** The model correctly predicts the negative class.

**Definitions:**

*   **Precision:**  Out of all the instances predicted as positive, how many were actually positive?  It measures the accuracy of the positive predictions.
    *   Formula:  `Precision = TP / (TP + FP)`
*   **Recall (Sensitivity):** Out of all the actual positive instances, how many were correctly predicted as positive? It measures the model's ability to find all the positive instances.
    *   Formula:  `Recall = TP / (TP + FN)`
*   **F1-Score:** The harmonic mean of precision and recall. It provides a single score that balances both precision and recall. It's useful when you want to find a balance between minimizing false positives and false negatives.
    *   Formula: `F1-Score = 2 * (Precision * Recall) / (Precision + Recall)`

**How can we use it?**

These metrics help us understand different aspects of our model's performance:

*   **High Precision, Low Recall:**  The model is very careful about predicting the positive class, resulting in fewer false positives, but it misses many actual positive instances. This is suitable when the cost of a false positive is very high.
*   **Low Precision, High Recall:** The model identifies most of the actual positive instances but also makes a lot of false positive predictions.  This is suitable when the cost of a false negative is very high.
*   **High Precision, High Recall (High F1-Score):**  The ideal scenario where the model performs well in identifying positive instances without making too many false positive errors.
*   **Low Precision, Low Recall (Low F1-Score):** The model is performing poorly. It's missing positive cases and misclassifying negative cases as positive.

**2- Provide an application scenario**

Consider a spam email detection system.

*   **Positive Class:** Spam email
*   **Negative Class:** Non-spam email (Ham)

*   **High Precision is important:** We want to minimize the number of legitimate (non-spam) emails incorrectly classified as spam (FP).  If a legitimate email is marked as spam, the user might miss important information. The cost of a false positive is high.

*   **High Recall is also important:** We want to correctly identify as many spam emails as possible (TP).  If a spam email is not detected, it can be annoying or even dangerous for the user. The cost of a false negative is also high, but perhaps slightly less so than a false positive.

In this scenario, we would want to use the F1-Score to find a balance between precision and recall, aiming for a model that correctly identifies a large proportion of spam emails without incorrectly classifying too many legitimate emails as spam. A high F1 score would indicate a well-performing spam detection system.

**3- Provide a method to apply in python**

We can use the `sklearn.metrics` module in Python to calculate these metrics.

python
from sklearn.metrics import precision_score, recall_score, f1_score

# Example:
y_true = [0, 1, 0, 1, 0, 0, 1, 1, 0, 1] # Actual labels
y_pred = [0, 1, 1, 1, 0, 0, 0, 1, 1, 1] # Predicted labels

# Calculate precision
precision = precision_score(y_true, y_pred)
print(f"Precision: {precision}")

# Calculate recall
recall = recall_score(y_true, y_pred)
print(f"Recall: {recall}")

# Calculate F1-score
f1 = f1_score(y_true, y_pred)
print(f"F1-Score: {f1}")

# For multi-class classification, you can specify the 'average' parameter:
# 'micro': Calculate metrics globally by counting the total true positives, false negatives and false positives.
# 'macro': Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
# 'weighted': Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.
# 'binary': Only report results for the class specified by pos_label. This is applicable only if targets (y_{true,pred}) contain exactly two classes.

from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))


**Explanation:**

1.  We import the necessary functions from `sklearn.metrics`.
2.  We define `y_true` (actual labels) and `y_pred` (predicted labels).
3.  We call `precision_score`, `recall_score`, and `f1_score` with the true and predicted labels as arguments.
4.  We print the results.
5. The classification_report function provides all three metrics and the support for each class.

**4- Provide a follow up question about that topic**

How does the choice of the `average` parameter in functions like `precision_score`, `recall_score`, and `f1_score` affect the interpretation of the metrics in a multi-class classification problem, and when would you choose one averaging method over another (e.g., 'micro', 'macro', 'weighted')?