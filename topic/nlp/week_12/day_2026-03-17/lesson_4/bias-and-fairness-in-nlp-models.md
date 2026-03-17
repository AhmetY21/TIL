---
title: "Bias and Fairness in NLP Models"
date: "2026-03-17"
week: 12
lesson: 4
slug: "bias-and-fairness-in-nlp-models"
---

# Topic: Bias and Fairness in NLP Models

## 1) Formal definition (what is it, and how can we use it?)

Bias and Fairness in NLP models refers to the phenomenon where these models exhibit systematic and unfair discrimination towards certain groups of people based on sensitive attributes such as gender, race, religion, sexual orientation, or disability. This bias arises because NLP models learn from vast amounts of text data, which often reflects existing societal biases.

**What is it?**

*   **Bias:** A systematic and consistent skewing of a model's output towards a particular outcome, disproportionately affecting certain groups. This skewing is not due to actual differences in the underlying data but rather reflects prejudices learned from the training data or inherent limitations of the model architecture.
*   **Fairness:**  A state where the model's performance (e.g., accuracy, error rates) is equitable across different demographic groups.  There isn't a single "correct" definition of fairness; different fairness metrics exist, and the appropriate one depends on the specific application and its ethical considerations.  Common fairness definitions include:

    *   **Statistical Parity:**  Ensuring that the probability of a positive outcome is the same across all groups.  (e.g., same acceptance rate for loan applications across all races).
    *   **Equal Opportunity:**  Ensuring that the true positive rate (TPR) is the same across all groups.  (e.g., same rate of correctly identifying qualified candidates across all genders).
    *   **Predictive Parity:** Ensuring that the positive predictive value (PPV) is the same across all groups. (e.g., same probability that a positive prediction from the model is actually true, across all age groups).

**How can we use it?**

Understanding bias and fairness in NLP allows us to:

*   **Identify and Mitigate Harmful Outcomes:** By detecting bias, we can proactively adjust models to avoid perpetuating stereotypes or discriminatory practices in applications like hiring, loan approvals, criminal justice, and healthcare.
*   **Develop More Ethical AI Systems:** Incorporating fairness considerations into the model development lifecycle promotes responsible AI practices and builds trust in these systems.
*   **Improve Model Performance for All Groups:** Debiasing can sometimes lead to more robust and generalizable models that perform better across diverse populations.
*   **Inform Policy and Regulations:** Research into bias and fairness in NLP can provide valuable insights for policymakers seeking to regulate AI systems and ensure they are used responsibly.

## 2) Application scenario

**Scenario:** Resume Screening

Imagine an NLP model is used to automatically screen resumes for job applications. The model is trained on historical resume data, including successful candidates from the past. If the historical data disproportionately represents one gender or race in certain roles, the model may learn to favor resumes from that group, even if other candidates are equally qualified.

**Consequences of Bias:**

*   **Discrimination:** The model could systematically downrank or reject resumes from underrepresented groups, perpetuating existing inequalities in the workforce.
*   **Missed Opportunities:**  The company could miss out on qualified candidates from diverse backgrounds, hindering innovation and reducing the talent pool.
*   **Legal and Reputational Risks:**  Using a biased resume screening tool could expose the company to legal challenges and damage its reputation.

**Fairness Metric to Consider:**

*   **Equal Opportunity:** We might want to ensure that the *true positive rate* (the proportion of truly qualified candidates who are correctly identified) is the same for all demographic groups. This means that the model should be equally effective at identifying qualified candidates, regardless of their gender, race, etc.

## 3) Python method (if possible)

```python
# Example using Fairlearn library (requires installation: pip install fairlearn)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate, count
from fairlearn.reductions import DemographicParity, ExponentiatedGradient
from fairlearn.datasets import fetch_adult

# Load the Adult dataset (contains demographic information)
data = fetch_adult(as_frame=True)
X = data.data
y = data.target
A = data.data["sex"]  # 'sex' column as our sensitive attribute (e.g., gender)

# Preprocess: Handle missing values and encode categorical features (simplified)
X = X.fillna(X.mean(numeric_only=True))
X = pd.get_dummies(X, columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country'])

# Split data into training and testing sets
X_train, X_test, y_train, A_train, A_test = train_test_split(X, y, A, test_size=0.3, random_state=42)


# 1. Train a Baseline Model (potentially biased)
baseline_model = LogisticRegression(solver='liblinear', random_state=42)
baseline_model.fit(X_train, y_train)
y_pred_baseline = baseline_model.predict(X_test)


# 2. Evaluate Baseline Model and Check for Disparities

def evaluate_fairness(y_true, y_pred, sensitive_attribute):
    """Evaluates fairness metrics using Fairlearn."""
    metric_fns = {
        "overall_accuracy": accuracy_score,
        "selection_rate": selection_rate,
        "count": count
    }
    metric_frame = MetricFrame(metrics=metric_fns,
                               y_true=y_true,
                               y_pred=y_pred,
                               sensitive_features=sensitive_attribute)
    print(metric_frame.overall)
    print("-" * 20)
    print(metric_frame.by_group)


print("Baseline Model Evaluation:")
evaluate_fairness(y_test, y_pred_baseline, A_test)


# 3. Train a Fair Model using Fairlearn's Reductions approach
#    (DemographicParity ensures similar selection rates across groups)
constraint = DemographicParity()
#  ExponentiatedGradient trains multiple models to satisfy the constraint
mitigator = ExponentiatedGradient(estimator=LogisticRegression(solver='liblinear', random_state=42),
                                  constraints=constraint)

mitigator.fit(X_train, y_train, sensitive_features=A_train)
y_pred_fair = mitigator.predict(X_test)


# 4. Evaluate the Fair Model
print("\nFair Model Evaluation:")
evaluate_fairness(y_test, y_pred_fair, A_test)


# Note:  This is a simplified example.  More sophisticated preprocessing,
# hyperparameter tuning, and fairness constraint selection are crucial in real-world applications.
```

**Explanation:**

1.  **Data Loading and Preprocessing:** The code loads the Adult dataset, which includes sensitive attributes like gender ("sex").  It preprocesses the data by handling missing values and converting categorical variables into numerical ones using one-hot encoding.
2.  **Baseline Model:** A standard Logistic Regression model is trained on the data *without* any fairness considerations.
3.  **Fairness Evaluation:**  The `evaluate_fairness` function calculates accuracy, selection rate (proportion of positive predictions), and group counts using the `MetricFrame` from Fairlearn. This helps identify disparities in the baseline model's performance across different demographic groups.
4.  **Fair Model Training:**  Fairlearn's `ExponentiatedGradient` algorithm is used to train a model that satisfies a `DemographicParity` constraint. Demographic Parity aims to ensure that the selection rate (positive prediction rate) is similar across different groups defined by the sensitive attribute.  The `ExponentiatedGradient` algorithm iteratively trains and combines multiple models to achieve this fairness goal.
5.  **Evaluation of Fair Model:**  The `evaluate_fairness` function is called again to assess the performance of the fair model.  You should observe a reduction in disparities in the selection rate compared to the baseline model, potentially at the cost of a small decrease in overall accuracy.

**Important Considerations:**

*   **Fairness Metrics:** The choice of fairness metric (e.g., Statistical Parity, Equal Opportunity, Predictive Parity) depends on the specific application and its ethical implications.
*   **Trade-offs:**  Debiasing can sometimes involve trade-offs between accuracy and fairness. It's essential to carefully consider these trade-offs and choose the approach that best aligns with the application's goals.
*   **Data Quality:**  The quality of the training data is crucial.  Even with debiasing techniques, biased data can still lead to unfair outcomes.

## 4) Follow-up question

How does the choice of a specific fairness metric (e.g., Demographic Parity, Equal Opportunity, Predictive Parity) affect the mitigation strategy and the resulting model's performance in a real-world application, and what factors should be considered when selecting the most appropriate metric?