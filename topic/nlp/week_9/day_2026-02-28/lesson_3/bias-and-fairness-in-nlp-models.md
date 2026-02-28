---
title: "Bias and Fairness in NLP Models"
date: "2026-02-28"
week: 9
lesson: 3
slug: "bias-and-fairness-in-nlp-models"
---

# Topic: Bias and Fairness in NLP Models

## 1) Formal definition (what is it, and how can we use it?)

Bias and fairness in NLP models refer to the presence of systematic and unfair discrimination within these models against specific groups of individuals. These biases often stem from the data used to train the models, which may reflect societal stereotypes, historical inequalities, or data collection imbalances. This can result in NLP models that perpetuate or amplify existing biases, leading to unfair or discriminatory outcomes for affected groups.

**Formal definition:**

*   **Bias:** Systematic and repeatable errors in a model's predictions that are not due to randomness but rather to underlying assumptions or prejudices learned from the training data.
*   **Fairness:** The absence of systematic discrimination against any individual or group based on protected characteristics such as race, gender, religion, ethnicity, sexual orientation, or disability. Fairness can be defined in several ways mathematically (e.g., demographic parity, equal opportunity, predictive parity), each focusing on a different aspect of equitable treatment. There's no single universally accepted definition of fairness; the appropriate definition depends on the specific application and context.

**How can we use it?**

Understanding bias and fairness is crucial for:

*   **Developing ethical NLP systems:** Ensuring that NLP models do not unfairly discriminate against or disadvantage specific groups.
*   **Improving model performance:** Biased models can make inaccurate predictions for certain groups, leading to overall lower performance.
*   **Building trust and accountability:** Creating NLP systems that are transparent, explainable, and accountable for their decisions.
*   **Meeting legal and regulatory requirements:** Avoiding potential legal and regulatory issues related to discrimination.
*   **Advancing social justice:** Mitigating the harmful effects of biased AI systems on marginalized communities.

## 2) Application scenario

Consider a resume screening application that uses NLP to identify suitable candidates for job openings. The model is trained on historical resume data, which unfortunately contains a skewed representation of male candidates in leadership positions. As a result, the NLP model learns to associate male names and pronouns with leadership qualities and responsibilities.

When a new resume arrives with a female name, the model might underestimate the candidate's leadership experience, even if she possesses comparable or superior qualifications to male candidates in the training data. Consequently, the female candidate may be unfairly overlooked for the job, perpetuating gender inequality in the workplace.

This scenario highlights the risk of bias in NLP applications affecting hiring decisions, potentially leading to discriminatory outcomes based on gender. Similar scenarios can arise with other protected characteristics and impact other applications like loan approvals, credit scoring, or even criminal risk assessment.

## 3) Python method (if possible)

While a single Python method cannot magically remove all biases, there are libraries and techniques we can use to detect and mitigate them. One popular library for fairness auditing is `Fairlearn`.  This example shows how to use Fairlearn to assess the disparity in prediction outcomes for different groups.

```python
from fairlearn.metrics import MetricFrame, count, selection_rate
from sklearn.metrics import accuracy_score, recall_score, precision_score
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Sample data (replace with your actual dataset)
data = {
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100),
    'protected_attribute': np.random.choice(['A', 'B'], size=100), # Sensitive attribute
    'target': np.random.randint(0, 2, size=100) # Binary target variable
}

df = pd.DataFrame(data)

# Separate features, target, and protected attribute
X = df[['feature1', 'feature2']]
y = df['target']
A = df['protected_attribute'] # A represents the sensitive attribute

# Split data into training and testing sets
X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(X, y, A, test_size=0.3, random_state=42)

# Train a simple Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Define metrics
accuracy = accuracy_score
recall = recall_score
precision = precision_score

# Evaluate performance with respect to the sensitive attribute
grouped_on_A = MetricFrame(
    metrics={
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "selection_rate": selection_rate,
        "count": count
    },
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=A_test,
)

print(grouped_on_A.overall)
print("------")
print(grouped_on_A.by_group)
```

This code first defines dummy dataset. Then trains logistic regression model. Using `MetricFrame` from Fairlearn library to calculate metrics like accuracy, recall, and precision, but critically, separates these by group, as defined by the sensitive feature (e.g., `protected_attribute`). The `selection_rate` provides insight into how frequently each group receives a positive prediction. Comparing metrics across groups reveals potential disparities and highlights the need for fairness interventions. The count metric shows the number of datapoints belonging to each group.

This is just a starting point. The `Fairlearn` library also provides tools for mitigation, such as reweighting and post-processing. Other libraries like `Aequitas` can also be used for bias auditing.

## 4) Follow-up question

What are some specific data augmentation techniques that can be used to mitigate bias related to under-represented groups in the training data for NLP models? For example, what techniques can be used to generate synthetic examples of sentences written by people belonging to the under-represented group to improve the model’s performance for this group?