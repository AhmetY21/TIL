---
title: "Doubly Robust Estimation (AIPW / DR Learners)"
date: "2026-02-13"
week: 7
lesson: 2
slug: "doubly-robust-estimation-aipw-dr-learners"
---

# Topic: Doubly Robust Estimation (AIPW / DR Learners)

## 1) Formal definition (what is it, and how can we use it?)

Doubly Robust (DR) estimation, specifically via methods like Augmented Inverse Propensity Weighting (AIPW) or DR Learners, aims to estimate the Average Treatment Effect (ATE) or other causal quantities while being "doubly robust."  This means it provides a consistent estimate if *either* the outcome model *or* the treatment assignment model is correctly specified. This is a powerful improvement over methods that rely on only one model being correct, offering increased robustness against model misspecification.

**Formal Definition:**

Let's define the following:

*   `Y`: Outcome variable
*   `A`: Treatment variable (binary, 0 or 1)
*   `X`: Confounders (observed covariates)
*   `Y(a)`: Potential outcome under treatment `a` (counterfactual)
*   `E[Y(a)]`: Average potential outcome under treatment `a`
*   `ATE = E[Y(1)] - E[Y(0)]`: Average Treatment Effect
*   `e(X) = P(A=1 | X)`: Propensity score (probability of treatment given covariates)
*   `m(X, A) = E[Y | A, X]`: Outcome model (expected outcome given treatment and covariates)
*   `m_1(X) = E[Y | A=1, X]`
*   `m_0(X) = E[Y | A=0, X]`

The AIPW estimator for `E[Y(a)]` is defined as:

`E[Y(a)]_AIPW = 1/n * Σ [I(A=a) * (Y - m(X,A)) / P(A=a|X) + m(X,a)]`

Where:

*   `I(A=a)` is an indicator function, equal to 1 if `A=a` and 0 otherwise.
*   The first term `I(A=a) * (Y - m(X,A)) / P(A=a|X)` is the inverse propensity weighted (IPW) component, adjusted by the residual from the outcome model.
*   The second term `m(X,a)` is the outcome model component.

Then, the AIPW estimator for ATE is:

`ATE_AIPW = E[Y(1)]_AIPW - E[Y(0)]_AIPW`

**How we can use it:**

1.  **Model the Propensity Score (e(X)):**  Estimate the probability of treatment assignment given covariates.  This is often done using logistic regression.

2.  **Model the Outcome (m(X, A)):** Estimate the expected outcome given treatment and covariates. This can be done using any appropriate regression model (linear, logistic, etc.).

3.  **Plug into the AIPW Formula:** Substitute the estimated propensity scores and outcome model predictions into the AIPW formula to obtain the ATE estimate.

DR learners extend this concept by using machine learning models for both the propensity score and outcome models, which can adapt better to complex relationships than simpler parametric models.  The "doubly robust" property holds as long as one of the models (propensity score or outcome model) is correctly specified. If both are correct, the estimator is even more efficient (lower variance).

## 2) Application scenario

**Scenario: A/B Testing with Confounding**

Imagine a company is running an A/B test to evaluate the effectiveness of a new website design (treatment A=1) compared to the old design (treatment A=0) on user engagement (Y - e.g., time spent on site).  However, the assignment to the new design wasn't perfectly randomized. Users with specific demographics (X - e.g., age, location, previous purchase history) were more likely to be assigned to the new design.

In this case, simply comparing the average engagement of users in the two groups would be biased due to these confounding factors.

**Applying Doubly Robust Estimation:**

1.  **Propensity Score Model:** We would build a model (e.g., logistic regression) to estimate the probability of a user being assigned to the new design (A=1) based on their demographics (X).  `e(X) = P(A=1 | X)`.

2.  **Outcome Model:** We would build a model (e.g., linear regression) to predict user engagement (Y) based on the website design they were assigned to (A) and their demographics (X). `m(X, A) = E[Y | A, X]`.

3.  **AIPW Estimation:** We would then use the estimated propensity scores and outcome model predictions in the AIPW formula to estimate the ATE, which represents the causal effect of the new website design on user engagement, accounting for the confounding effects of the demographics.

The doubly robust property ensures that even if our propensity score model is slightly off (e.g., some important interaction terms are missing), the ATE estimate will still be consistent if our outcome model is correctly specified (and vice versa).

## 3) Python method (if possible)

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split

def doubly_robust_estimation(data, outcome, treatment, confounders):
    """
    Estimates the ATE using Doubly Robust (AIPW) estimation.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        outcome (str): Name of the outcome variable column.
        treatment (str): Name of the treatment variable column (binary).
        confounders (list): List of column names representing confounders.

    Returns:
        float: Estimated Average Treatment Effect (ATE).
    """

    # Separate data
    Y = data[outcome].values
    A = data[treatment].values
    X = data[confounders].values

    # 1. Propensity Score Model
    propensity_model = LogisticRegression(solver='liblinear', random_state=42)  # or any other appropriate model
    propensity_model.fit(X, A)
    e_hat = propensity_model.predict_proba(X)[:, 1]  # Probability of treatment

    # 2. Outcome Model
    outcome_model = LinearRegression()  # or any other appropriate model
    X_A = np.concatenate([X, A.reshape(-1, 1)], axis=1)
    outcome_model.fit(X_A, Y)
    m_hat = outcome_model.predict(X_A)  # Predicted outcome for observed treatment

    # Predict potential outcomes: E[Y|A=1, X] and E[Y|A=0, X]
    X_A1 = np.concatenate([X, np.ones(len(X)).reshape(-1, 1)], axis=1) # A=1
    X_A0 = np.concatenate([X, np.zeros(len(X)).reshape(-1, 1)], axis=1) # A=0
    m1_hat = outcome_model.predict(X_A1)
    m0_hat = outcome_model.predict(X_A0)

    # 3. AIPW Estimation
    n = len(data)

    # AIPW for E[Y(1)]
    y1_aipw = np.mean((A * (Y - m_hat) / e_hat) + m1_hat)

    # AIPW for E[Y(0)]
    y0_aipw = np.mean(((1 - A) * (Y - m_hat) / (1 - e_hat)) + m0_hat)


    # ATE = E[Y(1)] - E[Y(0)]
    ate_aipw = y1_aipw - y0_aipw

    return ate_aipw

# Example Usage (with simulated data)
np.random.seed(42)
n = 1000
data = pd.DataFrame({
    'X1': np.random.randn(n),
    'X2': np.random.randn(n),
})
# Simulate treatment assignment, influenced by X1 and X2
data['A'] = np.random.binomial(1, 0.3 + 0.2 * data['X1'] - 0.1 * data['X2'])
# Simulate outcome, influenced by A, X1 and X2
data['Y'] = 2 + 1.5 * data['A'] + 0.5 * data['X1'] - 0.3 * data['X2'] + np.random.randn(n)

outcome = 'Y'
treatment = 'A'
confounders = ['X1', 'X2']

ate = doubly_robust_estimation(data.copy(), outcome, treatment, confounders) # use copy to avoid modifying data in place
print(f"Estimated ATE (Doubly Robust): {ate}")


# Example using training and testing splits to avoid overfitting
X = data[confounders].values
A = data[treatment].values
Y = data[outcome].values

X_train, X_test, A_train, A_test, Y_train, Y_test = train_test_split(X, A, Y, test_size=0.2, random_state=42)

# Propensity Model
propensity_model = LogisticRegression(solver='liblinear', random_state=42)
propensity_model.fit(X_train, A_train)
e_hat = propensity_model.predict_proba(X_test)[:, 1]

# Outcome Model
X_A_train = np.concatenate([X_train, A_train.reshape(-1, 1)], axis=1)
X_A_test = np.concatenate([X_test, A_test.reshape(-1, 1)], axis=1)
outcome_model = LinearRegression()
outcome_model.fit(X_A_train, Y_train)
m_hat = outcome_model.predict(X_A_test)

X_A1_test = np.concatenate([X_test, np.ones(len(X_test)).reshape(-1, 1)], axis=1)
X_A0_test = np.concatenate([X_test, np.zeros(len(X_test)).reshape(-1, 1)], axis=1)
m1_hat = outcome_model.predict(X_A1_test)
m0_hat = outcome_model.predict(X_A0_test)

# AIPW Estimation on the test set
y1_aipw = np.mean((A_test * (Y_test - m_hat) / e_hat) + m1_hat)
y0_aipw = np.mean(((1 - A_test) * (Y_test - m_hat) / (1 - e_hat)) + m0_hat)
ate_aipw = y1_aipw - y0_aipw

print(f"Estimated ATE (Doubly Robust - Train/Test Split): {ate_aipw}")
```

Key improvements in the code include:

*   Clarity and comments explaining each step.
*   Use of `predict_proba` for the logistic regression model to get probabilities.
*   Concatenation of X and A into `X_A` to properly fit the outcome model.
*   Prediction of *both* `E[Y|A=1,X]` and `E[Y|A=0,X]` to compute E[Y(1)] and E[Y(0)].
*   Incorporates a Train/Test split to improve generalization to unseen data and prevent overfitting.
*   Explicitly seeds the random number generator for reproducibility.
*   Handles numpy arrays for better performance.

## 4) Follow-up question

How does the performance of Doubly Robust estimation compare to other causal inference methods, such as Inverse Propensity Weighting (IPW) and Outcome Regression, especially when both models are misspecified or when the positivity assumption is violated?  What diagnostic checks can be used to assess the validity of the assumptions required for Doubly Robust estimation?