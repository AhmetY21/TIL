---
title: "Inverse Probability Weighting (IPW)"
date: "2026-02-12"
week: 7
lesson: 5
slug: "inverse-probability-weighting-ipw"
---

# Topic: Inverse Probability Weighting (IPW)

## 1) Formal definition (what is it, and how can we use it?)

Inverse Probability Weighting (IPW) is a statistical technique used in causal inference to estimate the causal effect of a treatment or intervention when there are confounding variables.  It aims to address the bias introduced by these confounders by creating a pseudo-population where treatment assignment is independent of the observed covariates.

Here's a breakdown:

* **Goal:** Estimate the average treatment effect (ATE), which is the average difference in outcomes if everyone received the treatment compared to if no one received the treatment.
* **Problem:** Confounding variables distort the observed association between treatment and outcome.  For example, sicker patients may be more likely to receive a treatment, biasing the observed treatment effect.
* **Solution:** IPW reweights each observation by the inverse of their probability of receiving the treatment they actually received, conditional on their observed covariates. This creates a "balanced" sample, similar to what you'd see in a randomized controlled trial where treatment assignment is independent of other observed factors.

**How it works mathematically:**

Let:

*   `A` be the treatment (e.g., 1 for treatment, 0 for control)
*   `X` be the observed covariates (confounders)
*   `Y` be the outcome

The IPW estimator for the ATE is:

ATE =  E[ Y / P(A = 1 | X) ] * P(A = 1) - E[ Y / P(A = 0 | X) ] * P(A = 0)

Or more practically, for sample size *n*:

ATE_hat = (1/n) * sum [ Y<sub>i</sub> * (A<sub>i</sub> / P(A<sub>i</sub> = 1 | X<sub>i</sub>)) ] - (1/n) * sum [ Y<sub>i</sub> * ((1-A<sub>i</sub>) / P(A<sub>i</sub> = 0 | X<sub>i</sub>)) ]

Where:

*   `P(A = 1 | X)` is the probability of receiving treatment given the covariates `X` (the propensity score).  This is typically estimated using a model like logistic regression.
*   `P(A = 0 | X)` is the probability of *not* receiving treatment given the covariates `X` (1 - propensity score if A is binary).
*   `A<sub>i</sub>` is the treatment actually received by individual `i`.
*   `Y<sub>i</sub>` is the outcome observed for individual `i`.
* E[] denotes the empirical estimate.

**Key Assumptions:**

* **Consistency:**  The observed outcome is the potential outcome under the received treatment.
* **Positivity (Overlap):** For every combination of covariates `X`, there must be a non-zero probability of receiving both treatment and control (0 < P(A = 1 | X) < 1). This means no group defined by the covariates is deterministically always treated or never treated.  This is crucial; if P(A=a|X) = 0, you'll divide by zero.
* **Conditional Exchangeability (No Unmeasured Confounding):**  All relevant confounders are observed and included in `X`.  There are no unmeasured confounders that influence both treatment and outcome.  This is the most critical and often untestable assumption.

## 2) Application scenario

Consider a study investigating the effect of a new educational program (`A = 1` if a student participated, `A = 0` otherwise) on student test scores (`Y`).  It's likely that students who participate in the program are systematically different from those who don't.  For example, students with lower grades or higher motivation (measured by variables in `X` like GPA, attendance, parent involvement) might be more likely to enroll.  Simply comparing the test scores of students who participated to those who didn't will likely be biased.

IPW can be used to estimate the causal effect of the program by:

1.  Estimating the propensity score P(A = 1 | X) for each student, using logistic regression (or another suitable model) to predict program participation based on their GPA, attendance, and parent involvement.
2.  Calculating the inverse probability weights: `A_i / P(A_i = 1 | X_i)` for participants and `(1 - A_i) / P(A_i = 0 | X_i)` for non-participants.
3.  Applying the IPW estimator (described in Section 1) to estimate the ATE of the program on test scores.

The IPW estimator will re-weight the data, giving more weight to non-participants with characteristics similar to participants (e.g., lower GPA) and more weight to participants with characteristics similar to non-participants (e.g., higher GPA). This creates a pseudo-population where program participation is no longer associated with these pre-existing characteristics, allowing for a less biased estimate of the program's effect.

## 3) Python method (if possible)

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

def estimate_ate_ipw(data, treatment_col, outcome_col, confounder_cols):
  """
  Estimates the Average Treatment Effect (ATE) using Inverse Probability Weighting.

  Args:
    data: Pandas DataFrame containing the data.
    treatment_col: Name of the column representing the treatment (binary: 0 or 1).
    outcome_col: Name of the column representing the outcome.
    confounder_cols: List of column names representing the confounders.

  Returns:
    ate_estimate: Estimated Average Treatment Effect.
  """

  # 1. Estimate propensity scores
  X = data[confounder_cols]
  A = data[treatment_col]

  propensity_model = LogisticRegression(solver='liblinear', random_state=42)  # or another suitable solver
  propensity_model.fit(X, A)
  propensity_scores = propensity_model.predict_proba(X)[:, 1]  # Probability of treatment (A=1)

  # Handle potential positivity violations by trimming or truncating propensity scores
  # This is CRUCIAL!  Leaving this out will lead to unstable weights and high variance
  # Example of trimming:
  epsilon = 0.05 # Adjust based on your data
  propensity_scores = np.clip(propensity_scores, epsilon, 1 - epsilon)


  # 2. Calculate inverse probability weights
  weights = np.where(A == 1, 1 / propensity_scores, 1 / (1 - propensity_scores))

  # 3. Estimate ATE
  Y = data[outcome_col]
  ate_estimate = np.mean(weights * Y * (A == 1)) - np.mean(weights * Y * (A == 0))


  return ate_estimate



# Example usage (assuming you have a DataFrame called 'df')
# Create example data if needed
np.random.seed(42)
n = 100
df = pd.DataFrame({
    'X1': np.random.randn(n),
    'X2': np.random.randn(n),
    'Treatment': np.random.randint(0, 2, n), # 0 or 1
    'Outcome': np.random.randn(n)
})
# The following line creates an outcome that is correlated with both treatment and confounders.
df['Outcome'] =  0.5 * df['Treatment'] + 0.3 * df['X1'] - 0.2 * df['X2'] + np.random.randn(n)

treatment_col = 'Treatment'
outcome_col = 'Outcome'
confounder_cols = ['X1', 'X2']

ate = estimate_ate_ipw(df, treatment_col, outcome_col, confounder_cols)

print(f"Estimated ATE using IPW: {ate}")
```

**Important considerations:**

*   **Propensity Score Model:** The accuracy of the propensity score model is crucial.  Consider using cross-validation and different model specifications (e.g., including interaction terms or non-linear terms).
*   **Positivity Violation:**  If there are individuals or groups with propensity scores close to 0 or 1, the weights become very large, leading to unstable estimates. Trimming or truncating the propensity scores is a common technique to mitigate this, but it introduces some bias (essentially making stronger assumptions).  The `epsilon` variable in the example code shows how to trim the propensity scores to avoid extreme weights.
*   **Variance:** IPW can have high variance, especially when there are extreme weights.  Consider using variance reduction techniques like stabilized weights or using more sophisticated estimators (e.g., augmented IPW, which is doubly robust).
*   **Diagnostics:**  Check for balance in the confounders after weighting.  You can compare the means and variances of the confounders between the treated and untreated groups after weighting.  Large differences suggest that the IPW estimator is not doing a good job of balancing the groups.

## 4) Follow-up question

What are the advantages and disadvantages of using Augmented Inverse Probability Weighting (AIPW) compared to regular IPW, and in what situations is AIPW preferred?