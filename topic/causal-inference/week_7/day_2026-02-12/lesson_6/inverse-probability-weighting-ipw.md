---
title: "Inverse Probability Weighting (IPW)"
date: "2026-02-12"
week: 7
lesson: 6
slug: "inverse-probability-weighting-ipw"
---

# Topic: Inverse Probability Weighting (IPW)

## 1) Formal definition (what is it, and how can we use it?)

Inverse Probability Weighting (IPW) is a statistical technique used in causal inference to estimate the average treatment effect (ATE) by creating a pseudo-population in which treatment assignment is independent of measured confounders.  It addresses confounding by weighting each observation by the inverse of the probability that they received the treatment they actually received, *given* their observed covariates.

**More formally:**

Let:

*   `A` be the binary treatment variable (1 for treated, 0 for untreated).
*   `X` be a set of confounders (observed covariates).
*   `Y` be the outcome variable.
*   `e(X) = P(A=1 | X)` be the propensity score, the probability of receiving treatment given the covariates `X`.

The IPW estimator estimates the ATE as follows:

1.  **Estimate the Propensity Score:** This is often done using logistic regression, where the treatment `A` is regressed on the covariates `X`.  This gives us an estimate of `e(X)`.

2.  **Calculate the Weights:**  Each observation is assigned a weight:

    *   If `A=1` (treated), the weight is `1 / e(X)`.
    *   If `A=0` (untreated), the weight is `1 / (1 - e(X))`.

3.  **Estimate the ATE:** The ATE is estimated as the weighted difference in means of the outcome variable `Y` between the treated and untreated groups:

    ATE =  mean(Y * (A / e(X)))  -  mean(Y * ((1-A) / (1-e(X))))

**How we use it:**

IPW allows us to estimate the causal effect of a treatment even when there is confounding. By re-weighting the data, IPW aims to create a situation similar to a randomized controlled trial, where treatment assignment is independent of the observed covariates.  It's a crucial tool for estimating causal effects from observational data.

**Important Considerations:**

*   **Positivity/Overlap Assumption:** IPW relies on the *positivity* or *overlap* assumption, which states that for every combination of covariate values `X`, there must be a non-zero probability of receiving both treatment and control. In other words, `0 < P(A=1 | X) < 1` for all `X`.  Violations of this assumption can lead to unstable and biased estimates, especially if propensity scores are very close to 0 or 1.
*   **Model Specification:** The accuracy of the IPW estimator depends heavily on the correct specification of the propensity score model. If the model is misspecified (e.g., important confounders are omitted), the resulting ATE estimate will be biased.
*   **Stabilized Weights:** To reduce the variance of the IPW estimator, *stabilized weights* are often used.  These are calculated as `P(A=a) / P(A=a | X)`, where `P(A=a)` is the marginal probability of treatment `a` (e.g., the proportion of treated individuals in the sample).  These are less susceptible to extreme weights.

## 2) Application scenario

Imagine you are analyzing a dataset of patients who received a new drug to treat high blood pressure. You want to determine if the drug causally reduces blood pressure.  However, the decision to prescribe the drug was not randomized.  Doctors were more likely to prescribe the drug to patients with more severe hypertension and other pre-existing conditions (age, obesity, diabetes). These conditions are confounders, as they affect both the likelihood of receiving the drug and the outcome (blood pressure).

In this scenario, you can use IPW to estimate the effect of the drug.

1.  **Identify Confounders (X):** Age, obesity, diabetes, baseline blood pressure, etc.
2.  **Estimate Propensity Score (e(X)):**  Use logistic regression to predict the probability of receiving the drug (A=1) based on the confounders (X).  `A ~ Age + Obesity + Diabetes + BaselineBloodPressure`
3.  **Calculate Weights:**  Calculate the inverse probability weights as described above.
4.  **Estimate ATE:**  Calculate the weighted difference in mean blood pressure between the treated and untreated groups. This provides an estimate of the drug's causal effect on blood pressure, adjusted for the confounding effects of the observed covariates.

## 3) Python method (if possible)

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

def estimate_ate_ipw(data, treatment_col, outcome_col, confounders):
    """
    Estimates the Average Treatment Effect (ATE) using Inverse Probability Weighting (IPW).

    Args:
        data (pd.DataFrame): The dataset.
        treatment_col (str): Name of the treatment column (binary: 0 or 1).
        outcome_col (str): Name of the outcome column.
        confounders (list): List of column names representing confounders.

    Returns:
        float: Estimated ATE.
    """

    # 1. Estimate Propensity Score
    formula = treatment_col + " ~ " + " + ".join(confounders)
    model = smf.glm(formula=formula, data=data, family=sm.families.Binomial()).fit()
    propensity_scores = model.predict(data)

    # 2. Calculate Weights
    weights = np.where(data[treatment_col] == 1, 1 / propensity_scores, 1 / (1 - propensity_scores))


    # 3. Estimate ATE
    ate = np.mean(data[outcome_col] * weights * data[treatment_col]) - np.mean(data[outcome_col] * weights * (1 - data[treatment_col]))

    return ate


# Example usage with dummy data:
data = pd.DataFrame({
    'treatment': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    'outcome': [2, 5, 1, 6, 3, 7, 2, 8, 1, 9],
    'age': [30, 40, 35, 45, 32, 42, 37, 47, 33, 43],
    'income': [50000, 70000, 60000, 80000, 55000, 75000, 65000, 85000, 58000, 78000]
})

treatment_col = 'treatment'
outcome_col = 'outcome'
confounders = ['age', 'income']

ate = estimate_ate_ipw(data, treatment_col, outcome_col, confounders)
print(f"Estimated ATE: {ate}")
```

## 4) Follow-up question

How does IPW compare to other causal inference methods like propensity score matching or regression adjustment, and when might one be preferred over the others? What are the relative strengths and weaknesses of each approach?