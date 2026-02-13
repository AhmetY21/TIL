---
title: "Doubly Robust Estimation (AIPW / DR Learners)"
date: "2026-02-13"
week: 7
lesson: 1
slug: "doubly-robust-estimation-aipw-dr-learners"
---

# Topic: Doubly Robust Estimation (AIPW / DR Learners)

## 1) Formal definition (what is it, and how can we use it?)

Doubly Robust (DR) estimation, often implemented using Augmented Inverse Propensity Weighting (AIPW), is a technique for estimating causal effects that leverages both a propensity score model (modeling treatment assignment) and an outcome model (modeling the potential outcome).  The key property of DR estimators is that they provide consistent estimates of the Average Treatment Effect (ATE) or other causal quantities if *either* the propensity score model *or* the outcome model is correctly specified.  This "double robustness" is a significant advantage over methods like inverse propensity weighting (IPW) or outcome regression, which require both models to be correct to avoid bias.

Formally, let:

*   `Y` be the observed outcome.
*   `A` be the binary treatment indicator (1 for treated, 0 for control).
*   `X` be the set of observed confounders.
*   `E[Y|A=a, X]` represent the conditional expectation of Y given A=a and X.
*   `e(X) = P(A=1|X)` be the propensity score, i.e., the probability of receiving treatment given X.
*   `Y(a)` denote the potential outcome if everyone was treated with treatment `a`.

The ATE is defined as `E[Y(1) - Y(0)]`.  The AIPW estimator for the ATE can be expressed as:

```
ATE_AIPW = (1/n) * Σ [ (Y_i * (A_i / e(X_i)) - E[Y|A=1, X_i] * (A_i - e(X_i)) / e(X_i)) - (Y_i * ((1 - A_i) / (1 - e(X_i))) - E[Y|A=0, X_i] * (A_i - e(X_i)) / (1 - e(X_i))) ]
```

Where:

*   The summation is over all `i` from 1 to `n` (the number of observations).
*   `Y_i`, `A_i`, and `X_i` are the observed values for the `i`-th individual.
*   `e(X_i)` is the estimated propensity score for the `i`-th individual.
*   `E[Y|A=1, X_i]` and `E[Y|A=0, X_i]` are the estimated conditional expectations for the `i`-th individual under treatment and control, respectively.

The terms `E[Y|A=1, X]` and `E[Y|A=0, X]` are often estimated using separate regression models. The term  `Y_i * (A_i / e(X_i)) - E[Y|A=1, X_i] * (A_i - e(X_i)) / e(X_i)` is the AIPW estimator for the outcome under treatment, and similarly for the outcome under control. These terms include both an IPW component and an adjustment based on the outcome model.  The subtraction calculates the difference in these estimated outcomes to estimate the ATE.

How can we use it?  We use it to estimate causal effects, such as the ATE, in observational studies where treatment assignment is not randomized and confounding is present.  We need to specify two models: one to predict the probability of treatment and another to predict the outcome, conditional on treatment and covariates.  If *either* of these models is correctly specified, the AIPW estimator will be consistent.

## 2) Application scenario

Consider a healthcare scenario where we want to estimate the effect of a new medication on reducing hospital readmission rates.  We have observational data collected from a hospital system.  Patients were not randomly assigned to the medication; doctors made treatment decisions based on patient characteristics such as age, pre-existing conditions, and severity of illness.  Therefore, simply comparing readmission rates between patients who received the medication and those who didn't would likely be biased due to confounding.

We can use Doubly Robust Estimation (AIPW) to estimate the treatment effect.  We would:

1.  **Propensity Score Model:** Build a model (e.g., logistic regression) to predict the probability of a patient receiving the medication based on their age, pre-existing conditions, and illness severity (`X`). This estimates `e(X)`.
2.  **Outcome Model:** Build two separate models (e.g., regression) to predict readmission rates: one for patients who received the medication and one for patients who did not, both using age, pre-existing conditions, and illness severity (`X`) as predictors. These estimate `E[Y|A=1, X]` and `E[Y|A=0, X]`.
3.  **AIPW Estimation:** Plug the predicted propensity scores and outcome model predictions into the AIPW formula to calculate the estimated ATE.

Even if our propensity score model is slightly mis-specified (e.g., we've missed some subtle factor influencing treatment decisions), the AIPW estimator will still provide a consistent estimate of the treatment effect if our outcome model is correctly specified. Conversely, if our outcome model is mis-specified but our propensity score model is accurate, the AIPW estimator will still be consistent. This makes it a robust method for handling confounding in observational data.

## 3) Python method (if possible)

```python
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd

def aipw_estimation(df, treatment, outcome, confounders):
    """
    Estimates the Average Treatment Effect (ATE) using Augmented Inverse Propensity Weighting (AIPW).

    Args:
        df (pd.DataFrame): DataFrame containing treatment, outcome, and confounders.
        treatment (str): Name of the treatment column.
        outcome (str): Name of the outcome column.
        confounders (list): List of column names representing confounders.

    Returns:
        float: Estimated ATE using AIPW.
    """

    # 1. Propensity Score Model
    propensity_formula = f"{treatment} ~ " + " + ".join(confounders)
    propensity_model = smf.glm(formula=propensity_formula, data=df, family=sm.families.Binomial()).fit()
    propensity_scores = propensity_model.predict(df)

    # 2. Outcome Models
    outcome_formula = f"{outcome} ~ {treatment} + " + " + ".join(confounders)
    outcome_model = smf.ols(formula=outcome_formula, data=df).fit()
    # Alternatively, fit two separate models
    outcome_model_treated = smf.ols(formula=f"{outcome} ~ " + " + ".join(confounders), data=df[df[treatment]==1]).fit()
    outcome_model_untreated = smf.ols(formula=f"{outcome} ~ " + " + ".join(confounders), data=df[df[treatment]==0]).fit()


    # Predict outcomes under treatment and control
    df_treated = df.copy()
    df_treated[treatment] = 1
    outcome_predictions_treated = outcome_model_treated.predict(df_treated) if outcome_model_treated is not None else outcome_model.predict(df_treated)

    df_untreated = df.copy()
    df_untreated[treatment] = 0
    outcome_predictions_untreated = outcome_model_untreated.predict(df_untreated) if outcome_model_untreated is not None else outcome_model.predict(df_untreated)


    # 3. AIPW Estimator
    aipw_values = (
        (df[treatment] * (df[outcome] - outcome_predictions_treated) / propensity_scores)
        - ((1 - df[treatment]) * (df[outcome] - outcome_predictions_untreated) / (1 - propensity_scores))
        + outcome_predictions_treated
        - outcome_predictions_untreated
    )

    ate_aipw = np.mean(aipw_values)
    return ate_aipw

# Example usage:
if __name__ == '__main__':
    # Create a sample DataFrame
    data = {
        'treatment': np.random.binomial(1, 0.5, 100),
        'outcome': np.random.normal(0, 1, 100),
        'confounder1': np.random.normal(0, 1, 100),
        'confounder2': np.random.normal(0, 1, 100)
    }
    df = pd.DataFrame(data)

    # Define variables
    treatment_col = 'treatment'
    outcome_col = 'outcome'
    confounder_cols = ['confounder1', 'confounder2']

    # Estimate the ATE using AIPW
    ate = aipw_estimation(df, treatment_col, outcome_col, confounder_cols)
    print(f"Estimated ATE using AIPW: {ate}")
```

This code:

1.  Defines a function `aipw_estimation` that takes a DataFrame, treatment column name, outcome column name, and a list of confounders as input.
2.  Fits a logistic regression model for the propensity score using `statsmodels`.
3.  Fits an OLS regression model (or two separate models) for the outcome, depending on the specification.
4.  Calculates the AIPW estimate using the formula mentioned earlier, using predicted propensity scores and outcomes.
5.  Returns the estimated ATE.

**Important notes:**

*   The code assumes binary treatment and continuous outcome. The models can be adjusted for different outcome types (e.g., logistic regression for binary outcomes).
*   The function uses `statsmodels` for model fitting.
*   This is a basic implementation.  In a real-world scenario, you would want to include cross-validation to prevent overfitting and potentially explore different model specifications.
*   Ensure sufficient overlap between treatment and control groups for each value of the confounders (positivity assumption).  If propensity scores are too close to 0 or 1, IP weighting can become unstable.

## 4) Follow-up question

How do you assess the sensitivity of the AIPW estimate to violations of the unconfoundedness (ignorability) assumption? Specifically, what methods can be used to explore the potential impact of unobserved confounders on the estimated causal effect when using Doubly Robust estimation?