---
title: "Two-Stage Least Squares (2SLS) and LATE"
date: "2026-02-16"
week: 8
lesson: 3
slug: "two-stage-least-squares-2sls-and-late"
---

# Topic: Two-Stage Least Squares (2SLS) and LATE

## 1) Formal definition (what is it, and how can we use it?)

Two-Stage Least Squares (2SLS) is an instrumental variable (IV) method used to estimate the causal effect of an endogenous variable (a variable correlated with the error term) on an outcome variable. The endogeneity can arise from omitted variable bias, simultaneity, or measurement error. 2SLS addresses this by using an instrumental variable (Z) that satisfies two key conditions:

*   **Relevance:** Z is correlated with the endogenous variable (X).
*   **Exclusion Restriction:** Z affects the outcome (Y) *only* through its effect on the endogenous variable (X). In other words, Z is independent of the error term in the outcome equation (Y).

Here's how 2SLS works in two stages:

**Stage 1:** Regress the endogenous variable (X) on the instrumental variable(s) (Z) and any other exogenous covariates (W) included in the model. This produces predicted values of the endogenous variable (X̂).

    X = αZ + βW + v
    X̂ = α̂Z + β̂W

**Stage 2:** Regress the outcome variable (Y) on the predicted values of the endogenous variable (X̂) and any other exogenous covariates (W).

    Y = γX + δW + ε
    Y = γX̂ + δW + ε

The coefficient γ from the second stage is the 2SLS estimate of the causal effect of X on Y.

**Local Average Treatment Effect (LATE):**  2SLS estimates the *Local Average Treatment Effect (LATE)*, which is the average treatment effect *only* for the *compliers*. Compliers are individuals whose treatment status changes in response to the instrument.

*   **Always-takers:** Always take the treatment, regardless of the instrument's value.
*   **Never-takers:** Never take the treatment, regardless of the instrument's value.
*   **Defiers:** Take the treatment when *not* encouraged by the instrument and do *not* take the treatment when encouraged. (often ignored due to monotonicity assumptions)
*   **Compliers:** Take the treatment when encouraged by the instrument and do *not* take the treatment when *not* encouraged.

LATE is important because the 2SLS estimate is *not* necessarily the average treatment effect (ATE) for the entire population. It only applies to the subpopulation of compliers. The LATE provides a causal estimate specific to those individuals whose treatment decision is affected by the instrument.

## 2) Application scenario

**Example: Effect of Education on Wages (Endogeneity: Ability Bias)**

Suppose we want to estimate the effect of education (years of schooling, X) on wages (Y).  A simple regression of wages on education might be biased because ability (unobserved) is likely correlated with both education and wages.  Individuals with higher ability are more likely to pursue more education *and* earn higher wages, even without the extra education.  This is omitted variable bias.

We can use proximity to a college as an instrument (Z). People who live closer to a college are more likely to attend college, regardless of their ability.

*   **Relevance:** Proximity to college is correlated with years of schooling.
*   **Exclusion Restriction:** Proximity to college *only* affects wages through its impact on educational attainment. We assume that living near a college doesn't directly increase someone's wages independently of their education level.

In this scenario, 2SLS would estimate the LATE: the average effect of additional schooling on wages *only for those people whose decision to attend college was influenced by the availability of a local college* (the compliers).

## 3) Python method (if possible)

```python
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np

# Simulate some data
np.random.seed(0)
n = 1000
z = np.random.normal(0, 1, n)  # Instrument (proximity to college)
w = np.random.normal(0, 1, n)  # Exogenous covariate (parental income)
u = np.random.random(n)        # Unobserved ability, correlated with X and Y
x = 0.5 * z + 0.3 * w + 0.2 * u + np.random.normal(0, 0.5, n) # Endogenous variable (education)
y = 0.8 * x + 0.4 * w + 0.5 * u + np.random.normal(0, 0.5, n) # Outcome variable (wages)

data = pd.DataFrame({'y': y, 'x': x, 'z': z, 'w': w})

# Stage 1: Regress x on z and w
stage1 = smf.ols("x ~ z + w", data=data).fit()
data['x_hat'] = stage1.predict(data)

# Stage 2: Regress y on x_hat and w
stage2 = smf.ols("y ~ x_hat + w", data=data).fit()
print(stage2.summary())

# Using statsmodels.api for 2SLS directly
import statsmodels.api as sm
from statsmodels.sandbox.regression.iv2sls import IV2SLS

res_iv = IV2SLS(data['y'], data[['w']], data['x'], data['z']).fit()
print(res_iv.summary)
```

This code snippet performs the following:

1.  **Simulates data:** Creates a dataset with an endogenous variable `x`, an outcome `y`, an instrument `z`, and a covariate `w`. The endogeneity is created by the unobserved variable `u` which correlates with both `x` and `y`.
2.  **Performs 2SLS manually:**  Fits the two-stage least squares regression as described above, first predicting `x` in stage 1, then using the predicted `x` in stage 2 to estimate the effect on `y`.
3.  **Performs 2SLS using `IV2SLS`:** Uses the `IV2SLS` function from `statsmodels.sandbox.regression.iv2sls` to perform the 2SLS estimation directly. The parameters passed are: outcome variable `y`, exogenous covariates `w`, endogenous variable `x`, and instrumental variable `z`.
4.  **Prints Results:** The results from both the manual 2SLS and the `IV2SLS` methods are printed. These will be very similar. Look for the coefficient on 'x_hat' (in the manual method) or `x` (in the `IV2SLS` method) to find the 2SLS estimate.

## 4) Follow-up question

How can we test the validity of the instrumental variable (specifically, the exclusion restriction), and what are some common challenges in finding good instruments in real-world applications?