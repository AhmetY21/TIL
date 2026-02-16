---
title: "Instrumental Variables (IV): Relevance and Exclusion"
date: "2026-02-16"
week: 8
lesson: 2
slug: "instrumental-variables-iv-relevance-and-exclusion"
---

# Topic: Instrumental Variables (IV): Relevance and Exclusion

## 1) Formal definition (what is it, and how can we use it?)

Instrumental Variables (IV) is a statistical technique used in causal inference to estimate the causal effect of a treatment or exposure (X) on an outcome (Y) when there is confounding, meaning the relationship between X and Y is influenced by other variables (U) that affect both. IVs exploit a third variable (Z), the *instrument*, that is related to the treatment but not directly to the outcome, except through the treatment. This allows us to isolate the causal effect of X on Y.

More formally:

*   **Instrument (Z):** A variable that influences the treatment (X) but does *not* directly influence the outcome (Y), except through its effect on X.

To be a valid instrument, Z must satisfy two key conditions:

*   **Relevance:** The instrument (Z) is correlated with the treatment (X). Mathematically, cov(Z, X) != 0. This means that the instrument must have a significant effect on the likelihood or intensity of the treatment.  A weak instrument has a small correlation with X, which can lead to biased estimates of the causal effect.

*   **Exclusion Restriction:** The instrument (Z) affects the outcome (Y) only through its effect on the treatment (X). Mathematically, cov(Z, Y | X, U) = 0, where U represents unobserved confounders. This means that there are no direct pathways from Z to Y other than through X. This is the hardest assumption to verify, as it states a *lack* of effect.

In essence, IV allows us to "mimic" a randomized controlled trial (RCT) even when we can't directly randomize the treatment itself. By leveraging the instrument, we can estimate the effect of X on Y by examining how changes in Z influence changes in Y, knowing that the only way Z can influence Y is through X (given the exclusion restriction).

Two common methods for implementing IV are:

1.  **Two-Stage Least Squares (2SLS):** A two-step regression approach:
    *   *Stage 1:* Regress the treatment (X) on the instrument (Z) and any control variables. Obtain the predicted values of X (denoted as X_hat).
    *   *Stage 2:* Regress the outcome (Y) on the predicted values of X (X_hat) and any control variables.  The coefficient on X_hat in this second regression is the IV estimate of the causal effect of X on Y.

2.  **Wald Estimator:** A simplified version of 2SLS applicable when both X and Z are binary. It calculates the causal effect as:

    Causal Effect = (Change in Y caused by Z) / (Change in X caused by Z) =  (E[Y | Z=1] - E[Y | Z=0]) / (E[X | Z=1] - E[X | Z=0])

## 2) Application scenario

**Scenario:** We want to estimate the effect of education (X) on income (Y). However, there are likely unobserved confounders (U) like ability, family background, and motivation that affect both education and income. These confounders make it difficult to isolate the causal effect of education.

**Instrument:** Distance to the nearest college (Z).

**Relevance:** Students who live closer to a college are more likely to attend college (cov(Z, X) != 0).  The closer you live, the lower the barriers to entry (e.g., commute time, cost).

**Exclusion Restriction:** Distance to college affects income *only* through its effect on education (cov(Z, Y | X, U) = 0).  We assume that distance to college doesn't directly influence your earning potential in other ways, *except* by influencing how much education you get.  This is a strong assumption that might be violated if, for example, living in an area with many colleges provides other opportunities (jobs, networking) independent of your education level.

**Using IV:** We can use the distance to college as an instrument for education to estimate the causal effect of education on income.  We would first estimate the relationship between distance to college and years of education, and then use that relationship to estimate the effect of predicted education on income.

## 3) Python method (if possible)

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np

# Simulate some data
np.random.seed(0)
n = 500

# Simulate unobserved confounder
U = np.random.normal(0, 1, n)

# Simulate instrument (distance to college)
Z = np.random.normal(0, 1, n)

# Simulate treatment (education) - affected by instrument and confounder
X = 0.5*Z + 0.7*U + np.random.normal(0, 1, n)

# Simulate outcome (income) - affected by treatment and confounder
Y = 2*X + 0.9*U + np.random.normal(0, 1, n)

df = pd.DataFrame({'Y': Y, 'X': X, 'Z': Z, 'U':U}) #Added 'U' to dataframe

# Two-Stage Least Squares (2SLS) using statsmodels

# Stage 1: Regress X on Z
first_stage = smf.ols('X ~ Z', data=df).fit()
df['X_hat'] = first_stage.predict(df)  # Predicted values of X

# Stage 2: Regress Y on X_hat
second_stage = smf.ols('Y ~ X_hat', data=df).fit()

print("2SLS Results:")
print(second_stage.summary())

# Alternative using statsmodels' IV2SLS class.
from statsmodels.sandbox.regression.iv import IV2SLS

iv_model = IV2SLS(df['Y'], df['X'], df['Z']).fit()

print("\nIV2SLS Results (more concise):")
print(iv_model.summary)

# OLS (ignoring confounding) for comparison
ols_model = smf.ols('Y ~ X', data=df).fit()
print("\nOLS Results (biased due to confounding):")
print(ols_model.summary())

```

**Explanation:**

*   We simulate data with an unobserved confounder (U) affecting both X and Y.
*   We define our instrument Z.
*   We perform 2SLS using `statsmodels`. First we regress X on Z, and then Y on the predicted values of X.
*   We compare the 2SLS results to OLS, which is biased due to the confounding. The coefficient on X in the OLS regression is closer to 2 + 0.9*0.7 = 2.63 because it captures the causal effect plus the effect of the confounder U.  The IV results should be closer to 2 because it's designed to remove that bias.
*   The `IV2SLS` class provides a more direct way to implement IV in `statsmodels`.

## 4) Follow-up question

How do you test for weak instruments and what are the consequences of using a weak instrument? What are some approaches to dealing with weak instruments?