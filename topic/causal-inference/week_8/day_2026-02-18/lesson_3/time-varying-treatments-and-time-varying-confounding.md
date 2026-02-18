---
title: "Time-Varying Treatments and Time-Varying Confounding"
date: "2026-02-18"
week: 8
lesson: 3
slug: "time-varying-treatments-and-time-varying-confounding"
---

# Topic: Time-Varying Treatments and Time-Varying Confounding

## 1) Formal definition (what is it, and how can we use it?)

Time-varying treatments and time-varying confounding occur when both the treatment received and the confounding variables (factors that influence both treatment and outcome) change over time. This creates a complex causal structure where past treatment can affect future confounders, future confounders can affect future treatment, and both past treatment and future confounders can affect the outcome. This scenario violates the standard assumptions of many simple causal inference methods, like regression, propensity score matching, or inverse probability of treatment weighting (IPTW) when applied naively.

Specifically:

*   **Time-Varying Treatment:** The treatment variable's value can change for a given individual at different time points. For example, starting, stopping, or changing dosage of a medication.

*   **Time-Varying Confounding:** The confounding variables also change over time for a given individual.  Crucially, these confounders are themselves *affected by past treatment*. This creates a feedback loop that complicates causal inference. Consider a study on the effect of exercise on weight loss, where exercise is the treatment, and the time-varying confounder is energy levels. Prior exercise may influence future energy levels, and those future energy levels could then influence whether an individual exercises at that future time.

Why is this a problem?  If we ignore the time-varying nature, and the fact that confounders are affected by past treatment, we can introduce bias.  For example, suppose healthier individuals were *more likely* to exercise at any given time *and* healthier individuals are more likely to lose weight.  If prior exercise increases "healthiness" (the confounder), and increased "healthiness" makes people more likely to exercise in the future, *and* increases their propensity to lose weight, standard methods will not correctly estimate the effect of exercise on weight loss.

**How can we use it?** To address this challenge, we need methods that account for this complex causal structure.  Two popular approaches are:

*   **G-methods:** G-computation, g-estimation of structural nested models, and inverse probability of treatment weighting (IPTW) with marginal structural models.  These methods explicitly model the longitudinal process and estimate the effect of a *treatment regime* (a sequence of treatments over time) on the outcome. They essentially simulate the effect of following a particular treatment strategy.

*   **Marginal Structural Models (MSMs):**  These models directly estimate the effect of treatment, conditional only on baseline (pre-treatment) covariates.  The time-varying confounders are dealt with by using weights based on the inverse probability of treatment, given past treatment and past covariates.  This aims to remove the confounding influence of the time-varying variables. MSMs provide population-average causal effects, assuming that treatments at all time points are independent of the time-varying confounders, *conditional on baseline covariates*.

## 2) Application scenario

Consider a study investigating the effect of a new drug (Treatment: `D`) on a patient's disease severity (Outcome: `Y`). The data is collected over three time points (t=0, 1, 2).

*   `D_t`: Drug use at time *t* (1 = yes, 0 = no).
*   `Y_t`: Disease severity at time *t* (continuous variable).
*   `C_t`:  A patient's overall health condition at time *t* (continuous variable). `C_t` is a time-varying confounder because it influences both the decision to prescribe the drug at time *t+1* and the patient's disease severity at time *t+1*. Furthermore, `C_t` might be *affected* by the drug use at time *t-1*.

A doctor might be more likely to prescribe the drug to patients with worse health (`C_t`).  At the same time, a patient's health status (`C_t`) directly impacts the severity of their disease (`Y_t`).  Crucially, the drug itself might improve the patient's health status, meaning that `D_t` influences `C_t+1`.

If we simply regress `Y_2` on `D_0`, `D_1`, and `D_2`, we will likely get a biased estimate of the drug's effect. This is because `C_1` and `C_2` confound the relationship between the drug and the disease severity. Simple adjustment for `C_1` and `C_2` will *not* suffice because they are themselves affected by prior treatment.

In this scenario, we could use a marginal structural model (MSM) to estimate the effect of the drug on disease severity.  The MSM would aim to estimate what the disease severity would have been if *all* patients had been treated according to a particular treatment regime, regardless of their changing health status.

## 3) Python method (if possible)

While there isn't a single, ubiquitous function that performs all steps of MSM estimation perfectly, here's an example using `statsmodels` and `sklearn` to illustrate the IPTW approach common in MSMs. This example assumes you already have a longitudinal dataset in a Pandas DataFrame.  This is a simplified example and would require substantial adaptation for real-world use.

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

# Sample Longitudinal Data (replace with your actual data)
data = pd.DataFrame({
    'patient_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
    'time': [0, 1, 2, 0, 1, 2, 0, 1, 2],
    'D': [0, 1, 0, 1, 1, 0, 0, 0, 1],  # Treatment
    'Y': [5, 3, 2, 2, 1, 4, 6, 5, 3],  # Outcome
    'C': [2, 1, 0, 0, 1, 2, 4, 3, 2],  # Time-varying Confounder
    'baseline_covariate': [1, 1, 1, 0, 0, 0, 1, 1, 1] #baseline covariate
})

# 1. Estimate Propensity Scores (Probability of treatment given past)

def estimate_propensity_scores(df):
    """Estimates propensity scores for each time point using logistic regression."""
    propensity_scores = []
    for t in df['time'].unique():
        df_t = df[df['time'] == t].copy()
        # create lagged variables
        if t > 0:
            past_data = df[df['time'] < t]
            #group by patient and get the last row of each patient's data BEFORE time t
            past_data = past_data.groupby('patient_id').last().reset_index()
            df_t = pd.merge(df_t, past_data[['patient_id', 'D', 'C']], on='patient_id', suffixes=('', '_lagged'), how='left')
            df_t = df_t.fillna(0) # Handle the first time point (t=0) when there are no lagged variables
            X = df_t[['C', 'baseline_covariate', 'C_lagged', 'D_lagged']] # Features for propensity score model
        else:
            X = df_t[['C', 'baseline_covariate']]
        y = df_t['D']  # Treatment is the target variable

        model = LogisticRegression(solver='liblinear') # Or other suitable classifier
        model.fit(X, y)
        propensity = model.predict_proba(X)[:, 1]  # Probability of treatment
        df_t['propensity'] = propensity
        propensity_scores.append(df_t)

    return pd.concat(propensity_scores)

data = estimate_propensity_scores(data)

# 2. Calculate Inverse Probability of Treatment Weights (IPTW)
def calculate_iptw(df):
    """Calculates IPTW based on estimated propensity scores."""
    df['weight'] = 1.0
    for t in df['time'].unique():
        df_t = df[df['time'] == t].copy()
        df.loc[df['time'] == t, 'weight'] = (df_t['D'] / df_t['propensity']) + ((1 - df_t['D']) / (1 - df_t['propensity']))
    # Calculate cumulative product of weights for each patient
    df['cumulative_weight'] = df.groupby('patient_id')['weight'].cumprod()

    return df

data = calculate_iptw(data)

# 3. Fit Weighted Outcome Model (Marginal Structural Model)
# Here, we fit a simple linear regression, weighted by the IPTW.
# This models the outcome as a function of treatment and baseline covariates,
# adjusted for the time-varying confounding.

X = data[['D', 'time', 'baseline_covariate']] # treatment, time, and baseline covariates
X = sm.add_constant(X) # Add a constant for the intercept
y = data['Y']
weights = data['cumulative_weight']

model = sm.WLS(y, X, weights=weights)  # Weighted Least Squares
results = model.fit()
print(results.summary())

#Interpretation: The coefficient for 'D' now (hopefully) provides a less biased estimate
#of the effect of the drug on the disease severity, adjusted for the time-varying confounding.
```

**Important Notes:**

*   **Stabilization:**  In practice, propensity scores close to 0 or 1 can lead to very large weights and unstable estimates.  *Stabilized weights* are often used to mitigate this.
*   **Positivity Assumption:** MSM relies on the *positivity assumption*:  For every combination of past treatments and confounders, there must be a non-zero probability of receiving each treatment option.  Violation of this assumption can lead to extreme weights and biased results.
*   **Model Specification:** The propensity score model and the outcome model must be correctly specified. Mis-specification can lead to biased estimates.
*   **Software Packages:** More specialized packages like `causalinference` in R or `econml` in Python can provide more sophisticated implementations of MSMs and other g-methods, often with built-in features for stabilization and diagnostics.  Also consider `linearmodels` for panel data.

## 4) Follow-up question

What are some diagnostic checks one should perform after fitting an MSM to assess the validity of the model and the plausibility of its assumptions, beyond simply looking at standard regression diagnostics? And how would you attempt to address violations of those assumptions?