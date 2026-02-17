---
title: "Difference-in-Differences (DiD)"
date: "2026-02-17"
week: 8
lesson: 2
slug: "difference-in-differences-did"
---

# Topic: Difference-in-Differences (DiD)

## 1) Formal definition (what is it, and how can we use it?)

Difference-in-Differences (DiD) is a quasi-experimental method used to estimate the causal effect of a treatment (e.g., a policy intervention) by comparing the changes in outcomes over time between a treatment group and a control group. It leverages the fact that the treatment group is exposed to the intervention while the control group is not.

The key assumption underlying DiD is the **parallel trends assumption**. This assumption states that, in the absence of the treatment, the treatment and control groups would have followed similar trends in the outcome variable. In other words, any pre-existing differences between the groups are assumed to remain constant over time. DiD tries to isolate the *incremental* impact of the treatment above and beyond the pre-existing differences and trends.

Formally, the DiD estimator can be expressed as:

`DiD = (E[Y | Treated=1, Post=1] - E[Y | Treated=1, Pre=1]) - (E[Y | Treated=0, Post=1] - E[Y | Treated=0, Pre=1])`

Where:

*   `Y` is the outcome variable.
*   `Treated` is a binary variable indicating whether a unit belongs to the treatment group (1) or the control group (0).
*   `Post` is a binary variable indicating the period after the treatment (1) or before the treatment (0).
*   `E[.]` denotes the expected value.

In words, the DiD estimator is the difference between the change in the outcome variable for the treated group and the change in the outcome variable for the control group. This difference is attributed to the causal effect of the treatment.

DiD can be used to evaluate the effectiveness of various interventions, such as:

*   Policy changes (e.g., new regulations, tax reforms).
*   Social programs (e.g., job training programs, healthcare initiatives).
*   Marketing campaigns.

## 2) Application scenario

Imagine a city government implements a new anti-smoking policy in only one part of the city (the treatment area). They want to know if the policy has actually reduced smoking rates. They have smoking rate data for both the treatment area and a comparable control area (where the policy wasn't implemented) both before and after the policy change.

*   **Treatment Group:** Residents of the area where the anti-smoking policy was implemented.
*   **Control Group:** Residents of the area where the anti-smoking policy was *not* implemented.
*   **Outcome Variable:** Smoking rate (e.g., percentage of residents who smoke).
*   **Pre-Treatment Period:** Time before the policy was implemented.
*   **Post-Treatment Period:** Time after the policy was implemented.

If the anti-smoking policy was effective, we'd expect to see a larger *decrease* in smoking rates in the treatment area compared to the control area after the policy was implemented. DiD allows us to quantify this difference while accounting for any pre-existing differences in smoking rates between the two areas. For example, if the treatment area already had slightly lower smoking rates to begin with, DiD accounts for this.

## 3) Python method (if possible)

```python
import pandas as pd
import statsmodels.formula.api as sm

# Sample Data (replace with your actual data)
data = pd.DataFrame({
    'area': ['Treatment', 'Treatment', 'Control', 'Control'] * 2,
    'time': ['Pre', 'Post'] * 4,
    'smoking_rate': [20, 15, 22, 20, 18, 12, 21, 19] # Example smoking rates
})

# Create dummy variables for the model
data['treatment'] = data['area'].apply(lambda x: 1 if x == 'Treatment' else 0)
data['post'] = data['time'].apply(lambda x: 1 if x == 'Post' else 0)
data['interaction'] = data['treatment'] * data['post']

# Print the dataframe to check its format
print(data)

# Fit the OLS regression model
model = sm.ols('smoking_rate ~ treatment + post + interaction', data=data)
results = model.fit()

# Print the regression results
print(results.summary())

# The coefficient of the 'interaction' term is the DiD estimate.
```

Explanation:

1.  **Data Preparation:**  The code creates a Pandas DataFrame representing the data. This will often come from a CSV or database. Crucially, it includes columns for the area (treatment or control), time period (pre or post), and the outcome variable (smoking\_rate).
2.  **Dummy Variables:** Dummy variables are created for 'treatment' (1 if treatment area, 0 otherwise) and 'post' (1 if post-treatment period, 0 otherwise).  An 'interaction' term is created by multiplying the treatment and post dummy variables. This is the key term in the DiD regression.
3.  **OLS Regression:**  An ordinary least squares (OLS) regression is performed using the `statsmodels` library.  The formula `smoking_rate ~ treatment + post + interaction` specifies the regression model.  This is the core of the DiD implementation.
4.  **Interpretation:** The `results.summary()` output provides the regression results.  The coefficient associated with the `interaction` term is the DiD estimate.  This coefficient represents the estimated causal effect of the anti-smoking policy on smoking rates. You can interpret the coefficient of the `interaction` term as the estimated effect of the treatment. Also examine the p-value of the interaction term to assess statistical significance.

## 4) Follow-up question

How can we test the parallel trends assumption when using DiD, and what can be done if the assumption is violated?