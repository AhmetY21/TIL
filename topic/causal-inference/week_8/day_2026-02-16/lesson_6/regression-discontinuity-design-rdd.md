---
title: "Regression Discontinuity Design (RDD)"
date: "2026-02-16"
week: 8
lesson: 6
slug: "regression-discontinuity-design-rdd"
---

# Topic: Regression Discontinuity Design (RDD)

## 1) Formal definition (what is it, and how can we use it?)

Regression Discontinuity Design (RDD) is a quasi-experimental research design that allows researchers to estimate the causal effect of a treatment or intervention by exploiting a discontinuity in the assignment rule. In essence, individuals are assigned to treatment or control based on a continuous variable (called the "running variable" or "forcing variable") and a predetermined cutoff. Those just above the cutoff receive the treatment, while those just below do not.

The core idea is that individuals just above and just below the cutoff are, on average, very similar except for their treatment status. Therefore, any jump (discontinuity) in the outcome variable at the cutoff can be attributed to the effect of the treatment. This relies on the assumption that the running variable itself does not directly cause the outcome (only indirectly through the treatment) and that other factors influencing the outcome are smoothly distributed around the cutoff.

Two main types of RDD exist:

*   **Sharp RDD:**  Treatment assignment is entirely determined by the cutoff.  If the running variable is above the cutoff, the individual *always* receives treatment; if below, they *never* receive treatment. This is the simpler and more commonly used type.

*   **Fuzzy RDD:**  The probability of receiving treatment jumps at the cutoff, but treatment assignment is not completely determined by the running variable.  Some individuals above the cutoff might not receive treatment, and some below might receive it.  Fuzzy RDD requires an instrumental variable approach, using the assignment rule (being above or below the cutoff) as an instrument for actual treatment receipt.

We use RDD to estimate a *local* average treatment effect (LATE) - the average treatment effect for those individuals whose treatment status is determined by the cutoff.  This makes RDD useful when a randomized controlled trial (RCT) is not feasible or ethical.

## 2) Application scenario

A common application scenario for RDD is evaluating the impact of scholarships on academic performance. Suppose a university offers a scholarship to students who score above a certain threshold on a standardized test (the running variable).  Students just above the threshold receive the scholarship (the treatment), while those just below do not.

The outcome variable of interest might be the students' GPA after one year of college. We can use RDD to estimate the causal effect of receiving the scholarship on GPA by comparing the GPA of students slightly above the threshold to the GPA of students slightly below the threshold.  The assumption is that these two groups of students are otherwise similar in terms of their academic ability, motivation, and other factors that might affect GPA. Any significant difference in GPA at the threshold can be attributed to the effect of the scholarship.

Another application is determining the effect of legal drinking age on mortality. The legal drinking age (say, 21) is the threshold, and the running variable is age. The outcome is mortality. We can compare mortality rates of people just above and just below the age of 21 to see if access to alcohol causes a jump in mortality.

## 3) Python method (if possible)

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Simulate some data (Sharp RDD)
np.random.seed(123)
n = 200
cutoff = 0

# Running variable
x = np.random.normal(0, 1, n)

# Treatment assignment
treatment = (x > cutoff).astype(int)

# True effect of treatment
treatment_effect = 1.5

# Outcome variable (with some noise)
y = 2 + 0.5*x + treatment_effect*treatment + np.random.normal(0, 0.5, n)

# Create a Pandas DataFrame
data = pd.DataFrame({'x': x, 'treatment': treatment, 'y': y})

# Estimate the RDD model using statsmodels
# We use a linear regression with an interaction term
model = smf.ols("y ~ x + treatment + x:treatment", data=data).fit()
print(model.summary())


# Plot the data and the regression lines
x_plot = np.linspace(min(x), max(x), 100)
y_pred_below = model.params['Intercept'] + model.params['x'] * x_plot
y_pred_above = (model.params['Intercept'] + model.params['treatment']) + (model.params['x'] + model.params['x:treatment']) * x_plot


plt.scatter(x, y, label='Data')
plt.plot(x_plot, y_pred_below, color='red', label='Regression Line (Below Cutoff)')
plt.plot(x_plot, y_pred_above, color='green', label='Regression Line (Above Cutoff)')
plt.axvline(x=cutoff, color='black', linestyle='--', label='Cutoff')
plt.xlabel('Running Variable (x)')
plt.ylabel('Outcome (y)')
plt.title('Regression Discontinuity Design')
plt.legend()
plt.show()

# Estimate the jump at the cutoff (treatment effect)
treatment_effect_estimate = model.params['treatment']
print(f"Estimated Treatment Effect: {treatment_effect_estimate}")
```

The Python code simulates data for a sharp RDD, estimates the effect of the treatment using linear regression with interaction terms, plots the regression lines, and prints the estimated treatment effect. Key points:

*   `statsmodels` is used for the regression analysis.
*   The model includes the running variable (`x`), the treatment indicator (`treatment`), and an interaction term (`x:treatment`).  The interaction term allows for different slopes on either side of the cutoff.
*   The coefficient on `treatment` estimates the jump in the outcome at the cutoff.
*   Plotting the data and regression lines is essential for visualizing the discontinuity and assessing the validity of the RDD assumptions.
* Bandwidth selection is an important step and cross validation can be used to find an optimal bandwidth.
## 4) Follow-up question

What are the main threats to the validity of RDD, and what diagnostic tests can be used to assess these threats? Specifically, what are some tests you can perform on the running variable near the cutoff?