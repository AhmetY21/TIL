---
title: "Mediation Analysis: Direct vs Indirect Effects"
date: "2026-02-18"
week: 8
lesson: 1
slug: "mediation-analysis-direct-vs-indirect-effects"
---

# Topic: Mediation Analysis: Direct vs Indirect Effects

## 1) Formal definition (what is it, and how can we use it?)

Mediation analysis is a statistical technique used to understand *how* an independent variable (X, the cause) influences a dependent variable (Y, the outcome) through one or more mediating variables (M, the mediator).  It aims to decompose the total effect of X on Y into:

*   **Direct Effect:** The effect of X on Y that is *not* transmitted through the mediator M. It's the causal pathway X -> Y directly.  This is often denoted as *c'*.

*   **Indirect Effect:** The effect of X on Y that *is* transmitted through the mediator M. It's the causal pathway X -> M -> Y.  This is often calculated as the product of the effect of X on M (*a*) and the effect of M on Y controlling for X (*b*), hence *a* * b*.

*   **Total Effect:** The overall effect of X on Y, irrespective of whether it's direct or indirect.  It's often denoted as *c*.  Ideally, the total effect equals the sum of the direct and indirect effects (c = c' + a*b), though this holds true most clearly in linear models.

We use mediation analysis to:

*   **Explain causal mechanisms:**  Uncover the processes through which a cause leads to an effect.  Instead of just knowing that X influences Y, we learn *why* and *how*.
*   **Identify intervention points:** If we understand that X affects Y through M, we can target M with interventions to modify Y.
*   **Evaluate interventions:** Assess whether an intervention works as intended by verifying that it influences the proposed mediator and that changes in the mediator lead to changes in the outcome.
*   **Explore complex causal pathways:**  Handle scenarios with multiple mediators and more complex relationships.

Key Assumptions:

*   **Causal Ordering:** X causes M, and both X and M cause Y.  The assumed temporal order is crucial.
*   **No Unmeasured Confounding:**  There are no unmeasured variables that affect both X and M, both M and Y (controlling for X), or both X and Y.  This is a crucial and often difficult-to-satisfy assumption.
*   **Linearity (often assumed):** The relationships between X, M, and Y are linear.  While extensions exist for non-linear relationships, the classic approach assumes linearity.  Generalized Linear Models (GLMs) can handle non-normal outcomes.
*   **Additivity (often assumed):** Effects are additive. This implies that the effect of M on Y doesn't depend on the value of X.

## 2) Application scenario

Imagine a study investigating the impact of a new exercise program (X) on overall happiness (Y). Researchers suspect that the exercise program improves physical fitness (M), which in turn leads to increased happiness.

*   **X:** Participation in the exercise program (binary: yes/no)
*   **M:** Physical fitness level (measured on a scale)
*   **Y:** Overall happiness (measured on a scale)

Mediation analysis can help determine:

*   **Direct Effect:** Does the exercise program *directly* increase happiness, even without considering changes in fitness levels? (X -> Y)
*   **Indirect Effect:** Does the exercise program increase happiness *because* it improves physical fitness? (X -> M -> Y)
*   **Total Effect:** What is the overall impact of the exercise program on happiness, combining both direct and indirect effects? (X -> Y, total causal influence)

Understanding the relative magnitudes of the direct and indirect effects can inform the researchers about the most important mechanisms driving the increase in happiness. If the indirect effect is much larger than the direct effect, it suggests that improving physical fitness is the primary pathway through which the exercise program enhances happiness.  Interventions might then focus on maximizing fitness gains from the program.  Conversely, a large direct effect suggests other factors may be at play, and further investigation is warranted.

## 3) Python method (if possible)

The `statsmodels` and `pymc` packages provide ways to conduct mediation analysis in Python.  Here's an example using `statsmodels`:

```python
import statsmodels.formula.api as smf
import pandas as pd

# Sample data (replace with your actual data)
data = {'X': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        'M': [2, 5, 3, 6, 1, 4, 2, 7, 3, 5],
        'Y': [4, 8, 5, 9, 3, 7, 4, 10, 5, 8]}
df = pd.DataFrame(data)

# Equation 1: Mediator model (X -> M)
model_M = smf.ols('M ~ X', data=df).fit()
print("Mediator Model:")
print(model_M.summary())

# Equation 2: Outcome model (X + M -> Y)
model_Y = smf.ols('Y ~ X + M', data=df).fit()
print("\nOutcome Model:")
print(model_Y.summary())

# Calculate the indirect effect (a*b)
a = model_M.params['X']  # Effect of X on M
b = model_Y.params['M']  # Effect of M on Y (controlling for X)
indirect_effect = a * b

# Calculate the direct effect (c')
direct_effect = model_Y.params['X']  # Effect of X on Y (controlling for M)

# Calculate the total effect (c) (obtained from a regression of Y on X alone)
model_total = smf.ols('Y ~ X', data=df).fit()
total_effect = model_total.params['X'] # Effect of X on Y (without M in the model)

print(f"\nIndirect Effect (a*b): {indirect_effect}")
print(f"Direct Effect (c'): {direct_effect}")
print(f"Total Effect (c): {total_effect}")

# For significance testing of the indirect effect, bootstrapping or other methods
# like the Sobel test are often used (beyond the scope of this basic example).

#Bootstrapping with Statsmodels:

import numpy as np

def get_indirect_effect(data):
  model_M_boot = smf.ols('M ~ X', data=data).fit()
  model_Y_boot = smf.ols('Y ~ X + M', data=data).fit()
  a_boot = model_M_boot.params['X']
  b_boot = model_Y_boot.params['M']
  return a_boot * b_boot

n_bootstraps = 1000
indirect_effects = []

for _ in range(n_bootstraps):
    #Resample data with replacement
    resample = df.sample(frac=1, replace=True)
    indirect_effects.append(get_indirect_effect(resample))

#Calculate confidence interval from the bootstrapped indirect effects.
lower_ci = np.percentile(indirect_effects, 2.5)
upper_ci = np.percentile(indirect_effects, 97.5)

print(f"\nBootstrapped 95% Confidence Interval for Indirect Effect: ({lower_ci:.4f}, {upper_ci:.4f})")
```

**Explanation:**

1.  **Data:**  Create a Pandas DataFrame with columns for X, M, and Y.  *Replace the sample data with your actual data.*
2.  **Mediator Model (X -> M):** Fit a regression model predicting M from X.  The coefficient for X (a) represents the effect of X on M.
3.  **Outcome Model (X + M -> Y):** Fit a regression model predicting Y from both X and M. The coefficient for M (b) represents the effect of M on Y, *controlling for X*.  The coefficient for X (c') represents the direct effect of X on Y, *controlling for M*.
4.  **Calculations:** Calculate the indirect effect (a\*b), direct effect (c'), and total effect (estimated by regressing Y on X alone).
5.  **Bootstrapping:**
    * The `get_indirect_effect` function calculates the indirect effect from a given dataset.
    * The code then resamples the data with replacement `n_bootstraps` times, calculating the indirect effect for each resampled dataset.
    * Finally, it calculates a 95% confidence interval for the indirect effect using the percentile method.  If this confidence interval does not contain zero, it provides evidence that the indirect effect is statistically significant.

**Important Notes:**

*   The `statsmodels` package primarily uses frequentist statistics. For Bayesian mediation analysis, consider the `pymc` package.
*   The provided code demonstrates a basic approach.  More advanced methods, such as bootstrapping or Sobel tests, are needed for assessing the statistical significance of the indirect effect.  The example shows bootstrapping.
*   Carefully check the assumptions of mediation analysis.  Violations of these assumptions can lead to biased results.
*   Consider using causal identification techniques (e.g., instrumental variables, front-door criterion) to address unmeasured confounding.

## 4) Follow-up question

How does the interpretation of direct and indirect effects change when dealing with non-linear relationships between X, M, and Y? What specific statistical techniques or modeling approaches are better suited for these scenarios compared to standard linear regression?