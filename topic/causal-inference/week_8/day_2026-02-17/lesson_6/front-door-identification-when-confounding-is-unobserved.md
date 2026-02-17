---
title: "Front-Door Identification (When Confounding is Unobserved)"
date: "2026-02-17"
week: 8
lesson: 6
slug: "front-door-identification-when-confounding-is-unobserved"
---

# Topic: Front-Door Identification (When Confounding is Unobserved)

## 1) Formal definition (what is it, and how can we use it?)

Front-door identification is a causal inference technique used to identify the causal effect of a treatment *X* on an outcome *Y* when there is unobserved confounding between *X* and *Y*. It leverages a mediator *Z* on the causal pathway from *X* to *Y* to bypass the unobserved confounder.

**How it works:**

The front-door criterion relies on three key assumptions:

1.  *X* causally influences *Z* (i.e., *X* -> *Z*).
2.  *Z* causally influences *Y* (i.e., *Z* -> *Y*).
3.  All back-door paths from *Z* to *Y* are blocked (relative to *X*). This typically requires observing all confounders of *Z* and *Y* (denoted *W*).  Critically, there can still be unobserved confounding between *X* and *Y* itself.

**The Front-Door Formula:**

If the front-door criterion holds, the causal effect of *X* on *Y* is identifiable and can be calculated using the following formula:

P(y | do(x)) = Σ<sub>z</sub> P(z | x) Σ<sub>x'</sub> P(y | x', z) P(x')

Where:

*   P(y | do(x)) is the causal effect of setting *X* to *x* on *Y*.
*   P(z | x) is the probability of *Z* given *X*. This is identifiable from observational data.
*   P(y | x', z) is the probability of *Y* given *X* and *Z*. This is also identifiable from observational data.
*   P(x') is the marginal probability of *X*.
*   Σ<sub>z</sub>  and  Σ<sub>x'</sub> denote summation over all possible values of *Z* and *X'*, respectively. If *Z* and/or *X* are continuous, these become integrals.

**How we use it:**

The front-door criterion allows us to estimate a causal effect when direct identification of the causal effect is impossible due to unobserved confounding.  We essentially break the causal pathway *X* -> *Y* into two parts: *X* -> *Z* and *Z* -> *Y*, and estimate these components using observational data and the front-door assumptions. We then combine these estimates to obtain the overall causal effect.

## 2) Application scenario

Imagine studying the effect of a new advertising campaign (*X*) on product sales (*Y*).  There might be unobserved factors (e.g., overall economic conditions, brand loyalty) that influence both the advertising campaign's effectiveness and the product sales themselves. These are unobserved confounders.

However, you believe the advertising campaign primarily affects sales *through* increased customer awareness (*Z*). You can measure customer awareness, and you can also control for other factors that might influence both customer awareness and product sales, such as existing marketing efforts (*W*).

In this scenario:

*   *X*: Advertising campaign
*   *Y*: Product sales
*   *Z*: Customer awareness
*   Unobserved Confounder: Economic Conditions, Brand Loyalty
*   *W*: Existing marketing efforts

If we assume that *X* only affects *Y* *through* *Z*, and we have blocked all back-door paths from *Z* to *Y* by controlling for *W*, the front-door criterion is satisfied. We can then use the front-door formula to estimate the causal effect of the advertising campaign on product sales, even with the unobserved confounding.

## 3) Python method (if possible)

While there isn't a single function specifically named "frontdoor_adjustment" in common causal inference libraries, you can implement the front-door formula using existing tools from libraries like `causalinference`, `DoWhy`, or `pgmpy`. Here's an example using `pgmpy`:

```python
import numpy as np
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork

# Generate some example data (replace with your actual data)
np.random.seed(42)
n_samples = 1000
X = np.random.randint(0, 2, n_samples)  # Advertising Campaign (0 or 1)
U = np.random.normal(0, 1, n_samples)   # Unobserved Confounder
Z = X + 0.5 * U + np.random.normal(0, 0.5, n_samples) # Awareness
W = np.random.randint(0, 2, n_samples) # Existing Marketing
Y = 2 * Z + 0.3 * X + 0.7* W + U + np.random.normal(0, 1, n_samples) # Sales

data = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z, 'W':W})
data['X'] = data['X'].astype(int)
data['W'] = data['W'].astype(int)

# Discretize Z and Y for simplicity with MaximumLikelihoodEstimator (optional, but helps with estimation)
data['Z'] = pd.qcut(data['Z'], q=3, labels=False)
data['Y'] = pd.qcut(data['Y'], q=3, labels=False)


# Define the Bayesian Network structure reflecting the causal diagram
model = BayesianNetwork([('X', 'Z'), ('Z', 'Y'), ('X', 'Y'), ('W','Y')])


# Learn parameters from the data using Maximum Likelihood Estimation
model.fit(data, estimator=MaximumLikelihoodEstimator)


# Implement the Front-Door Formula
def frontdoor_adjustment(model, data, x_value):
    """
    Calculates the causal effect of X on Y using the front-door criterion.
    """
    y_values = data['Y'].unique()
    x_values = data['X'].unique()
    z_values = data['Z'].unique()

    total_effect = 0
    for z in z_values:
        p_z_given_x = model.predict_probability({'X': x_value})['Z'][z]


        inner_sum = 0
        for x_prime in x_values:
            p_y_given_xprime_z = model.predict_probability({'X': x_prime, 'Z': z, 'W':data['W']})['Y'] #Predict Y|X',Z,W (Note, W must be provided as evidence)
            p_x_prime = np.mean(data['X'] == x_prime)

            inner_sum += p_y_given_xprime_z * p_x_prime

        total_effect += p_z_given_x * inner_sum[y_values[0]] # Select the first value of y. This needs to be incorporated into an outer loop to get the full distribution.

    return total_effect

# Example usage:  Estimate the causal effect of setting X=1 vs X=0
effect_of_x_1 = frontdoor_adjustment(model, data, 1)
effect_of_x_0 = frontdoor_adjustment(model, data, 0)
causal_effect = effect_of_x_1 - effect_of_x_0

print(f"Causal Effect of X=1 vs X=0 on Y (approximately): {causal_effect}")
```

**Important Notes:**

*   This is a simplified example.  Real-world implementations would involve more robust estimation techniques, especially when dealing with continuous variables.  Propensity score weighting or g-computation could be more appropriate.
*   The BayesianNetwork approach provides a framework, but implementing the full front-door formula requires carefully extracting conditional probabilities from the fitted model.  This example uses `.predict_probability()` which assumes categorical/discrete variables for easier computation.  Discretization may introduce bias.
*  This code does *not* fully compute P(y|do(x)). It calculates the probability of the *first* Y value, given do(X=x), for both X=1 and X=0, then it returns the difference between these values. To obtain the full P(y|do(x)) distribution, you would need to loop through all values of 'y' and perform a similar calculation for each.

## 4) Follow-up question

How does the identifiability of the front-door criterion change if the mediator *Z* is also affected by unobserved confounders that also affect *Y*?  Specifically, what conditions are necessary for the front-door criterion to *still* hold in the presence of unobserved confounding between *Z* and *Y*?